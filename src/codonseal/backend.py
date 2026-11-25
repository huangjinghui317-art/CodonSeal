"""Local MLX backend. Generation is preserved from the original implementation."""
import hashlib
import numpy as np
from .core import (COLOR_STRENGTH, SEMANTIC_STRENGTH, TEMPERATURE, TOP_K, VARIANTS, LockState, color_features, color_table, pair_seed, polarity, probabilities, standardized, support)

class Decoder:
    def __init__(self, model_path, key):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        self.model, self.tokenizer = load(model_path)
        self.embedding = self.model.model.embed_tokens
        self.size = self.model.args.vocab_size
        self.key = key
        self.wrong = hashlib.sha256(b"CodonSeal wrong key\0" + key).digest()
        self.colors = color_table(key, self.size)
        self.wrong_colors = color_table(self.wrong, self.size)
        self.stops = set(self.tokenizer.eos_token_ids) | {151643}
        self.special = set(self.tokenizer.all_special_ids) | self.stops

    def prompt(self, text):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def top_candidates(self, inputs, cache):
        mx = self.mx
        hidden = self.model.model(mx.array(inputs, dtype=mx.int32), cache=cache)[:, -1, :]
        output = self.embedding.as_linear(hidden) if self.model.args.tie_word_embeddings else self.model.lm_head(hidden)
        output = output.astype(mx.float32)
        ids = mx.argpartition(-output, TOP_K - 1, axis=-1)[:, :TOP_K]
        values = mx.take_along_axis(output, ids, axis=-1)
        mx.eval(ids, values)
        ids, values = np.array(ids), np.array(values)
        order = np.lexsort((ids, -values), axis=1)
        return np.take_along_axis(ids, order, axis=1), np.take_along_axis(values, order, axis=1), output

    def cache(self, prefixes):
        from mlx_lm.models.cache import BatchKVCache
        length = max(map(len, prefixes))
        pads = [length - len(p) for p in prefixes]
        cache = [BatchKVCache(pads) for _ in self.model.model.layers]
        return [[0] * pad + p for p, pad in zip(prefixes, pads)], cache

    def features(self, ids, locks, base, wrong=False):
        active = ~np.isin(ids, list(self.special))
        colors = color_features(self.wrong_colors if wrong else self.colors, ids, locks)
        mx = self.mx
        vectors = self.embedding(mx.array(ids)).astype(mx.float32)
        vectors = vectors / mx.maximum(mx.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)
        length = max(map(len, locks))
        indices = mx.array([list(lock) + [0] * (length - len(lock)) for lock in locks])
        mask = mx.array([[1.] * len(lock) + [0.] * (length - len(lock)) for lock in locks])
        lock_vectors = self.embedding(indices).astype(mx.float32)
        lock_vectors = lock_vectors / mx.maximum(mx.linalg.norm(lock_vectors, axis=-1, keepdims=True), 1e-8)
        centers = (lock_vectors * mask[:, :, None]).sum(axis=1)
        centers = centers / mx.maximum(mx.linalg.norm(centers, axis=-1, keepdims=True), 1e-8)
        semantic = np.array((vectors * centers[:, None, :]).sum(axis=-1))
        signs = np.array([polarity(self.wrong if wrong else self.key, lock) for lock in locks])
        return (COLOR_STRENGTH * standardized(colors, base, active),
                SEMANTIC_STRENGTH * standardized(semantic * signs[:, None], base, active))

    def generate(self, jobs, max_tokens, seed):
        inputs, cache = self.cache([self.prompt(row["prompt"]) for row, _ in jobs])
        rng = [np.random.default_rng(pair_seed(seed, row["id"])) for row, _ in jobs]
        states = [LockState() for _ in jobs]
        tokens = [[] for _ in jobs]
        traces = [[] for _ in jobs]
        nll = [[] for _ in jobs]
        stops = ["length"] * len(jobs)
        finished = np.zeros(len(jobs), dtype=bool)
        for _ in range(max_tokens):
            ids, raw, output = self.top_candidates(inputs, cache)
            counts = support(raw)
            logits = raw / TEMPERATURE
            base = probabilities(logits)
            locks = [s.before(int(count), variant == "recent") if not finished[i] else ()
                     for i, (s, count, (_, variant)) in enumerate(zip(states, counts, jobs))]
            active = [i for i, (_, variant) in enumerate(jobs) if locks[i] and variant != "baseline"]
            if active:
                color, semantic = self.features(ids[active], [locks[i] for i in active], base[active])
                for j, i in enumerate(active):
                    variant = jobs[i][1]
                    if variant in ("color", "full", "recent"):
                        logits[i] += color[j]
                    if variant in ("semantic", "full", "recent"):
                        logits[i] += semantic[j]
            probs = probabilities(logits)
            sampled = [int(ids[i, min(TOP_K - 1, int(np.searchsorted(np.cumsum(p), rng[i].random())))])
                       for i, p in enumerate(probs)]
            mx = self.mx
            values = np.array(mx.logsumexp(output, axis=-1) -
                              mx.take_along_axis(output, mx.array(sampled)[:, None], axis=-1)[:, 0])
            for i, token in enumerate(sampled):
                if finished[i]:
                    continue
                if token in self.stops:
                    finished[i], stops[i] = True, "eos"
                    continue
                tokens[i].append(token)
                traces[i].append({"count": int(counts[i]), "lock": list(locks[i])})
                nll[i].append(float(values[i]))
                states[i].after(int(counts[i]), token)
            if finished.all():
                break
            inputs = [[token] for token in sampled]
        return [{"ids": ids, "text": self.tokenizer.decode(ids), "trace": trace, "nll": losses, "finish": stop}
                for ids, trace, losses, stop in zip(tokens, traces, nll, stops)]

    def replay(self, jobs):
        # A separate teacher-forced forward on the received text, including retokenization.
        inputs, cache = self.cache([prefix for prefix, _ in jobs])
        sequences = [seq for _, seq in jobs]
        states = [LockState() for _ in jobs]
        recent = [LockState() for _ in jobs]
        results = [{"scores": {v: 0. for v in VARIANTS[1:]},
                    "wrong_scores": {v: 0. for v in VARIANTS[1:]},
                    "events": {"lock": 0, "recent": 0}, "outside_top_k": 0,
                    "counts": [], "key_positions": []} for _ in jobs]
        for step in range(max(map(len, sequences), default=0)):
            ids, raw, _ = self.top_candidates(inputs, cache)
            counts = support(raw)
            logits = raw / TEMPERATURE
            base = probabilities(logits)
            positions = np.full(len(jobs), -1, dtype=int)
            observed = [0] * len(jobs)
            locks = [()] * len(jobs)
            controls = [()] * len(jobs)
            for i, seq in enumerate(sequences):
                if step >= len(seq):
                    continue
                token = observed[i] = seq[step]
                count = int(counts[i])
                locks[i] = states[i].before(count)
                controls[i] = recent[i].before(count, recent_control=True)
                states[i].after(count, token)
                recent[i].after(count, token)
                results[i]["counts"].append(count)
                hits = np.flatnonzero(ids[i] == token)
                if len(hits):
                    positions[i] = hits[0]
                else:
                    results[i]["outside_top_k"] += 1
            for name, features in (("lock", locks), ("recent", controls)):
                active = [i for i, lock in enumerate(features) if lock and positions[i] >= 0]
                if not active:
                    continue
                for i in active:
                    results[i]["events"][name] += 1
                    if name == "lock":
                        results[i]["key_positions"].append(step)
                for wrong in (False, True):
                    color, semantic = self.features(ids[active], [features[i] for i in active], base[active], wrong)
                    biases = {"recent": color + semantic} if name == "recent" else {
                        "color": color, "semantic": semantic, "full": color + semantic}
                    for variant, bias in biases.items():
                        normalizer = np.log(np.sum(base[active] * np.exp(bias), axis=1))
                        for j, i in enumerate(active):
                            llr = float(bias[j, positions[i]] - normalizer[j])
                            results[i]["wrong_scores" if wrong else "scores"][variant] += llr
            inputs = [[token] for token in observed]
        return results

class MLXBackend(Decoder):
    def replay_qa(self, record, body):
        """Independent teacher forcing; original full color + semantic LLR.

        Neither generation traces nor ground truth answers enter the detector.
        """
        mx = self.mx
        sequence = self.tokenizer.encode(record['text'], add_special_tokens=False)
        prefix = [151643] if body else self.prompt(record['prompt'])
        result = dict(score=0., wrong_score=0., events=0, outside=0, counts=[],
                      key_positions=[], retokenization_equal=sequence == record['ids'])
        if not sequence:
            return result
        hidden = self.model.model(mx.array([prefix + sequence[:-1]], dtype=mx.int32))[0, len(prefix)-1:, :]
        mx.eval(hidden)
        state = LockState()
        for start in range(0, len(sequence), 32):
            output = (self.embedding.as_linear(hidden[start:start+32]) if self.model.args.tie_word_embeddings
                      else self.model.lm_head(hidden[start:start+32])).astype(mx.float32)
            candidates = mx.argpartition(-output, TOP_K-1, axis=-1)[:, :TOP_K]
            values = mx.take_along_axis(output, candidates, axis=-1)
            mx.eval(candidates, values)
            candidates, raw = np.array(candidates), np.array(values)
            order = np.lexsort((candidates, -raw), axis=1)
            candidates = np.take_along_axis(candidates, order, axis=1)
            raw = np.take_along_axis(raw, order, axis=1)
            counts = support(raw)
            locks, observed = [], []
            for index, count in enumerate(counts):
                token = sequence[start+index]
                lock = state.before(int(count))
                state.after(int(count), token)
                match = np.flatnonzero(candidates[index] == token)
                result['outside'] += int(not len(match))
                result['counts'].append(int(count))
                locks.append(lock)
                observed.append(int(match[0]) if len(match) else -1)
            active = [i for i, lock in enumerate(locks) if lock and observed[i] >= 0]
            if active:
                base = probabilities(raw[active] / TEMPERATURE)
                for wrong in (False, True):
                    color, semantic = self.features(candidates[active], [locks[i] for i in active], base, wrong)
                    bias = color + semantic
                    normalizer = np.log(np.sum(base * np.exp(bias), axis=-1))
                    result['wrong_score' if wrong else 'score'] += sum(
                        float(bias[j, observed[i]] - normalizer[j]) for j, i in enumerate(active))
                result['events'] += len(active)
                result['key_positions'].extend(start+i for i in active)
        del hidden
        mx.clear_cache()
        return result
