# CodonSeal

CodonSeal embeds a keyed signal in generated Python code. It seals a lock from a run of low-choice positions and uses it at the next eligible key position to bias sampling. Two variants share the lock construction and the frozen Qwen generator.

| Method | How it selects lock and key positions | Detection inputs |
|---|---|---|
| **CodonSeal** | A small WPN predictor estimates candidate richness and the expected next-token embedding from the two most recent answer tokens. The legacy Tiny model attends to a 16-slot sequence whose preceding 14 slots contain the fixed padding token. | Answer text, tokenizer, WPN weights, static token geometry and key. The original prompt and generator are unnecessary. |
| **CodonSeal-oracle** | The full generator provides candidate richness and token scores. The original semantic coefficient is 2.0. | Answer text, original prompt, original model and key. The detector uses the prompt-conditioned log-likelihood ratio. |
| **Unmarked** | The same generator samples without watermark bias. | Serves as the paired negative set for both detection methods. |

## HumanEval results

<!-- RESULTS:START -->
### All five seeds

Mean ± sample standard deviation across the five generation seeds.

| Method | AUROC | TPR at ≤5% FPR | Actual FPR | Pass@1 | Self-PPL |
|---|---:|---:|---:|---:|---:|
| Unmarked | — | — | — | 50.98% ± 2.46% | 1.1159 ± 0.0080 |
| CodonSeal-oracle | 0.9208 ± 0.0139 | 65.37% ± 5.46% | 4.88% ± 0.00% | 49.63% ± 2.35% | 1.1359 ± 0.0083 |
| CodonSeal | 0.7207 ± 0.0052 | 27.32% ± 2.77% | 4.76% ± 0.27% | 47.80% ± 1.96% | 1.1777 ± 0.0086 |

### Per-seed results

| Seed | CodonSeal AUROC | Oracle AUROC | CodonSeal Pass@1 | Oracle Pass@1 | Unmarked Pass@1 |
|---:|---:|---:|---:|---:|---:|
| 2026 | 0.7243 | 0.9299 | 49.39% | 46.34% | 51.83% |
| 2027 | 0.7235 | 0.8982 | 45.12% | 49.39% | 48.78% |
| 2028 | 0.7180 | 0.9191 | 46.95% | 52.44% | 50.00% |
| 2029 | 0.7248 | 0.9337 | 47.56% | 51.22% | 49.39% |
| 2030 | 0.7128 | 0.9232 | 50.00% | 48.78% | 54.88% |

<!-- RESULTS:END -->

See [per-seed summary](results/merix/summary.csv), [aggregate metrics](results/summary.json) and [per-task records for all five seeds](results/merix/). The per-task CSVs contain raw detector scores, sandbox pass flags, answer token counts and summed model NLL. These are sufficient to recompute the reported metrics.

### Protocol

- All 164 official HumanEval problems appear in each method and each seed. Each method generates one answer per question per seed. The methods share Qwen2.5-Coder-1.5B-Instruct (MLX 4-bit), temperature 0.8, an original pre-bias top-64 candidate set, a 256-token cap and the same per-question random stream definition.
- Generation seeds are 2026–2030. WPN training uses seed 42 and one checkpoint shared by all five generation seeds. The public benchmark key is derived from the label in [the configuration](configs/protocol.json). Use an external key file for private experiments.
- The WPN trains on 192 Open-R1 problems and validates on 48 other problems. The frozen [split file](configs/openr1_splits.json) also records the 64 external development problems on which the detector aggregation was selected. WPN fitting uses 13,577 distinct training token positions and 2,983 validation positions. The checkpoint minimizes the validation objective. The teacher conditions on the Open-R1 problem when constructing labels. The WPN consumes two answer tokens during generation and detection.
- HumanEval provides the final evaluation examples. The archived data preparation screened the Open-R1 candidates for high statement overlap and normalized AST overlap with HumanEval. The screen cannot guarantee identification of paraphrases or overlap in the base model's pretraining.
- AUROC uses raw scores, and curves group ties. TPR at 5% FPR is the maximum attainable empirical TPR with FPR at or below 5%. The table includes the observed FPR. Pass@1 is the fraction of submitted answers passing official tests in the OS sandbox. Self-PPL is computed from the original Qwen probabilities on generated answer tokens, conditioned on the prompt.

## Reproduce the experiments

The published implementation targets **Apple silicon macOS** with Metal, Python 3.11 and `/usr/bin/sandbox-exec`. Evaluation executes generated code only in the provided restricted sandbox. Training and evaluation store large assets and state in the git-ignored `local/` directory. Run the following from the repository root with an editable install.

```bash
git clone https://github.com/huangjinghui317-art/CodonSeal.git
cd CodonSeal
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install -e . --no-deps --no-build-isolation
```

### Download and train

```bash
python -m codonseal.assets --with-training
python -m codonseal.train
```

The asset command verifies pinned SHA-256 hashes for the Qwen model, the HumanEval file and four Open-R1 shards. The model and training downloads total about 1.4 GB. Training extracts the pinned problem IDs, rebuilds the frozen vocabulary geometry, and obtains candidate counts and expected embeddings by teacher forcing the reference Python answers. It saves the best WPN weights to `local/wpn/tiny2/best.safetensors` and geometry to `local/wpn/geometry.safetensors`. Its state and training corpus hashes support resuming in the same output directory. Results can vary across hardware and numerical library builds. The archived geometry and checkpoint hashes are in [the experiment configuration](configs/protocol.json).

An independent rebuild on the benchmark host reproduced both archived SHA-256 hashes. The standalone CodonSeal demo also reproduced a saved score using a directory with only the tokenizer, WPN and geometry.

### Run the benchmark and export the tables

```bash
python -m codonseal.evaluate
python -m codonseal.export --run local/run --output results --update-readme
```

The evaluation command resumes committed results when the same configuration and assets are used. Open `local/run/progress.html` for progress. A partial run with a single seed can be started using `python -m codonseal.evaluate --seeds 2026 --out local/smoke`. The public exporter expects all five specified seeds and the public benchmark key. The exporter writes the summary and a per-task CSV for each of the five seeds to `results/merix`. Per-seed raw generation and grading state remain local.

To use a separately generated private key, create a binary key outside the repository and pass `--key-file /path/to/key` with a new `--out` directory.

### Generate or detect an individual answer

Create a UTF-8 `prompt.txt` containing the exact prompt. Then run:

```bash
python scripts/demo.py generate --method CodonSeal --prompt prompt.txt > answer.txt
python scripts/demo.py detect --method CodonSeal --answer answer.txt
python scripts/demo.py generate --method CodonSeal-oracle --prompt prompt.txt > oracle.txt
python scripts/demo.py detect --method CodonSeal-oracle --prompt prompt.txt --answer oracle.txt
python scripts/demo.py generate --method unmarked --prompt prompt.txt > unmarked.txt
```

The detector returns a real-valued score. Calibrate a threshold on data from the intended deployment distribution before converting scores to decisions. The public experiment key is suitable only for benchmarking.

## Files and license

- `src/codonseal/core.py` defines the lock/key state and keyed features.
- `src/codonseal/backend.py` and `src/codonseal/watermark.py` implement generation and both detectors.
- `src/codonseal/train.py` rebuilds the frozen two-token WPN configuration.
- `src/codonseal/evaluate.py` and `src/codonseal/export.py` produce the benchmark and public tables.
- `configs/` records source revisions, hashes and the fixed experimental design.
- `results/` contains derived public metrics.

The source code is released under the [MIT License](LICENSE). See [NOTICE](NOTICE.md) for third-party sources. The paper-oriented experiments use a single generator, a single benchmark, one WPN training seed and no attacks.
