"""CodonSeal's causal lock state and candidate scoring."""
from dataclasses import dataclass, field
import hashlib
import hmac
import numpy as np

TOP_K = 64
DELTA = 2.5
LOW = 2
HIGH = 4
MIN_LOCK = 2
MAX_LOCK = 8
LOCK_TTL = 12
TEMPERATURE = .8
COLOR_STRENGTH = 1.
SEMANTIC_STRENGTH = 1.
VARIANTS = ("baseline", "color", "semantic", "full", "recent")


@dataclass
class LockState:
    low_run: list = field(default_factory=list)
    recent: list = field(default_factory=list)
    pending: tuple = ()
    sealed_at: int = -1
    position: int = 0

    def before(self, count, recent_control=False):
        if count > LOW:
            if len(self.low_run) >= MIN_LOCK:
                self.pending = tuple(self.low_run)
                self.sealed_at = self.position
            self.low_run.clear()
        if self.position - self.sealed_at > LOCK_TTL:
            self.pending = ()
        if count < HIGH:
            return ()
        lock = self.pending
        if recent_control and lock:
            lock = tuple(self.recent[-4:])
        self.pending = ()
        return lock

    def after(self, count, token):
        if count <= LOW:
            self.low_run = (self.low_run + [token])[-MAX_LOCK:]
        self.recent = (self.recent + [token])[-4:]
        self.position += 1


def color_table(key, size):
    order = sorted(range(size), key=lambda token: hmac.digest(key, token.to_bytes(4, "little"), "sha256"))
    colors = np.empty(size, dtype=np.int8)
    colors[order] = np.arange(size) % 3
    return colors


def polarity(key, lock):
    message = b"semantic:" + np.asarray(lock, dtype="<u4").tobytes()
    return 1 if hmac.digest(key, message, "sha256")[0] & 1 else -1


def support(logits):
    return np.sum(logits >= logits[:, :1] - DELTA, axis=1)


def probabilities(logits):
    unnormalized = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return unnormalized / unnormalized.sum(axis=-1, keepdims=True)


def standardized(values, base, active):
    center = np.sum(values * base, axis=1, keepdims=True)
    scale = np.sqrt(np.sum((values - center) ** 2 * base, axis=1, keepdims=True))
    # Floor prevents amplifying near-constant features to large biases.
    return np.clip((values - center) / np.maximum(scale, .08), -2., 2.) * active


def color_features(table, candidates, locks):
    return np.asarray([-np.bincount(table[list(lock)], minlength=3)[table[ids]] / len(lock)
                       for ids, lock in zip(candidates, locks)])


def pair_seed(seed, identifier):
    payload = f"{seed}:{identifier}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
