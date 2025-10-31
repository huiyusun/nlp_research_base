# word2vec_sgns.py
# From-scratch Word2Vec (Skip-Gram with Negative Sampling) in NumPy
# No external ML libs required.

from __future__ import annotations
import math
import random
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Iterable

import numpy as np

# Optional: spaCy tokenizer
try:
    import spacy

    _HAVE_SPACY = True
except Exception:
    _HAVE_SPACY = False


# --------------------------
# Tokenizers
# --------------------------
def simple_tokenize(text: str) -> List[str]:
    # Minimal tokenizer; replace with your own for better results.
    # Lowercase, keep letters/numbers and split on whitespace.
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split()


# --- spaCy tokenizer (preferable to the simple one) ---
_SPACY_NLP = None


def _load_spacy_pipeline(model_name: str = "en_core_web_sm"):
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    if not _HAVE_SPACY:
        raise RuntimeError("spaCy is not installed. Run: pip install spacy && python -m spacy download en_core_web_sm")
    try:
        # Light pipeline: we only need the tokenizer. Disable heavy components.
        nlp = spacy.load(model_name, disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"])
    except Exception:
        # Fallback to a blank English tokenizer if the model isn't installed
        nlp = spacy.blank("en")
    _SPACY_NLP = nlp
    return _SPACY_NLP


def spacy_tokenize(text: str, model_name: str = "en_core_web_sm") -> List[str]:
    """
    Tokenize with spaCy. Keeps punctuation as separate tokens and preserves
    useful tokens like numbers and contractions better than the simple tokenizer.
    """
    nlp = _load_spacy_pipeline(model_name)
    # Use the tokenizer directly for speed; no need to create Doc via nlp(text) with disabled pipes.
    doc = nlp.make_doc(text)
    return [t.text.lower() for t in doc if not t.is_space]


# --------------------------
# Dataset + Vocab
# --------------------------
@dataclass
class Vocab:
    word2id: Dict[str, int]
    id2word: List[str]
    counts: np.ndarray  # frequency per id
    total_tokens: int


def build_vocab(tokens: List[str], min_count: int = 5) -> Vocab:
    from collections import Counter
    cnt = Counter(tokens)
    # Filter by min_count
    words = [w for w, c in cnt.items() if c >= min_count]
    # Guarantee at least something
    if not words:
        # fall back to top-k if everything filtered
        words = [w for w, _ in cnt.most_common(30000)]
    words.sort(key=lambda w: (-cnt[w], w))
    word2id = {w: i for i, w in enumerate(words)}
    id2word = words
    counts = np.array([cnt[w] for w in id2word], dtype=np.int64)
    total_tokens = sum(cnt.values())
    return Vocab(word2id, id2word, counts, total_tokens)


def corpus_to_ids(tokens: List[str], vocab: Vocab) -> List[int]:
    return [vocab.word2id[w] for w in tokens if w in vocab.word2id]


# --------------------------
# Subsampling frequent words (Mikolov et al.)
# keep_prob(w) = (sqrt(t/f) + t/f)
# --------------------------
def subsample(corpus_ids: List[int], counts: np.ndarray, t: float = 1e-5) -> List[int]:
    total = counts.sum()
    freqs = counts / total
    keep_prob = (np.sqrt(t / np.maximum(freqs, 1e-12)) + t / np.maximum(freqs, 1e-12))
    keep_prob = np.clip(keep_prob, 0.0, 1.0)
    rng = np.random.default_rng(42)
    return [w for w in corpus_ids if rng.random() < keep_prob[w]]


# --------------------------
# Negative sampling distribution: unigram^(3/4)
# We draw via inverse-CDF sampling on cumulative probabilities.
# --------------------------
class NegativeSampler:
    def __init__(self, counts: np.ndarray, power: float = 0.75):
        prob = counts.astype(np.float64) ** power
        prob /= prob.sum()
        self.cum = np.cumsum(prob)
        self.rng = np.random.default_rng(123)

    def draw(self, k: int) -> np.ndarray:
        u = self.rng.random(k)
        return np.searchsorted(self.cum, u, side="left")

    def draw_matrix(self, batch_size: int, k: int) -> np.ndarray:
        # shape: [batch_size, k]
        u = self.rng.random((batch_size, k))
        return np.searchsorted(self.cum, u, side="left")


# --------------------------
# Training pairs generator (Skip-gram)
# --------------------------
def generate_skipgram_pairs(
        corpus_ids: List[int],
        window: int = 5,
        dynamic_window: bool = True,
        seed: int = 7
) -> Iterable[Tuple[int, int]]:
    rng = random.Random(seed)
    n = len(corpus_ids)
    for i, center in enumerate(corpus_ids):
        w = rng.randint(1, window) if dynamic_window else window
        left = max(0, i - w)
        right = min(n, i + w + 1)
        for j in range(left, right):
            if j == i:
                continue
            yield center, corpus_ids[j]


# --------------------------
# Word2Vec Model (SGNS) with NumPy
# --------------------------
@dataclass
class W2VConfig:
    dim: int = 300
    window: int = 5
    negative_k: int = 5
    min_count: int = 5
    subsample_t: float = 1e-5
    epochs: int = 2
    lr: float = 0.025
    lr_min: float = 0.0001
    seed: int = 1
    batch_size: int = 1024
    dynamic_window: bool = True
    ns_power: float = 0.75


class Word2VecSGNS:
    def __init__(self, vocab: Vocab, cfg: W2VConfig):
        self.vocab = vocab
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        V, D = len(vocab.id2word), cfg.dim
        # Input (center) embeddings and Output (context) embeddings
        self.W_in = (rng.random((V, D)) - 0.5) / D
        self.W_out = np.zeros((V, D), dtype=np.float64)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # numerically stable sigmoid
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        neg = ~pos
        out[pos] = 1 / (1 + np.exp(-x[pos]))
        expx = np.exp(x[neg])
        out[neg] = expx / (1 + expx)
        return out

    def train(self, corpus_ids: List[int]) -> None:
        # Subsample frequent words for efficiency
        if self.cfg.subsample_t > 0:
            corpus_ids = subsample(corpus_ids, self.vocab.counts, t=self.cfg.subsample_t)

        # Negative sampler
        ns = NegativeSampler(self.vocab.counts, power=self.cfg.ns_power)

        # Precompute all pairs (you can stream instead for large corpora)
        pairs = list(generate_skipgram_pairs(corpus_ids, self.cfg.window, self.cfg.dynamic_window, seed=self.cfg.seed))
        total_pairs = len(pairs)
        if total_pairs == 0:
            print("No training pairs; check min_count/subsampling/window.")
            return

        # Training
        lr0 = self.cfg.lr
        lr_min = self.cfg.lr_min
        batch_size = self.cfg.batch_size
        neg_k = self.cfg.negative_k

        order = np.arange(total_pairs)
        rng = np.random.default_rng(self.cfg.seed)

        for epoch in range(1, self.cfg.epochs + 1):
            rng.shuffle(order)
            # Linear LR decay per epoch pass
            for start in range(0, total_pairs, batch_size):
                end = min(start + batch_size, total_pairs)
                idx = order[start:end]
                centers = np.array([pairs[i][0] for i in idx], dtype=np.int64)
                contexts = np.array([pairs[i][1] for i in idx], dtype=np.int64)

                # Draw negatives (B, K)
                negs = ns.draw_matrix(len(centers), neg_k)

                # Gather embeddings
                v_c = self.W_in[centers]  # (B, D)
                v_o = self.W_out[contexts]  # (B, D)
                v_n = self.W_out[negs]  # (B, K, D)

                # Positive score and grad
                pos_score = np.sum(v_c * v_o, axis=1)  # (B,)
                pos_sig = self._sigmoid(pos_score)  # (B,)
                # Gradients w.r.t v_c and v_o for positive
                # dL/d(pos_score) = (sigma - 1)
                g_pos = (pos_sig - 1.0).reshape(-1, 1)  # (B,1)

                # Negative scores and grads
                neg_score = np.einsum("bd,bkd->bk", v_c, v_n)  # (B, K)
                neg_sig = self._sigmoid(-neg_score)  # (B, K)
                # dL/d(-score) = (sigma), so dL/d(score) = -sigma
                g_neg = -neg_sig  # (B, K)

                # Learning rate schedule (min floor)
                progress = (epoch - 1) + (start / total_pairs)
                max_progress = max(self.cfg.epochs - 1, 1e-9)
                lr = max(lr_min, lr0 * (1.0 - progress / (self.cfg.epochs)))

                # Update v_c (input embeddings)
                # grad from positive: g_pos * v_o
                grad_vc_pos = g_pos * v_o  # (B, D)
                # grad from negatives: sum_k g_neg[:,k] * v_n[:,k,:]
                grad_vc_neg = np.einsum("bk,bkd->bd", g_neg, v_n)  # (B, D)
                grad_vc = grad_vc_pos + grad_vc_neg

                # Update W_in
                # We need to accumulate per index because centers may repeat within batch
                np.add.at(self.W_in, centers, -lr * grad_vc)

                # Update v_o (output embeddings for positives)
                grad_vo = g_pos * v_c
                np.add.at(self.W_out, contexts, -lr * grad_vo)

                # Update v_n (output embeddings for negatives)
                # grad per negative word: g_neg[:,k] * v_c
                grad_vn = g_neg[..., None] * v_c[:, None, :]  # (B, K, D)
                # Scatter-add to W_out for negative indices
                for k in range(neg_k):
                    np.add.at(self.W_out, negs[:, k], -lr * grad_vn[:, k, :])

            print(f"Epoch {epoch}/{self.cfg.epochs} done. LR now ~ {lr:.6f}")

    # --------------------------
    # Inference helpers
    # --------------------------
    @property
    def embeddings(self) -> np.ndarray:
        # Many implementations return input vectors as final word vectors
        return self.W_in

    def word_vector(self, word: str) -> np.ndarray:
        return self.embeddings[self.vocab.word2id[word]]

    def most_similar(self, word: str, topk: int = 10, exclude_self: bool = True) -> List[Tuple[str, float]]:
        if word not in self.vocab.word2id:
            raise KeyError(f"'{word}' not in vocab.")
        wv = self.word_vector(word)
        E = self.embeddings
        # cosine similarity
        denom = np.linalg.norm(E, axis=1) * (np.linalg.norm(wv) + 1e-12)
        sims = (E @ wv) / (denom + 1e-12)
        idx = np.argsort(-sims)
        out = []
        for i in idx:
            if exclude_self and self.vocab.id2word[i] == word:
                continue
            out.append((self.vocab.id2word[i], float(sims[i])))
            if len(out) >= topk:
                break
        return out

    def analogy(self, a: str, b: str, c: str, topk: int = 5) -> List[Tuple[str, float]]:
        # Solve a : b :: c : ?
        for w in (a, b, c):
            if w not in self.vocab.word2id:
                raise KeyError(f"'{w}' not in vocab.")
        va, vb, vc = self.word_vector(a), self.word_vector(b), self.word_vector(c)
        query = vb - va + vc
        E = self.embeddings
        denom = np.linalg.norm(E, axis=1) * (np.linalg.norm(query) + 1e-12)
        sims = (E @ query) / (denom + 1e-12)
        # Exclude a,b,c
        ex = {self.vocab.word2id[w] for w in (a, b, c)}
        sims[list(ex)] = -1e9
        idx = np.argsort(-sims)[:topk]
        return [(self.vocab.id2word[i], float(sims[i])) for i in idx]

    # --------------------------
    # Save / Load
    # --------------------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "W_in.npy"), self.W_in)
        np.save(os.path.join(path, "W_out.npy"), self.W_out)
        with open(os.path.join(path, "vocab.json"), "w") as f:
            json.dump({
                "id2word": self.vocab.id2word,
                "counts": self.vocab.counts.tolist(),
                "total_tokens": self.vocab.total_tokens
            }, f)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(asdict(self.cfg), f)
        print(f"Saved model to {path}")

    @staticmethod
    def load(path: str) -> "Word2VecSGNS":
        W_in = np.load(os.path.join(path, "W_in.npy"))
        W_out = np.load(os.path.join(path, "W_out.npy"))
        with open(os.path.join(path, "vocab.json")) as f:
            vj = json.load(f)
        vocab = Vocab(
            word2id={w: i for i, w in enumerate(vj["id2word"])},
            id2word=vj["id2word"],
            counts=np.array(vj["counts"], dtype=np.int64),
            total_tokens=int(vj["total_tokens"])
        )
        with open(os.path.join(path, "config.json")) as f:
            cfg = W2VConfig(**json.load(f))
        model = Word2VecSGNS(vocab, cfg)
        model.W_in = W_in
        model.W_out = W_out
        return model


# --------------------------
# Demo / CLI
# --------------------------
DEMO_TEXT = """
King and queen rule the kingdom. The king is a man. The queen is a woman.
Paris is the capital of France. Berlin is the capital of Germany.
A man and a woman can be leaders. The kingdom respects the king and the queen.
"""


def load_corpus_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Word2Vec SGNS from scratch (NumPy).")
    parser.add_argument("--text", type=str, default="", help="Path to a raw text file (UTF-8). If empty, use demo text.")
    parser.add_argument("--tokenizer", type=str, default="simple", choices=["simple", "spacy"],
                        help="Which tokenizer to use. 'spacy' generally yields better tokenization.")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm", help="spaCy model name (if --tokenizer=spacy).")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--neg", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--lr_min", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--subsample_t", type=float, default=0.0, help="Set to 1e-5 for large corpora.")
    parser.add_argument("--save_dir", type=str, default="", help="Optional directory to save model.")
    args = parser.parse_args()

    text = load_corpus_from_file(args.text) if args.text else DEMO_TEXT
    # Choose tokenizer
    if args.tokenizer == "spacy":
        if not _HAVE_SPACY:
            raise RuntimeError("Asked for --tokenizer=spacy but spaCy is not installed. Install with: pip install spacy")
        tokens = spacy_tokenize(text, model_name=args.spacy_model)
    else:
        tokens = simple_tokenize(text)
    vocab = build_vocab(tokens, min_count=args.min_count)
    corpus_ids = corpus_to_ids(tokens, vocab)

    cfg = W2VConfig(
        dim=args.dim,
        window=args.window,
        negative_k=args.neg,
        min_count=args.min_count,
        subsample_t=args.subsample_t,
        epochs=args.epochs,
        lr=args.lr,
        lr_min=args.lr_min,
        batch_size=args.batch_size
    )

    print(f"Vocab size: {len(vocab.id2word)} | Tokens: {len(tokens)} | Trainable tokens: {len(corpus_ids)}")
    model = Word2VecSGNS(vocab, cfg)
    model.train(corpus_ids)

    # Quick sanity checks
    probe_words = [w for w in ["king", "queen", "man", "woman", "paris", "france"] if w in vocab.word2id]
    for w in probe_words:
        sims = model.most_similar(w, topk=5)
        print(f"\nMost similar to '{w}':")
        for s, sc in sims:
            print(f"  {s:>10s}  {sc: .4f}")

    if all(w in vocab.word2id for w in ["king", "man", "woman"]):
        print("\nAnalogy: king - man + woman ≈ ?")
        print(model.analogy("man", "woman", "king", topk=3))

    if args.save_dir:
        model.save(args.save_dir)


if __name__ == "__main__":
    main()
