#!/usr/bin/env python3
"""
Compute unsmoothed unigram and bigram probabilities from a training corpus.
- Each line in the input file is a sentence.
- This script automatically adds <s> at the start and </s> at the end of each sentence for counting.
"""

from collections import Counter, defaultdict
import sys
import re
import spacy  # type: ignore

_NLP = spacy.blank("en")  # fast, no model download required
_SPACY_TOKENIZER = _NLP.tokenizer


def tokenize(line: str):
    doc = _SPACY_TOKENIZER(line)
    return [t.text for t in doc if not t.is_space]


def train_ngrams(corpus_file, add_start=True, add_end=True):
    unigram_counts = Counter()
    bigram_counts = Counter()

    with open(corpus_file, "r") as f:
        for line in f:
            words = tokenize(line)
            if not words:
                continue

            # optionally add sentence boundary tokens
            tokens = words
            if add_start:
                tokens = ['<s>'] + tokens
            if add_end:
                tokens = tokens + ['</s>']

            # update unigrams
            unigram_counts.update(tokens)

            # update bigrams (consecutive pairs)
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                bigram_counts[bigram] += 1

    return unigram_counts, bigram_counts


def compute_probs(unigram_counts, bigram_counts):
    total_unigrams = sum(unigram_counts.values())
    unigram_probs = {w: c / total_unigrams for w, c in unigram_counts.items()}

    # For each history, sum outgoing bigrams to get denominator
    history_counts = defaultdict(int)
    for (w1, w2), c in bigram_counts.items():
        history_counts[w1] += c

    bigram_probs = {}
    for (w1, w2), c in bigram_counts.items():
        bigram_probs[(w1, w2)] = c / history_counts[w1]

    return unigram_probs, bigram_probs


def print_probs(unigram_probs, bigram_probs):
    print("=== Unigram probabilities ===")
    for w, p in sorted(unigram_probs.items()):
        print(f"P({w}) = {p:.4f}")

    print("\n=== Bigram probabilities ===")
    for (w1, w2), p in sorted(bigram_probs.items()):
        print(f"P({w2}|{w1}) = {p:.4f}")


def main():
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"
    unigram_counts, bigram_counts = train_ngrams(corpus_path, add_start=True, add_end=True)
    unigram_probs, bigram_probs = compute_probs(unigram_counts, bigram_counts)
    print_probs(unigram_probs, bigram_probs)


if __name__ == "__main__":
    main()
