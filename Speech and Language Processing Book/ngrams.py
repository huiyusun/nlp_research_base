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
import random
import argparse
import math

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


def perplexity_unigram(unigram_probs, test_file, add_start=True, add_end=True):
    """Compute unsmoothed unigram perplexity on a test file.
    Returns math.inf if any token has zero probability under the model.
    """
    total_tokens = 0
    log_prob_sum = 0.0
    with open(test_file, "r") as f:
        for line in f:
            words = tokenize(line)
            if not words:
                continue
            tokens = words
            if add_start:
                tokens = ['<s>'] + tokens
            if add_end:
                tokens = tokens + ['</s>']
            for w in tokens:
                p = unigram_probs.get(w, 0.0)
                if p <= 0.0:
                    return math.inf
                log_prob_sum += math.log(p)
                total_tokens += 1
    if total_tokens == 0:
        return math.inf
    return math.exp(-log_prob_sum / total_tokens)


def perplexity_bigram(bigram_probs, test_file, add_start=True, add_end=True):
    """Compute unsmoothed bigram perplexity on a test file.
    Returns math.inf if any bigram has zero probability under the model.
    """
    total_bigrams = 0
    log_prob_sum = 0.0
    with open(test_file, "r") as f:
        for line in f:
            words = tokenize(line)
            if not words:
                continue
            tokens = words
            if add_start:
                tokens = ['<s>'] + tokens
            if add_end:
                tokens = tokens + ['</s>']
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i + 1]
                p = bigram_probs.get((w1, w2), 0.0)
                if p <= 0.0:
                    return math.inf
                log_prob_sum += math.log(p)
                total_bigrams += 1
    if total_bigrams == 0:
        return math.inf
    return math.exp(-log_prob_sum / total_bigrams)


def generate_sentence(bigram_probs, max_length=30):
    # Build a mapping from history to list of (token, prob)
    transitions = defaultdict(list)
    for (w1, w2), p in bigram_probs.items():
        transitions[w1].append((w2, p))
    sentence = []
    current = '<s>'
    for _ in range(max_length):
        if current not in transitions:
            break
        next_tokens, probs = zip(*transitions[current])
        next_word = random.choices(next_tokens, probs, k=1)[0]
        if next_word == '</s>':
            break
        sentence.append(next_word)
        current = next_word
    return ' '.join(sentence)


def main():
    parser = argparse.ArgumentParser(description="Train MLE unigram/bigram models; print probs and optionally compute perplexity.")
    parser.add_argument("corpus", nargs="?", default="corpus.txt", help="Training corpus (one sentence per line)")
    parser.add_argument("--no-start", action="store_true", help="Do not add <s> to sentence start")
    parser.add_argument("--no-end", action="store_true", help="Do not add </s> to sentence end")
    parser.add_argument("--perplexity", metavar="TEST_FILE", help="Compute perplexity on TEST_FILE")
    parser.add_argument("--model", choices=["unigram", "bigram"], default="bigram", help="Model to use for perplexity computation")
    parser.add_argument("--no-generate", action="store_true", help="Skip random sentence generation")
    args = parser.parse_args()

    add_start = not args.no_start
    add_end = not args.no_end

    unigram_counts, bigram_counts = train_ngrams(args.corpus, add_start=add_start, add_end=add_end)
    unigram_probs, bigram_probs = compute_probs(unigram_counts, bigram_counts)

    # Always show probabilities
    print_probs(unigram_probs, bigram_probs)

    # Optional perplexity
    if args.perplexity:
        if args.model == "unigram":
            ppl = perplexity_unigram(unigram_probs, args.perplexity, add_start=add_start, add_end=add_end)
        else:
            ppl = perplexity_bigram(bigram_probs, args.perplexity, add_start=add_start, add_end=add_end)
        if math.isinf(ppl):
            print(f"\n=== Perplexity ({args.model}) on {args.perplexity} ===\nPerplexity: inf (zero-probability events encountered; unsmoothed model)")
        else:
            print(f"\n=== Perplexity ({args.model}) on {args.perplexity} ===\nPerplexity: {ppl:.4f}")

    # Optional generation
    if not args.no_generate:
        print("\n=== Randomly generated sentences ===")
        for _ in range(5):
            print(generate_sentence(bigram_probs))


if __name__ == "__main__":
    main()
