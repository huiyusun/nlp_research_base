# test_word2vec.py
from word2vec_sgns import Word2VecSGNS

model = Word2VecSGNS.load("models/my_w2v")

print("Vocab size:", len(model.vocab.id2word))
print("Dim:", model.cfg.dim)


def show_neighbors(word, k=10):
    try:
        print(f"\nMost similar to '{word}':")
        for w, s in model.most_similar(word, topk=k):
            print(f"  {w:>20s}  {s:.4f}")
    except KeyError:
        print(f"'{word}' not in vocab.")


def show_analogy(a, b, c, k=5):
    try:
        print(f"\nAnalogy: {a} : {b} :: {c} : ?")
        for w, s in model.analogy(a, b, c, topk=k):
            print(f"  {w:>20s}  {s:.4f}")
    except KeyError as e:
        print(e)


# Try a few words relevant to your corpus
for w in ["king", "queen", "new", "york", "data", "model"]:
    show_neighbors(w, k=10)

# Classic analogy
show_analogy("man", "woman", "king", k=5)
