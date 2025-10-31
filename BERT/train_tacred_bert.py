# file: train_tacred_bert.py
import json, os, random
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

import evaluate
from sklearn.metrics import f1_score, precision_recall_fscore_support

import torch
from torch.utils.data import Dataset

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


# -------------------------
# 1) TACRED-style loader
# -------------------------
# Expected example structure (per TACRED):
# {
#   "tokens": ["Barack","Obama","was","born","in","Hawaii","."],
#   "h": {"name": "Barack Obama", "pos": [0, 1], "type": "PERSON"},
#   "t": {"name": "Hawaii", "pos": [5, 5], "type": "LOCATION"},
#   "relation": "per:place_of_birth"
# }
#
# Sometimes keys are 'token' vs 'tokens'; 'subj'/'obj' vs 'h'/'t'. We support both.

def _norm_keys(ex):
    # Normalize to tokens, h, t, relation
    if "tokens" not in ex and "token" in ex:
        ex["tokens"] = ex["token"]
    if "h" not in ex and "subj" in ex:
        ex["h"] = ex["subj"]
    if "t" not in ex and "obj" in ex:
        ex["t"] = ex["obj"]
    return ex


def read_tacred(path: str) -> List[Dict[str, Any]]:
    data = json.load(open(path, "r"))
    data = [_norm_keys(ex) for ex in data]
    return data


# -------------------------
# 2) Entity-marking formatter
# -------------------------
# Strong baseline: wrap subject/object spans with special tokens.
# We’ll add TYPED markers (include entity type) — often helps on TACRED.
# Example: [E1:PERSON] Barack Obama [/E1] ... [E2:LOCATION] Hawaii [/E2]
#
# If you prefer untyped markers, set TYPED=False below.

TYPED = True
UNTYPED_MARKERS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]


def mark_entities(tokens, h_span, t_span, h_type=None, t_type=None):
    # spans are [start, end] inclusive indices over tokens
    # ensure subject comes before object when inserting to keep indices valid
    (s1, e1) = (h_span[0], h_span[1])
    (s2, e2) = (t_span[0], t_span[1])

    # order by start index descending so we insert safely
    pairs = sorted(
        [("H", s1, e1, h_type), ("T", s2, e2, t_type)],
        key=lambda x: x[1],
        reverse=True,
    )
    toks = tokens[:]
    for tag, s, e, typ in pairs:
        if TYPED:
            start_tok = f"[E1:{typ}]" if tag == "H" else f"[E2:{typ}]"
        else:
            start_tok = "[E1]" if tag == "H" else "[E2]"
        end_tok = "[/E1]" if tag == "H" else "[/E2]"
        toks = toks[:s] + [start_tok] + toks[s:e + 1] + [end_tok] + toks[e + 1:]
    return " ".join(toks)


# -------------------------
# 3) Dataset wrapper
# -------------------------
class TacredREDataset(Dataset):
    def __init__(self, data, tokenizer, label2id):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.examples = []
        for ex in data:
            tokens = ex["tokens"]
            # POS may be [start, end], sometimes stored in "pos" or "indices"
            h_span = ex["h"].get("pos") or ex["h"].get("indices")
            t_span = ex["t"].get("pos") or ex["t"].get("indices")
            h_type = ex["h"].get("type", "ENT")
            t_type = ex["t"].get("type", "ENT")

            text = mark_entities(tokens, h_span, t_span, h_type, t_type)
            self.examples.append({
                "text": text,
                "label": ex["relation"]
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        enc = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=256,
            padding=False,
            return_tensors=None,
        )
        enc["labels"] = self.label2id[item["label"]]
        return enc


# -------------------------
# 4) Metrics (micro-F1 excluding no_relation)
# -------------------------
def tacred_f1(eval_pred, id2label):
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    y_true = []
    y_pred = []
    for t, p in zip(labels, preds):
        lab = id2label[int(t)]
        if lab == "no_relation":
            continue
        y_true.append(lab)
        y_pred.append(id2label[int(p)])

    if len(y_true) == 0:  # edge case
        return {"micro_f1_excl_no_rel": 0.0}

    f1 = f1_score(y_true, y_pred, average="micro")
    p, r, f1_full, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    return {
        "precision_excl_no_rel": float(p),
        "recall_excl_no_rel": float(r),
        "micro_f1_excl_no_rel": float(f1),
    }


# -------------------------
# 5) Main
# -------------------------
def main(
        train_path="data/tacred/train.json",
        dev_path="data/tacred/dev.json",
        test_path="data/tacred/test.json",
        model_name="bert-base-uncased",
        out_dir="outputs/bert-tacred",
        seed=42,
):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)

    train_data = read_tacred(train_path)
    dev_data = read_tacred(dev_path)
    test_data = read_tacred(test_path)

    # Build label space from train (ensure no_relation exists)
    labels = sorted({ex["relation"] for ex in train_data})
    if "no_relation" not in labels:
        labels = ["no_relation"] + labels
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    # Prepare tokenizer & special tokens
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    if TYPED:
        # dynamic typed markers: we don't know all types upfront in vocab,
        # but we can safely keep them as raw strings—tokenizer will split them.
        # For best results, you can pre-declare frequent typed markers:
        # e.g., PERSON, ORGANIZATION, LOCATION, DATE, NUMBER, etc.
        extra = [
            "[E1:PERSON]", "[E2:PERSON]",
            "[E1:ORGANIZATION]", "[E2:ORGANIZATION]",
            "[E1:LOCATION]", "[E2:LOCATION]",
            "[E1:MISC]", "[E2:MISC]",
            "[/E1]", "[/E2]",
        ]
    else:
        extra = UNTYPED_MARKERS

    tokenizer.add_special_tokens({"additional_special_tokens": extra})

    # Model
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    # resize embeddings for added tokens
    model.resize_token_embeddings(len(tokenizer))

    # Datasets
    train_ds = TacredREDataset(train_data, tokenizer, label2id)
    dev_ds = TacredREDataset(dev_data, tokenizer, label2id)
    test_ds = TacredREDataset(test_data, tokenizer, label2id)

    # Collator
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training args (tune for your M1; small batch size to fit CPU/MPS)
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1_excl_no_rel",
        greater_is_better=True,
        fp16=False,  # set True on GPU with CUDA; M1 uses 'mps' automatically
        report_to="none",
    )

    def compute_metrics(eval_pred):
        return tacred_f1(eval_pred, id2label)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(dev_ds)
    print("[DEV] ", metrics)

    # Final test eval
    test_metrics = trainer.evaluate(test_ds)
    print("[TEST]", test_metrics)

    # Save
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train_path", default="data/tacred/train.json")
    p.add_argument("--dev_path", default="data/tacred/dev.json")
    p.add_argument("--test_path", default="data/tacred/test.json")
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--out_dir", default="outputs/bert-tacred")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(**vars(args))
