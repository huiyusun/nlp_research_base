import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ---------------------------
# Config (edit if you want)
# ---------------------------
DATA_DIR = os.path.join("datasets", "TACRED")
TRAIN_FILE = os.path.join(DATA_DIR, "train.json")
DEV_FILE = os.path.join(DATA_DIR, "dev.json")
TEST_FILE = os.path.join(DATA_DIR, "test.json")

MODEL_NAME = "bert-base-uncased"
OUT_DIR = "outputs/bert_tacred"
MAX_LEN = 256

# Use typed entity markers (often better). Set to False for untyped.
TYPED_MARKERS = True


# ---------------------------
# TACRED loader
# ---------------------------

def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_span(ex: Dict[str, Any], key: str) -> Tuple[int, int]:
    # Standard TACRED has ex["h"]["pos"] and ex["t"]["pos"], inclusive indices.
    # Some variants use different field names; keep it simple but robust.
    span = ex[key].get("pos") or ex[key].get("indices")
    if span is None or len(span) != 2:
        raise ValueError(f"Missing span for {key} in example: {ex.keys()}")
    return int(span[0]), int(span[1])


def _get_type(ex: Dict[str, Any], key: str) -> str:
    return str(ex[key].get("type") or "ENT")


def _insert_markers(tokens: List[str], h_span: Tuple[int, int], t_span: Tuple[int, int], h_type: str, t_type: str) -> str:
    """Return a single string with entity markers inserted.

    TACRED spans are inclusive token indices.

    Example output (typed):
      [E1:PERSON] barack obama [/E1] ... [E2:LOCATION] hawaii [/E2]
    """
    (hs, he) = h_span
    (ts, te) = t_span

    # Insert from the right to avoid index shifts.
    pairs = sorted(
        [("H", hs, he, h_type), ("T", ts, te, t_type)],
        key=lambda x: x[1],
        reverse=True,
    )

    toks = tokens[:]
    for tag, s, e, typ in pairs:
        if TYPED_MARKERS:
            start_tok = f"[E1:{typ}]" if tag == "H" else f"[E2:{typ}]"
        else:
            start_tok = "[E1]" if tag == "H" else "[E2]"
        end_tok = "[/E1]" if tag == "H" else "[/E2]"
        toks = toks[:s] + [start_tok] + toks[s: e + 1] + [end_tok] + toks[e + 1:]

    return " ".join(toks)


# ---------------------------
# Dataset
# ---------------------------

@dataclass
class EncodedItem:
    input_ids: List[int]
    attention_mask: List[int]
    labels: int


class TacredDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, label2id: Dict[str, int]):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.items: List[EncodedItem] = []

        for ex in data:
            tokens = ex.get("tokens") or ex.get("token")
            if tokens is None:
                raise ValueError("Example missing 'tokens' field")

            h_span = _get_span(ex, "h")
            t_span = _get_span(ex, "t")
            h_type = _get_type(ex, "h")
            t_type = _get_type(ex, "t")

            text = _insert_markers(tokens, h_span, t_span, h_type, t_type)

            enc = tokenizer(
                text,
                truncation=True,
                max_length=MAX_LEN,
                padding=False,
            )

            rel = ex.get("relation")
            if rel is None:
                raise ValueError("Example missing 'relation'")

            self.items.append(
                EncodedItem(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    labels=label2id[rel],
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        return {
            "input_ids": it.input_ids,
            "attention_mask": it.attention_mask,
            "labels": it.labels,
        }


# ---------------------------
# Metrics (TACRED-style micro-F1 excluding no_relation)
# ---------------------------

def build_metrics(id2label: Dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # Micro-F1 excluding no_relation (common TACRED convention)
        y_true = []
        y_pred = []
        for t, p in zip(labels, preds):
            gold = id2label[int(t)]
            if gold == "no_relation":
                continue
            y_true.append(gold)
            y_pred.append(id2label[int(p)])

        if len(y_true) == 0:
            return {"micro_f1_excl_no_relation": 0.0}

        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        return {
            "precision_excl_no_relation": float(p),
            "recall_excl_no_relation": float(r),
            "micro_f1_excl_no_relation": float(f1),
        }

    return compute_metrics


def main():
    # 1) Load data
    train_data = _load_json(TRAIN_FILE)
    dev_data = _load_json(DEV_FILE)
    test_data = _load_json(TEST_FILE)

    # 2) Labels
    labels = sorted({ex["relation"] for ex in train_data})
    if "no_relation" not in labels:
        labels = ["no_relation"] + labels
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    # 3) Tokenizer + special tokens
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Add markers to tokenizer vocab so they are treated as single tokens.
    # For typed markers, we add common TACRED entity types; extra types will just be split,
    # but this still works fine.
    special = ["[/E1]", "[/E2]"]
    if TYPED_MARKERS:
        common_types = [
            "PERSON",
            "ORGANIZATION",
            "LOCATION",
            "DATE",
            "NUMBER",
            "TITLE",
            "MISC",
        ]
        for t in common_types:
            special.append(f"[E1:{t}]")
            special.append(f"[E2:{t}]")
    else:
        special += ["[E1]", "[E2]"]

    tokenizer.add_special_tokens({"additional_special_tokens": special})

    # 4) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    model.resize_token_embeddings(len(tokenizer))

    # 5) Datasets
    train_ds = TacredDataset(train_data, tokenizer, label2id)
    dev_ds = TacredDataset(dev_data, tokenizer, label2id)
    test_ds = TacredDataset(test_data, tokenizer, label2id)

    # 6) Trainer
    args = TrainingArguments(
        output_dir=OUT_DIR,
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1_excl_no_relation",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=build_metrics(id2label),
    )

    trainer.train()

    print("\n[DEV]")
    print(trainer.evaluate(dev_ds))

    print("\n[TEST]")
    print(trainer.evaluate(test_ds))

    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)


if __name__ == "__main__":
    main()
