import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ---------------------------
# Minimal config
# ---------------------------
DATA_DIR = os.path.join("data", "TACRED")
TRAIN_FILE = os.path.join(DATA_DIR, "train_sample.json")
# Use the full train file ONLY to build the label set (avoids dev/test unseen-label crashes when TRAIN_FILE is a small sample).
LABEL_SOURCE_FILE = os.path.join(DATA_DIR, "train.json")
DEV_FILE = os.path.join(DATA_DIR, "dev_sample.json")
TEST_FILE = os.path.join(DATA_DIR, "test.json")

MODEL_NAME = "roberta-base"  # better model: microsoft/deberta-v3-base
OUT_DIR = "outputs/roberta_tacred"
MAX_LEN = 96

# Debug/inspection
PREVIEW_N = 3  # print this many raw inputs (with markers) before training
PRINT_POS_EVERY = 1000  # during prediction, print every N-th *positive* (gold != no_relation)
PRINT_MAX_LINES = 5  # safety cap on how many lines to print per split

# Typed entity markers (based on subj_type/obj_type), e.g.
#   [SUBJ-PERSON] ... [/SUBJ-PERSON]   and   [OBJ-ORGANIZATION] ... [/OBJ-ORGANIZATION]
SUBJ_PREFIX = "SUBJ"
OBJ_PREFIX = "OBJ"


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tokens(ex: Dict[str, Any]) -> List[str]:
    # TACRED uses "token"; some variants use "tokens"
    toks = ex.get("token") or ex.get("tokens")
    if toks is None:
        raise ValueError(f"Example missing 'token'/'tokens'. Keys={list(ex.keys())}")
    return toks


def get_spans(ex: Dict[str, Any]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return (subj_span, obj_span) as inclusive token index spans."""
    required = ["subj_start", "subj_end", "obj_start", "obj_end"]
    if not all(k in ex for k in required):
        raise ValueError(f"Example missing one of {required}. Keys={list(ex.keys())}")

    subj_span = (int(ex["subj_start"]), int(ex["subj_end"]))
    obj_span = (int(ex["obj_start"]), int(ex["obj_end"]))
    return subj_span, obj_span


def get_types(ex: Dict[str, Any]) -> Tuple[str, str]:
    """Return (subj_type, obj_type) as strings."""
    if "subj_type" not in ex or "obj_type" not in ex:
        raise ValueError(f"Example missing 'subj_type'/'obj_type'. Keys={list(ex.keys())}")
    return str(ex["subj_type"]), str(ex["obj_type"])


def insert_markers(
        tokens: List[str],
        subj_span: Tuple[int, int],
        obj_span: Tuple[int, int],
        *,
        subj_type: Optional[str] = None,
        obj_type: Optional[str] = None,
) -> str:
    """Insert entity markers into tokens and return a single string.

    If subj_type/obj_type are provided, uses typed markers:
      [SUBJ-<TYPE>] ... [/SUBJ-<TYPE>] and [OBJ-<TYPE>] ... [/OBJ-<TYPE>]
    Otherwise, falls back to untyped markers [SUBJ]...[/SUBJ] and [OBJ]...[/OBJ].
    """
    (s1, e1) = subj_span
    (s2, e2) = obj_span

    def _mk(prefix: str, typ: Optional[str]) -> Tuple[str, str]:
        if typ is None:
            return f"[{prefix}]", f"[/{prefix}]"
        return f"[{prefix}-{typ}]", f"[/{prefix}-{typ}]"

    subj_start, subj_end = _mk(SUBJ_PREFIX, subj_type)
    obj_start, obj_end = _mk(OBJ_PREFIX, obj_type)

    # Insert from right to left to avoid index shifts
    spans = sorted(
        [("SUBJ", s1, e1), ("OBJ", s2, e2)],
        key=lambda x: x[1],
        reverse=True,
    )

    toks = tokens[:]
    for tag, s, e in spans:
        if tag == "SUBJ":
            start_tok, end_tok = subj_start, subj_end
        else:
            start_tok, end_tok = obj_start, obj_end
        toks = toks[:s] + [start_tok] + toks[s: e + 1] + [end_tok] + toks[e + 1:]

    return " ".join(toks)


def build_text(ex: Dict[str, Any]) -> str:
    """Return the human-readable BERT input string: sentence with entity markers."""
    tokens = get_tokens(ex)
    subj_span, obj_span = get_spans(ex)
    subj_type, obj_type = get_types(ex)
    return insert_markers(tokens, subj_span, obj_span, subj_type=subj_type, obj_type=obj_type)


def preview_inputs(split_name: str, data: List[Dict[str, Any]], tokenizer, n: int = PREVIEW_N) -> None:
    """Print what the input to BERT looks like before embeddings."""
    print(f"\n[PREVIEW INPUTS: {split_name}] (showing up to {n})")
    for i, ex in enumerate(data[:n]):
        text = build_text(ex)
        enc = tokenizer(text, truncation=True, max_length=MAX_LEN, padding=False)
        print(f"Text: {text}.\tRelation: {ex['relation']}")
        print(f"Wordpiece tokens: {tokenizer.convert_ids_to_tokens(enc['input_ids'])}")


@torch.no_grad()
def print_predictions_periodically(
        split_name: str,
        data: List[Dict[str, Any]],
        tokenizer,
        model,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        *,
        print_pos_every: int = PRINT_POS_EVERY,
        max_lines: int = PRINT_MAX_LINES,
) -> None:
    """Run the model and print (gold, pred) for every N-th positive example to avoid clutter."""

    model.eval()
    device = next(model.parameters()).device

    printed = 0
    pos_seen = 0

    print(f"\n[PREDICTIONS: {split_name}] printing every {print_pos_every}-th positive (gold != no_relation)")

    for ex in data:
        gold = ex["relation"]
        if gold != "no_relation":
            pos_seen += 1
            if pos_seen % print_pos_every != 0:
                continue
        else:
            continue  # user asked for relation examples; skip gold no_relation

        text = build_text(ex)
        enc = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).squeeze(0)
        pred_id = int(probs.argmax(dim=-1).item())
        topk = torch.topk(probs, k=min(5, probs.numel()))
        topk_ids = topk.indices.tolist()
        topk_probs = topk.values.tolist()

        pred = id2label[pred_id]

        print("\n---")
        print(f"pos_index={pos_seen}  gold={gold}  pred={pred}")
        print("top 5 label probabilities :")
        for rid, p in zip(topk_ids, topk_probs):
            print(f"  {id2label[int(rid)]}: {float(p):.4f}")

        print("text:")
        print(text)

        printed += 1
        if printed >= max_lines:
            print(f"\n(reached PRINT_MAX_LINES={max_lines}, stopping output for {split_name})")
            break

    if pos_seen == 0:
        print("(no positive examples found in this split)")


class TacredDataset(torch.utils.data.Dataset):
    """Simple on-the-fly tokenization dataset."""

    def __init__(self, data: List[Dict[str, Any]], tokenizer, label2id: Dict[str, int]):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.data[idx]
        tokens = get_tokens(ex)
        subj_span, obj_span = get_spans(ex)
        subj_type, obj_type = get_types(ex)
        text = insert_markers(tokens, subj_span, obj_span, subj_type=subj_type, obj_type=obj_type)

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )
        enc["labels"] = self.label2id[ex["relation"]]
        return enc


def build_metrics(id2label: Dict[int, str]):
    """Metrics for TACRED-style RE.

    - micro-F1 excluding gold no_relation (common TACRED convention)
    - accuracy including no_relation (sanity check)
    - % predicted no_relation (helps diagnose collapse)
    - % gold no_relation (context for imbalance)
    """

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        labels = np.asarray(labels)
        preds = np.asarray(preds)

        # Sanity metrics including no_relation
        acc_all = float((preds == labels).mean()) if len(labels) else 0.0

        pred_no_rel = sum(1 for p in preds if id2label[int(p)] == "no_relation")
        gold_no_rel = sum(1 for t in labels if id2label[int(t)] == "no_relation")
        n = len(labels)
        pred_no_rel_pct = float(pred_no_rel / n) if n else 0.0
        gold_no_rel_pct = float(gold_no_rel / n) if n else 0.0

        # TACRED micro-F1 excluding gold no_relation
        y_true, y_pred = [], []
        for gold_id, pred_id in zip(labels, preds):
            gold = id2label[int(gold_id)]
            if gold == "no_relation":
                continue
            y_true.append(gold)
            y_pred.append(id2label[int(pred_id)])

        if not y_true:
            return {
                "accuracy_all": acc_all,
                "pred_no_relation_pct": pred_no_rel_pct,
                "gold_no_relation_pct": gold_no_rel_pct,
                "precision_excl_no_relation": 0.0,
                "recall_excl_no_relation": 0.0,
                "micro_f1_excl_no_relation": 0.0,
            }

        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        return {
            "accuracy_all": acc_all,
            "pred_no_relation_pct": pred_no_rel_pct,
            "gold_no_relation_pct": gold_no_rel_pct,
            "precision_excl_no_relation": float(p),
            "recall_excl_no_relation": float(r),
            "micro_f1_excl_no_relation": float(f1),
        }

    return compute_metrics


def main():
    train_data = load_json(TRAIN_FILE)
    dev_data = load_json(DEV_FILE)
    test_data = load_json(TEST_FILE)

    # Quick data sanity: label imbalance (TACRED is heavily no_relation)
    def _label_stats(name: str, data: List[Dict[str, Any]]):
        total = len(data)
        no_rel = sum(1 for ex in data if ex.get("relation") == "no_relation")
        pos = total - no_rel
        pct_no = (no_rel / total) if total else 0.0
        print(f"[{name}] total={total}  positives={pos}  no_relation={no_rel} ({pct_no:.1%})")

    _label_stats("TRAIN", train_data)
    _label_stats("DEV", dev_data)
    _label_stats("TEST", test_data)

    # Label space from full TACRED train.json (labels are metadata; avoids sample-label crashes)
    label_source_data = load_json(LABEL_SOURCE_FILE) if LABEL_SOURCE_FILE != TRAIN_FILE else train_data
    labels = sorted({ex["relation"] for ex in label_source_data})
    if "no_relation" not in labels:
        labels = ["no_relation"] + labels

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    # Fail fast if dev/test contains unseen labels (usually means label source isn't full train)
    for split_name, split_data in [("dev", dev_data), ("test", test_data)]:
        unseen = sorted({ex["relation"] for ex in split_data if ex["relation"] not in label2id})
        if unseen:
            raise ValueError(
                f"Unseen relation labels in {split_name}: {unseen[:10]}"
                + (" ..." if len(unseen) > 10 else "")
                + "\nFix: ensure LABEL_SOURCE_FILE points to the full train.json containing all labels."
                + f"\nCurrently TRAIN_FILE={TRAIN_FILE} and LABEL_SOURCE_FILE={LABEL_SOURCE_FILE}"
            )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Add typed marker tokens to the tokenizer vocab so they stay atomic.
    # Collect entity types from the full label source (train.json) + dev/test (safe: types are metadata).
    def _collect_types(data: List[Dict[str, Any]]) -> set[str]:
        types: set[str] = set()
        for ex in data:
            if "subj_type" in ex:
                types.add(str(ex["subj_type"]))
            if "obj_type" in ex:
                types.add(str(ex["obj_type"]))
        return types

    all_types = set()
    all_types |= _collect_types(label_source_data)
    all_types |= _collect_types(dev_data)
    all_types |= _collect_types(test_data)

    marker_tokens: List[str] = []
    # untyped fallbacks
    marker_tokens.extend([f"[{SUBJ_PREFIX}]", f"[/{SUBJ_PREFIX}]", f"[{OBJ_PREFIX}]", f"[/{OBJ_PREFIX}]"])
    # typed
    for t in sorted(all_types):
        marker_tokens.extend([
            f"[{SUBJ_PREFIX}-{t}]",
            f"[/{SUBJ_PREFIX}-{t}]",
            f"[{OBJ_PREFIX}-{t}]",
            f"[/{OBJ_PREFIX}-{t}]",
        ])

    tokenizer.add_special_tokens({"additional_special_tokens": marker_tokens})

    # Show what the model actually sees (before embeddings)
    preview_inputs("train", train_data, tokenizer, n=PREVIEW_N)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    model.resize_token_embeddings(len(tokenizer))
    # Reduce activation memory on GPU/MPS
    model.gradient_checkpointing_enable()
    # Avoid caching key/values (mostly relevant for decoder models, but safe)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    train_ds = TacredDataset(train_data, tokenizer, label2id)
    dev_ds = TacredDataset(dev_data, tokenizer, label2id)
    test_ds = TacredDataset(test_data, tokenizer, label2id)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=150,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1_excl_no_relation",
        greater_is_better=True,
        dataloader_pin_memory=False,
        save_total_limit=2,
    )

    collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=build_metrics(id2label),
    )
    trainer.train()

    print("\n[DEV]")
    dev_metrics = trainer.evaluate(dev_ds)
    print(f"precision : {dev_metrics['eval_precision_excl_no_relation']:.6f}")
    print(f"recall    : {dev_metrics['eval_recall_excl_no_relation']:.6f}")
    print(f"f1        : {dev_metrics['eval_micro_f1_excl_no_relation']:.6f}")

    # Print remaining debug metrics compactly
    other_dev = {k: v for k, v in dev_metrics.items() if 'precision_excl' not in k and 'recall_excl' not in k and 'micro_f1_excl' not in k}
    print(other_dev)

    print("\n[TEST]")
    test_metrics = trainer.evaluate(test_ds)
    print(f"precision : {test_metrics['eval_precision_excl_no_relation']:.6f}")
    print(f"recall    : {test_metrics['eval_recall_excl_no_relation']:.6f}")
    print(f"f1        : {test_metrics['eval_micro_f1_excl_no_relation']:.6f}")

    # Print remaining debug metrics compactly
    other_test = {k: v for k, v in test_metrics.items() if 'precision_excl' not in k and 'recall_excl' not in k and 'micro_f1_excl' not in k}
    print(other_test)

    # Periodic qualitative inspection: gold vs predicted labels
    print_predictions_periodically(
        "dev",
        dev_data,
        tokenizer,
        trainer.model,
        label2id,
        id2label,
        print_pos_every=PRINT_POS_EVERY,
        max_lines=PRINT_MAX_LINES,
    )

    print_predictions_periodically(
        "test",
        test_data,
        tokenizer,
        trainer.model,
        label2id,
        id2label,
        print_pos_every=PRINT_POS_EVERY,
        max_lines=PRINT_MAX_LINES,
    )

    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)


if __name__ == "__main__":
    main()
