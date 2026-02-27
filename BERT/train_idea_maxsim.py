import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ---------------------------
# Config (Stage 1: MaxSim head + CE)
# ---------------------------
DATA_DIR = os.path.join("data", "TACRED")
TRAIN_FILE = os.path.join(DATA_DIR, "train_sample.json")
DEV_FILE = os.path.join(DATA_DIR, "dev_sample.json")
TEST_FILE = os.path.join(DATA_DIR, "test.json")

# Use full train ONLY to build label set + collect entity types for marker vocab
LABEL_SOURCE_FILE = os.path.join(DATA_DIR, "train.json")

MODEL_NAME = "roberta-base"  # try: microsoft/deberta-v3-base, bert-base-uncased
OUT_DIR = "outputs/maxsim_stage1_tacred"

MAX_LEN = 96
PROJ_DIM = 128  # ColBERT-style projection dim
CONTEXT_WINDOW = 32
TOPK = 4

# Debug / inspection
PREVIEW_N = 3

# Typed entity markers
SUBJ_PREFIX = "SUBJ"
OBJ_PREFIX = "OBJ"


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tokens(ex: Dict[str, Any]) -> List[str]:
    toks = ex.get("token") or ex.get("tokens")
    if toks is None:
        raise ValueError(f"Example missing 'token'/'tokens'. Keys={list(ex.keys())}")
    return toks


def get_spans(ex: Dict[str, Any]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    required = ["subj_start", "subj_end", "obj_start", "obj_end"]
    if not all(k in ex for k in required):
        raise ValueError(f"Example missing one of {required}. Keys={list(ex.keys())}")
    return (int(ex["subj_start"]), int(ex["subj_end"])), (int(ex["obj_start"]), int(ex["obj_end"]))


def get_types(ex: Dict[str, Any]) -> Tuple[str, str]:
    if "subj_type" not in ex or "obj_type" not in ex:
        raise ValueError(f"Example missing 'subj_type'/'obj_type'. Keys={list(ex.keys())}")
    return str(ex["subj_type"]), str(ex["obj_type"])


def insert_markers(
        tokens: List[str],
        subj_span: Tuple[int, int],
        obj_span: Tuple[int, int],
        *,
        subj_type: Optional[str],
        obj_type: Optional[str],
) -> str:
    """Insert typed entity markers and return a single string."""
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
    tokens = get_tokens(ex)
    subj_span, obj_span = get_spans(ex)
    subj_type, obj_type = get_types(ex)
    return insert_markers(tokens, subj_span, obj_span, subj_type=subj_type, obj_type=obj_type)


def preview_inputs(split_name: str, data: List[Dict[str, Any]], tokenizer, n: int = PREVIEW_N) -> None:
    print(f"\n[PREVIEW INPUTS: {split_name}] (showing up to {n})")
    for ex in data[:n]:
        text = build_text(ex)
        enc = encode_with_masks(ex, tokenizer)
        print(tokenizer.convert_ids_to_tokens(enc["input_ids"]))
        print(f"Text: {text}  Relation: {ex['relation']}")
        print(f"Tokens: {tokenizer.convert_ids_to_tokens(enc['input_ids'])}")


def _collect_types(data: List[Dict[str, Any]]) -> set[str]:
    types: set[str] = set()
    for ex in data:
        if "subj_type" in ex:
            types.add(str(ex["subj_type"]))
        if "obj_type" in ex:
            types.add(str(ex["obj_type"]))
    return types


def _mk_marker(prefix: str, typ: Optional[str], closing: bool) -> str:
    if typ is None:
        return f"[/{prefix}]" if closing else f"[{prefix}]"
    return f"[/{prefix}-{typ}]" if closing else f"[{prefix}-{typ}]"


def _find_marker_positions(input_ids: List[int], tokenizer, start_tok: str, end_tok: str) -> Tuple[int, int]:
    """Return (start_idx, end_idx) inclusive positions of the marker tokens in tokenized ids."""
    start_id = tokenizer.convert_tokens_to_ids(start_tok)
    end_id = tokenizer.convert_tokens_to_ids(end_tok)
    if start_id is None or end_id is None:
        raise ValueError(f"Marker token not in vocab: {start_tok} or {end_tok}")

    start_pos = None
    end_pos = None
    for i, tid in enumerate(input_ids):
        if tid == start_id and start_pos is None:
            start_pos = i
        elif tid == end_id and start_pos is not None:
            end_pos = i
            break

    if start_pos is None or end_pos is None or end_pos <= start_pos:
        print("DEBUG marker tokens:", start_tok, end_tok)
        print("DEBUG ids contain start/end?:",
              tokenizer.convert_tokens_to_ids(start_tok) in input_ids,
              tokenizer.convert_tokens_to_ids(end_tok) in input_ids)
        print("DEBUG first 200 tokens:", tokenizer.convert_ids_to_tokens(input_ids[:200]))
        raise ValueError(f"Could not find marker span {start_tok} ... {end_tok} in tokenized input.")

    return start_pos, end_pos


def encode_with_masks(ex: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Tokenize and create masks aligned to tokenized length.

    Key fix: don't use naive truncation, because it can cut off entity end markers.
    We tokenize long, find markers, then crop a window that keeps both entities.
    """

    text = build_text(ex)

    subj_type, obj_type = get_types(ex)
    subj_start_tok = _mk_marker(SUBJ_PREFIX, subj_type, closing=False)
    subj_end_tok = _mk_marker(SUBJ_PREFIX, subj_type, closing=True)
    obj_start_tok = _mk_marker(OBJ_PREFIX, obj_type, closing=False)
    obj_end_tok = _mk_marker(OBJ_PREFIX, obj_type, closing=True)

    # 1) tokenize without special tokens + without truncation
    enc_long = tokenizer(text, add_special_tokens=False, truncation=False, padding=False)
    ids_long: List[int] = enc_long["input_ids"]

    # 2) find markers in the long sequence
    s_start, s_end = _find_marker_positions(ids_long, tokenizer, subj_start_tok, subj_end_tok)
    o_start, o_end = _find_marker_positions(ids_long, tokenizer, obj_start_tok, obj_end_tok)

    # 3) crop a window of length (MAX_LEN - 2) that contains both entities
    inner_max = max(1, MAX_LEN - 2)

    if len(ids_long) > inner_max:
        left_ent = min(s_start, o_start)
        right_ent = max(s_end, o_end)

        ent_len = right_ent - left_ent + 1
        if ent_len >= inner_max:
            ws = left_ent
        else:
            pad = (inner_max - ent_len) // 2
            ws = max(0, left_ent - pad)

        ws = min(ws, len(ids_long) - inner_max)
        we = ws + inner_max

        ids = ids_long[ws:we]
        s_start -= ws;
        s_end -= ws
        o_start -= ws;
        o_end -= ws
    else:
        ids = ids_long

    # 4) add special tokens back
    # 4) add special tokens back (manual, compatible with all tokenizers)
    prefix: List[int] = []
    suffix: List[int] = []

    # Prefer CLS/SEP; fallback to BOS/EOS.
    if getattr(tokenizer, "cls_token_id", None) is not None:
        prefix = [int(tokenizer.cls_token_id)]
    elif getattr(tokenizer, "bos_token_id", None) is not None:
        prefix = [int(tokenizer.bos_token_id)]

    if getattr(tokenizer, "sep_token_id", None) is not None:
        suffix = [int(tokenizer.sep_token_id)]
    elif getattr(tokenizer, "eos_token_id", None) is not None:
        suffix = [int(tokenizer.eos_token_id)]

    input_ids: List[int] = prefix + ids + suffix
    attention_mask: List[int] = [1] * len(input_ids)

    # shift marker indices by prefix length (usually 1)
    offset = len(prefix)
    s_start += offset
    s_end += offset
    o_start += offset
    o_end += offset

    L = len(input_ids)

    subj_mask = [0] * L
    obj_mask = [0] * L
    ctx_mask = [0] * L

    for i in range(s_start + 1, s_end):
        if 0 <= i < L:
            subj_mask[i] = 1
    for i in range(o_start + 1, o_end):
        if 0 <= i < L:
            obj_mask[i] = 1

    # --- BETWEEN MASK computation ---
    between_mask = [0] * L

    # Determine which entity comes first based on marker positions
    subj_first = 1 if s_start < o_start else 0

    # Define the between region in token indices (exclusive of markers)
    if subj_first:
        between_left = s_end + 1
        between_right = o_start - 1
    else:
        between_left = o_end + 1
        between_right = s_start - 1

    if between_left <= between_right:
        for i in range(between_left, between_right + 1):
            if 0 <= i < L:
                between_mask[i] = 1

    special_ids = set(tokenizer.all_special_ids)
    marker_ids = {
        tokenizer.convert_tokens_to_ids(subj_start_tok),
        tokenizer.convert_tokens_to_ids(subj_end_tok),
        tokenizer.convert_tokens_to_ids(obj_start_tok),
        tokenizer.convert_tokens_to_ids(obj_end_tok),
    }

    ent_left = min(s_start, o_start)
    ent_right = max(s_end, o_end)
    win_left = max(0, ent_left - CONTEXT_WINDOW)
    win_right = min(L - 1, ent_right + CONTEXT_WINDOW)

    for i, tid in enumerate(input_ids):
        if i < win_left or i > win_right:
            continue
        if tid in special_ids:
            continue
        if tid in marker_ids:
            continue
        if subj_mask[i] == 1 or obj_mask[i] == 1:
            continue
        ctx_mask[i] = 1
        # Keep between_mask clean as well (exclude specials/markers/entities)
        if between_mask[i] == 1:
            between_mask[i] = 1

    # Cleanup: zero out invalid between positions (special/marker/entity tokens)
    for i, tid in enumerate(input_ids):
        if between_mask[i] != 1:
            continue
        if tid in special_ids or tid in marker_ids or subj_mask[i] == 1 or obj_mask[i] == 1:
            between_mask[i] = 0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "subj_mask": subj_mask,
        "obj_mask": obj_mask,
        "ctx_mask": ctx_mask,
        "between_mask": between_mask,
        "subj_first": int(subj_first),
    }


def tokenize_split(
        data: List[Dict[str, Any]],
        tokenizer,
        label2id: Dict[str, int],
        type2id: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Pre-tokenize a split once to make training much faster."""
    items: List[Dict[str, Any]] = []
    for ex in data:
        enc = encode_with_masks(ex, tokenizer)
        enc["labels"] = int(label2id[ex["relation"]])
        # Add entity type ids for query conditioning
        st, ot = get_types(ex)
        enc["subj_type_id"] = int(type2id[st])
        enc["obj_type_id"] = int(type2id[ot])
        items.append(enc)
    return items


class TacredDataset(torch.utils.data.Dataset):
    """Dataset of already-tokenized examples (fast)."""

    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def make_collator(tokenizer):
    """Pads input_ids/attention_mask with HF collator; pads custom masks with 0."""

    base = DataCollatorWithPadding(tokenizer)

    def collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        subj = [f.pop("subj_mask") for f in features]
        obj = [f.pop("obj_mask") for f in features]
        ctx = [f.pop("ctx_mask") for f in features]
        between = [f.pop("between_mask") for f in features]

        subj_type_id = [f.pop("subj_type_id") for f in features]
        obj_type_id = [f.pop("obj_type_id") for f in features]
        subj_first = [f.pop("subj_first") for f in features]

        batch = base(features)
        L = batch["input_ids"].shape[1]

        def _pad(mask_list: List[List[int]]) -> torch.Tensor:
            padded = [m + [0] * (L - len(m)) for m in mask_list]
            return torch.tensor(padded, dtype=torch.bool)

        batch["subj_mask"] = _pad(subj)
        batch["obj_mask"] = _pad(obj)
        batch["ctx_mask"] = _pad(ctx)
        batch["between_mask"] = _pad(between)
        batch["subj_type_id"] = torch.tensor(subj_type_id, dtype=torch.long)
        batch["obj_type_id"] = torch.tensor(obj_type_id, dtype=torch.long)
        batch["subj_first"] = torch.tensor(subj_first, dtype=torch.long)
        return batch

    return collate


class MaxSimREModel(nn.Module):
    """Query→document late-interaction RE model.

    Core idea (ColBERT-style):
      - Document = sentence token embeddings (projected + L2-normalized)
      - Query = a set of learned pattern probes per relation label, conditioned on
               (subj_type, obj_type, direction)
      - Score(label) = sum_h max_{j in evidence} <q_{label,h}, d_j>

    Evidence focuses on structure:
      - Prefer tokens strictly BETWEEN the entities (often contains trigger/prep/verb)
      - Fallback to local context window if between is empty

    Compositional patterns:
      - Multiple probes (H) per label allow capturing multi-token cues like "CEO" + "of".
    """

    def __init__(
            self,
            backbone_name: str,
            num_labels: int,
            num_types: int,
            proj_dim: int = PROJ_DIM,
            num_probes: int = 8,
    ):
        super().__init__()
        dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
        self.encoder = AutoModel.from_pretrained(backbone_name, dtype=dtype)
        self.proj = nn.Linear(self.encoder.config.hidden_size, proj_dim, bias=False)

        self.num_labels = int(num_labels)
        self.num_types = int(num_types)
        self.proj_dim = int(proj_dim)
        self.num_probes = int(num_probes)

        # Base learned query probes per label: [R, H, M]
        self.label_probes = nn.Parameter(torch.empty(self.num_labels, self.num_probes, self.proj_dim))
        nn.init.normal_(self.label_probes, mean=0.0, std=0.02)

        # Type conditioning (subj/object roles) and direction conditioning
        self.subj_type_emb = nn.Embedding(self.num_types, self.proj_dim)
        self.obj_type_emb = nn.Embedding(self.num_types, self.proj_dim)
        self.dir_emb = nn.Embedding(2, self.proj_dim)  # 0: obj-first, 1: subj-first

        nn.init.normal_(self.subj_type_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.obj_type_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dir_emb.weight, mean=0.0, std=0.02)

    def gradient_checkpointing_enable(self):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            subj_mask: torch.Tensor,
            obj_mask: torch.Tensor,
            ctx_mask: torch.Tensor,
            between_mask: torch.Tensor,
            subj_type_id: torch.Tensor,
            obj_type_id: torch.Tensor,
            subj_first: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = out.last_hidden_state  # [B, L, d]

        # Document tokens in probe space
        D = self.proj(H)  # [B, L, M]
        D = F.normalize(D, p=2, dim=-1)

        # Evidence mask: prefer BETWEEN tokens; fallback to local context if empty
        between_mask = between_mask.bool()
        ctx_mask = ctx_mask.bool()
        has_between = between_mask.any(dim=1)  # [B]
        evidence_mask = torch.where(has_between.unsqueeze(1), between_mask, ctx_mask)  # [B, L]

        # Build conditioned queries: base probes + type + direction
        # base: [R, H, M] -> [1, R, H, M]
        Q = self.label_probes.unsqueeze(0)

        st = self.subj_type_emb(subj_type_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,M]
        ot = self.obj_type_emb(obj_type_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,M]
        de = self.dir_emb(subj_first).unsqueeze(1).unsqueeze(2)  # [B,1,1,M]

        Q = Q + st + ot + de  # [B, R, H, M]
        Q = F.normalize(Q, p=2, dim=-1)

        # Similarity: [B, R, H, L]
        sim = torch.einsum("brhm,blm->brhl", Q, D)

        # Mask out non-evidence positions in L
        em = evidence_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
        sim = sim.masked_fill(~em, -1e4)

        # Late interaction: max over evidence tokens, sum over probes
        mx = sim.max(dim=3).values  # [B, R, H]
        mx = torch.where(mx > -1e3, mx, torch.zeros_like(mx))
        logits = mx.sum(dim=2)  # [B, R]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


def build_metrics(id2label: Dict[int, str]):
    """TACRED micro-F1 excluding gold no_relation + basic sanity stats."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        labels = np.asarray(labels)
        preds = np.asarray(preds)

        acc_all = float((preds == labels).mean()) if len(labels) else 0.0

        pred_no_rel = int(sum(1 for p in preds if id2label[int(p)] == "no_relation"))
        gold_no_rel = int(sum(1 for t in labels if id2label[int(t)] == "no_relation"))
        n = int(len(labels))
        pred_no_rel_pct = float(pred_no_rel / n) if n else 0.0
        gold_no_rel_pct = float(gold_no_rel / n) if n else 0.0

        y_true: List[str] = []
        y_pred: List[str] = []
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

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
        return {
            "accuracy_all": acc_all,
            "pred_no_relation_pct": pred_no_rel_pct,
            "gold_no_relation_pct": gold_no_rel_pct,
            "precision_excl_no_relation": float(p),
            "recall_excl_no_relation": float(r),
            "micro_f1_excl_no_relation": float(f1),
        }

    return compute_metrics


def marker_embed_norm(model: MaxSimREModel, tokenizer, tok: str) -> float:
    tid = tokenizer.convert_tokens_to_ids(tok)
    if tid is None or tid == tokenizer.unk_token_id:
        return float("nan")
    # works for most HF models: get_input_embeddings()
    emb = model.encoder.get_input_embeddings().weight.detach()
    return float(emb[tid].float().norm().cpu().item())


def main() -> None:
    train_data = load_json(TRAIN_FILE)
    dev_data = load_json(DEV_FILE)
    test_data = load_json(TEST_FILE)

    label_source_data = load_json(LABEL_SOURCE_FILE)
    labels = sorted({ex["relation"] for ex in label_source_data})
    if "no_relation" not in labels:
        labels = ["no_relation"] + labels

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    # Fail fast if dev/test contains unseen labels
    for split_name, split_data in [("dev", dev_data), ("test", test_data)]:
        unseen = sorted({ex["relation"] for ex in split_data if ex["relation"] not in label2id})
        if unseen:
            raise ValueError(
                f"Unseen relation labels in {split_name}: {unseen[:10]}"
                + (" ..." if len(unseen) > 10 else "")
                + f"\nFix: ensure LABEL_SOURCE_FILE points to full train.json. Currently: {LABEL_SOURCE_FILE}"
            )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Add typed marker tokens to vocab so they stay atomic
    all_types = set()
    all_types |= _collect_types(label_source_data)
    all_types |= _collect_types(dev_data)
    all_types |= _collect_types(test_data)

    type2id = {t: i for i, t in enumerate(sorted(all_types))}

    marker_tokens: List[str] = []
    marker_tokens.extend([f"[{SUBJ_PREFIX}]", f"[/{SUBJ_PREFIX}]", f"[{OBJ_PREFIX}]", f"[/{OBJ_PREFIX}]"])
    for t in sorted(all_types):
        marker_tokens.extend([
            f"[{SUBJ_PREFIX}-{t}]",
            f"[/{SUBJ_PREFIX}-{t}]",
            f"[{OBJ_PREFIX}-{t}]",
            f"[/{OBJ_PREFIX}-{t}]",
        ])

    tokenizer.add_special_tokens({"additional_special_tokens": marker_tokens})

    preview_inputs("train", train_data, tokenizer, n=PREVIEW_N)

    model = MaxSimREModel(
        MODEL_NAME,
        num_labels=len(labels),
        num_types=len(type2id),
        proj_dim=PROJ_DIM,
        num_probes=8,
    )

    # Resize token embeddings if encoder supports it (BERT/RoBERTa/DeBERTa do)
    if hasattr(model.encoder, "resize_token_embeddings"):
        model.encoder.resize_token_embeddings(len(tokenizer))

    # Freeze everything first
    for p in model.encoder.parameters():
        p.requires_grad = False

    # Unfreeze embeddings (so new marker tokens actually learn)
    emb = getattr(model.encoder, "embeddings", None)
    if emb is None and hasattr(model.encoder, "roberta"):
        emb = model.encoder.roberta.embeddings
    if emb is None:
        raise RuntimeError("Could not locate encoder embeddings module to unfreeze.")
    for p in emb.parameters():
        p.requires_grad = True

    # Unfreeze last N transformer layers (light finetune, big impact)
    UNFREEZE_LAST_N = 2
    layers = None
    if hasattr(model.encoder, "encoder") and hasattr(model.encoder.encoder, "layer"):
        layers = model.encoder.encoder.layer
    elif hasattr(model.encoder, "roberta") and hasattr(model.encoder.roberta, "encoder"):
        layers = model.encoder.roberta.encoder.layer
    if layers is None:
        raise RuntimeError("Could not locate encoder layers to partially unfreeze.")
    for layer in layers[-UNFREEZE_LAST_N:]:
        for p in layer.parameters():
            p.requires_grad = True

    tok = "[SUBJ-PERSON]"
    tid = tokenizer.convert_tokens_to_ids(tok)
    print("\n[CHECK A] marker token id:", tok, tid)
    print("[CHECK A] marker embedding norm BEFORE:", marker_embed_norm(model, tokenizer, tok))

    emb = model.encoder.get_input_embeddings().weight.detach()
    print("[CHECK A] marker embedding first 5 dims BEFORE:", emb[tid][:5].float().cpu().tolist())

    # Memory savers (MPS-friendly)
    if hasattr(model.encoder.config, "use_cache"):
        model.encoder.config.use_cache = False

    train_items = tokenize_split(train_data, tokenizer, label2id, type2id)
    dev_items = tokenize_split(dev_data, tokenizer, label2id, type2id)
    test_items = tokenize_split(test_data, tokenizer, label2id, type2id)

    train_ds = TacredDataset(train_items)
    dev_ds = TacredDataset(dev_items)
    test_ds = TacredDataset(test_items)

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
        logging_steps=200,
        report_to="none",
        load_best_model_at_end=False,
        metric_for_best_model="micro_f1_excl_no_relation",
        greater_is_better=True,
        dataloader_pin_memory=False,
        save_total_limit=2,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=make_collator(tokenizer),
        compute_metrics=build_metrics(id2label),
    )

    # ---- CHECK A (GRAD): verify marker embeddings receive gradient ----
    print("\n[CHECK A - GRAD]")
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    batch = {k: v.to(trainer.args.device) for k, v in batch.items()}

    model.train()
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    out["loss"].backward()

    tok = "[SUBJ-PERSON]"
    tid = tokenizer.convert_tokens_to_ids(tok)
    grad = model.encoder.get_input_embeddings().weight.grad

    print("[CHECK A - GRAD] token id:", tid)
    print("[CHECK A - GRAD] grad is None?:", grad is None)

    if grad is not None:
        gnorm = float(grad[tid].float().norm().detach().cpu().item())
        print("[CHECK A - GRAD] marker grad norm:", gnorm)

    # Clear grads so Trainer starts clean
    model.zero_grad(set_to_none=True)

    trainer.train()

    print("[CHECK A] marker embedding norm AFTER :", marker_embed_norm(model, tokenizer, tok))
    emb_after = model.encoder.get_input_embeddings().weight.detach()
    print("[CHECK A] marker embedding first 5 dims AFTER :", emb_after[tid][:5].float().cpu().tolist())

    print("\n[TRAIN PRED CHECK]")
    train_pred = trainer.predict(train_ds)
    train_logits = train_pred.predictions
    train_labels = np.asarray(train_pred.label_ids)
    train_preds = np.argmax(train_logits, axis=-1)

    pred_no_rel = float(np.mean(train_preds == label2id["no_relation"]))
    gold_no_rel = float(np.mean(train_labels == label2id["no_relation"]))
    print("train pred_no_relation_pct:", pred_no_rel)
    print("train gold_no_relation_pct:", gold_no_rel)

    # Also: how many unique labels predicted?
    uniq = np.unique(train_preds)
    print("unique predicted labels:", len(uniq), "example ids:", uniq[:20])

    print("\n[DEV]")
    dev_metrics = trainer.evaluate(dev_ds)
    print(f"precision : {dev_metrics['eval_precision_excl_no_relation']:.6f}")
    print(f"recall    : {dev_metrics['eval_recall_excl_no_relation']:.6f}")
    print(f"f1        : {dev_metrics['eval_micro_f1_excl_no_relation']:.6f}")
    other_dev = {k: v for k, v in dev_metrics.items() if "precision_excl" not in k and "recall_excl" not in k and "micro_f1_excl" not in k}
    print(other_dev)

    print("\n[TEST]")
    test_metrics = trainer.evaluate(test_ds)
    print(f"precision : {test_metrics['eval_precision_excl_no_relation']:.6f}")
    print(f"recall    : {test_metrics['eval_recall_excl_no_relation']:.6f}")
    print(f"f1        : {test_metrics['eval_micro_f1_excl_no_relation']:.6f}")
    other_test = {k: v for k, v in test_metrics.items() if "precision_excl" not in k and "recall_excl" not in k and "micro_f1_excl" not in k}
    print(other_test)

    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)


if __name__ == "__main__":
    main()
