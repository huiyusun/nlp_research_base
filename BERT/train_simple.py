from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

MODEL_NAME = "bert-base-uncased"  # Models: distilbert-base-uncased
DATASET = "yelp_review_full"
NUM_LABELS = 5
TRAIN_SAMPLES = 5000  # Keep these small for local iteration.
EVAL_SAMPLES = 1000
MAX_LENGTH = 128
MAX_STEPS = 1200  # Stop after a bounded number of optimizer steps (useful for quick local runs).


def compute_metrics(eval_pred):
    """Return simple, useful metrics for multi-class classification."""
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    # 1) Model + tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    # 2) Data
    ds = load_dataset(DATASET)
    print("RAW SAMPLE: ", ds["train"][0])

    # Select first, split second, tokenize third (fast + gives a real dev set).
    train_small = ds["train"].select(range(TRAIN_SAMPLES))
    split = train_small.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    dev_ds = split["test"]
    test_ds = ds["test"].select(range(EVAL_SAMPLES))

    tokenize = lambda ex: tok(ex["text"], truncation=True, max_length=MAX_LENGTH)
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"]).rename_column("label", "labels")
    dev_ds = dev_ds.map(tokenize, batched=True, remove_columns=["text"]).rename_column("label", "labels")
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"]).rename_column("label", "labels")

    train_ds.set_format(type="torch")
    dev_ds.set_format(type="torch")
    test_ds.set_format(type="torch")

    # 3) Training plumbing
    args = TrainingArguments(
        output_dir="outputs/bert_minimal",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        max_steps=MAX_STEPS,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="no",
        report_to=[],
        per_device_eval_batch_size=32,
        logging_steps=50,
        dataloader_pin_memory=False,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tok)

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    dev_metrics = trainer.evaluate(dev_ds)
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print("\nDEV:", dev_metrics)
    print("TEST:", test_metrics)


if __name__ == "__main__":
    main()
