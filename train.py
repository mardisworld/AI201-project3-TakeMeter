"""
train.py — fine-tune distilbert-base-uncased on the labeled Reddit comments and
evaluate it on the held-out test set.

Pipeline:
    1. load data/train.csv + data/val.csv + data/test.csv (from preprocess.py)
    2. tokenize with the DistilBERT tokenizer
    3. fine-tune a 4-class sequence classifier, selecting the best checkpoint on
       validation macro-F1
    4. because the classes are imbalanced (op_response/response dominate), the loss is
       weighted by inverse class frequency so analysis/question are not ignored
    5. evaluate on the test set and print accuracy / macro-F1 / per-class P-R-F1 /
       confusion matrix in the SAME format as baseline.py (so the two models compare
       directly), and save per-row predictions for failure analysis.

Setup:
    pip install "transformers>=4.40" datasets torch scikit-learn

Usage:
    python train.py
    python train.py --epochs 5 --lr 2e-5 --batch-size 16
    python train.py --model distilbert-base-uncased --outdir model
"""

import argparse
import csv
import os
import sys

from labels import LABEL_MAP, LABELS, LABEL_TO_ID, ID_TO_LABEL, NUM_LABELS
# reuse the baseline's scorer/printer so both models report identically
from baseline import score, print_report


def load_split(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"No rows in {path!r}. Run preprocess.py first.")
    texts = [r["text"] for r in rows]
    labels = [int(r["label_id"]) for r in rows]
    return texts, labels


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(here, "data")
    ap = argparse.ArgumentParser(description="Fine-tune DistilBERT on the labeled comments.")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--data-dir", default=data)
    ap.add_argument("--outdir", default=os.path.join(here, "model"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
        from torch import nn
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer, DataCollatorWithPadding, set_seed,
        )
    except ImportError as e:
        sys.exit(f"Missing dependency ({e}). Run: "
                 'pip install "transformers>=4.40" datasets torch scikit-learn')

    set_seed(args.seed)

    # ---- data ----
    train_texts, train_labels = load_split(os.path.join(args.data_dir, "train.csv"))
    val_texts, val_labels = load_split(os.path.join(args.data_dir, "val.csv"))
    test_texts, test_labels = load_split(os.path.join(args.data_dir, "test.csv"))

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def encode(texts, labels):
        enc = tokenizer(texts, truncation=True, max_length=args.max_length)
        return [
            {"input_ids": enc["input_ids"][i],
             "attention_mask": enc["attention_mask"][i],
             "labels": labels[i]}
            for i in range(len(texts))
        ]

    train_ds = encode(train_texts, train_labels)
    val_ds = encode(val_texts, val_labels)

    # ---- class weights (inverse frequency) so rare classes aren't drowned out ----
    counts = np.bincount(train_labels, minlength=NUM_LABELS).astype(float)
    counts[counts == 0] = 1.0
    class_weights = (counts.sum() / (NUM_LABELS * counts))
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print("class weights:", {ID_TO_LABEL[i]: round(float(w), 3) for i, w in enumerate(class_weights)})

    # ---- model ----
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    class WeightedTrainer(Trainer):
        """Trainer with class-weighted cross-entropy."""
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))
            loss = loss_fct(outputs.logits.view(-1, NUM_LABELS), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        # macro-F1 by hand (no sklearn dependency at this point)
        f1s = []
        for c in range(NUM_LABELS):
            tp = int(((preds == c) & (labels == c)).sum())
            fp = int(((preds == c) & (labels != c)).sum())
            fn = int(((preds != c) & (labels == c)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        return {"accuracy": float((preds == labels).mean()), "macro_f1": float(np.mean(f1s))}

    training_args = TrainingArguments(
        output_dir=args.outdir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        seed=args.seed,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"\nFine-tuning {args.model}: {args.epochs} epochs, lr={args.lr}, "
          f"batch={args.batch_size}\n")
    trainer.train()

    # ---- save the best model ----
    trainer.save_model(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    print(f"\nSaved fine-tuned model to {args.outdir}")

    # ---- evaluate on the test set ----
    test_ds = encode(test_texts, test_labels)
    pred_out = trainer.predict(test_ds)
    pred_ids = pred_out.predictions.argmax(axis=-1)

    results = [{
        "text": test_texts[i],
        "true_label": ID_TO_LABEL[test_labels[i]],
        "pred_label": ID_TO_LABEL[int(pred_ids[i])],
        "correct": int(test_labels[i] == int(pred_ids[i])),
    } for i in range(len(test_texts))]

    out_path = os.path.join(args.data_dir, "finetuned_predictions.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "true_label", "pred_label", "correct"])
        w.writeheader()
        w.writerows(results)

    accuracy, macro_f1, per_class, confusion, total, skipped = score(results)
    print_report(accuracy, macro_f1, per_class, confusion, total, skipped,
                 f"fine-tuned {args.model}")
    print(f"\nWrote per-row predictions to {out_path}")


if __name__ == "__main__":
    main()
