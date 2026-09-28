"""
train_promise.py
================
BERT fine-tuning on the PROMISE dataset.

Usage
-----
    python Train_PROMISE_12class.py --mode 2class
    python Train_PROMISE_12class.py --mode 12class
    python Train_PROMISE_12class.py --mode 11class

Expected CSV columns: "Requirement" (text), "Type" (label)
"""

import math
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed,
)

# =========================================================
# PATHS
#   <root>/1_data/raw/labeled/Promise/...
#   <root>/2_experiments/bert/scripts/..
#   <root>/3_models/bert/...
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR     = PROJECT_ROOT / "1_data" / "raw" / "labeled" / "Promise"
MODELS_DIR   = PROJECT_ROOT / "3_models" / "bert"

# =========================================================
# EXPERIMENT CONFIGS  (same hyperparameters as the original scripts)
# =========================================================
CONFIGS = {
    "2class": dict(
        data="PROMISE_exp_2class.csv", run_name="PROMISE_2class",
        model="bert-base-uncased", upper_labels=True,
        split="80_20", max_length=128, epochs=4,
        train_bs=16, eval_bs=16, grad_accum=1, lr=2e-5,
        warmup_ratio=0.1, scheduler="linear", weight_decay=0.0, patience=2,
        loss="ce", focal_gamma=None, oversample_min=None, min_class_total=None,
    ),
    "12class": dict(
        data="PROMISE_exp_12class.csv", run_name="PROMISE_12class",
        model="bert-base-uncased", upper_labels=False,
        split="60_20_20", max_length=256, epochs=20,
        train_bs=8, eval_bs=16, grad_accum=4, lr=3e-5,
        warmup_ratio=0.15, scheduler="cosine", weight_decay=0.01, patience=5,
        loss="focal", focal_gamma=2.0, oversample_min=100, min_class_total=15,
    ),
    "11class": dict(
        data="PROMISE_exp_NFRclass.csv", run_name="PROMISE_NFRclass",
        model="roberta-base", upper_labels=False,
        split="60_20_20", max_length=256, epochs=20,
        train_bs=8, eval_bs=16, grad_accum=4, lr=3e-5,
        warmup_ratio=0.15, scheduler="cosine", weight_decay=0.01, patience=5,
        loss="focal", focal_gamma=2.0, oversample_min=100, min_class_total=15,
    ),
}

TEXT_COL  = "Requirement"
LABEL_COL = "Type"
SEED      = 42


# =========================================================
# HELPERS
# =========================================================
class ReqDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts, self.labels = texts.astype(str).tolist(), labels.astype(int).tolist()
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length)
        enc["labels"] = self.labels[idx]
        return enc


class FocalLoss(nn.Module):
    """Focal loss: down-weights easy examples, focuses on hard / minority ones."""

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight, self.gamma = weight, gamma

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class CustomLossTrainer(Trainer):
    """loss_type: 'ce' (plain CrossEntropy) or 'focal' (class-weighted focal loss)."""

    def __init__(self, *args, loss_type="ce", class_weights=None, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type, self.class_weights, self.focal_gamma = loss_type, class_weights, focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")
        if self.loss_type == "focal":
            loss = FocalLoss(weight=self.class_weights, gamma=self.focal_gamma)(logits, labels)
        else:
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy":           accuracy_score(labels, preds),
        "macro_precision":    precision_score(labels, preds, average="macro",    zero_division=0),
        "macro_recall":       recall_score   (labels, preds, average="macro",    zero_division=0),
        "macro_f1":           f1_score       (labels, preds, average="macro",    zero_division=0),
        "weighted_precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "weighted_recall":    recall_score   (labels, preds, average="weighted", zero_division=0),
        "weighted_f1":        f1_score       (labels, preds, average="weighted", zero_division=0),
    }


def append_to_excel(row: dict, path: Path):
    df_new = pd.DataFrame([row])
    if path.exists():
        df_new = pd.concat([pd.read_excel(path), df_new], ignore_index=True)
    df_new.to_excel(path, index=False)


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="BERT - PROMISE dataset")
    parser.add_argument("--mode", choices=CONFIGS.keys(), required=True,
                        help="2class = FR vs NFR | 12class | 11class ")
    parser.add_argument("--data", default=None, help="Override dataset CSV path")
    parser.add_argument("--output_dir", default=None, help="Override output folder")
    parser.add_argument("--model_name", default=None, help="Override HF model name")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    cfg        = CONFIGS[args.mode]
    model_name = args.model_name or cfg["model"]
    data_path  = Path(args.data) if args.data else DATA_DIR / cfg["data"]
    output_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR / "bert_runs" / cfg["run_name"]
    epochs     = args.epochs or cfg["epochs"]
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: {args.mode} | Model: {model_name} | Device: {device}\nData: {data_path}\nOutput: {output_dir}\n")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}\nUse --data to point to the CSV.")

    # ---------- Load ----------
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    df[TEXT_COL]  = df[TEXT_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    if cfg["upper_labels"]:
        df[LABEL_COL] = df[LABEL_COL].str.upper()
    print("Full dataset distribution:\n", df[LABEL_COL].value_counts(), "\n")

    # ---------- Classes (auto-detected from CSV) ----------
    classes = sorted(df[LABEL_COL].unique().tolist())
    print(f"Detected classes ({len(classes)}): {classes}\n")

    counts = df[LABEL_COL].value_counts()
    if cfg["min_class_total"]:
        tiny = counts[counts < cfg["min_class_total"]].index.tolist()
        if tiny:
            print(f"Dropping classes with < {cfg['min_class_total']} samples: {tiny}")
            df = df[~df[LABEL_COL].isin(tiny)].reset_index(drop=True)
            classes = [c for c in classes if c not in tiny]
    counts = df[LABEL_COL].value_counts()
    too_few = counts[counts < 2].index.tolist()          # can't stratify with < 2
    if too_few:
        df = df[~df[LABEL_COL].isin(too_few)].reset_index(drop=True)
        classes = [c for c in classes if c not in too_few]

    # ---------- Labels (alphabetical order from LabelEncoder is used everywhere) ----------
    label_encoder  = LabelEncoder().fit(classes)
    sorted_classes = label_encoder.classes_.tolist()
    sorted_names   = sorted_classes
    num_labels     = len(sorted_classes)
    df["label_id"] = label_encoder.transform(df[LABEL_COL])
    print(f"Active classes ({num_labels}): {sorted_classes} | Samples: {len(df)}\n")

    # ---------- Split ----------
    if cfg["split"] == "80_20":
        train_df, test_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["label_id"])
        val_df = test_df
    else:
        train_df, temp_df = train_test_split(df, test_size=0.40, random_state=SEED, stratify=df["label_id"])
        val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label_id"])
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}\n")

    # ---------- Oversampling (train only) ----------
    if cfg["oversample_min"]:
        extra = []
        for cls in classes:
            cls_df = train_df[train_df[LABEL_COL] == cls]
            if len(cls_df) < cfg["oversample_min"]:
                extra.append(cls_df.sample(n=cfg["oversample_min"] - len(cls_df), replace=True, random_state=SEED))
        if extra:
            train_df = pd.concat([train_df] + extra, ignore_index=True)
        print("After oversampling (train):\n", train_df[LABEL_COL].value_counts(), "\n")

    # ---------- Class weights ----------
    class_weights = None
    if cfg["loss"] == "focal":
        w = compute_class_weight("balanced", classes=np.arange(num_labels), y=train_df["label_id"].values)
        class_weights = torch.tensor(w, dtype=torch.float).to(device)

    # ---------- Tokenizer / datasets ----------
    tokenizer     = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    ml = cfg["max_length"]
    train_ds = ReqDataset(train_df[TEXT_COL], train_df["label_id"], tokenizer, ml)
    val_ds   = ReqDataset(val_df[TEXT_COL],   val_df["label_id"],   tokenizer, ml)
    test_ds  = ReqDataset(test_df[TEXT_COL],  test_df["label_id"],  tokenizer, ml)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

    # ---------- Trainer ----------
    # warmup as absolute steps (works on old and new transformers versions)
    steps_per_epoch = math.ceil(len(train_ds) / (cfg["train_bs"] * cfg["grad_accum"]))
    warmup_steps    = int(cfg["warmup_ratio"] * steps_per_epoch * epochs)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg["train_bs"],
        per_device_eval_batch_size=cfg["eval_bs"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        warmup_steps=warmup_steps,
        lr_scheduler_type=cfg["scheduler"],
        weight_decay=cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_only_model=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        logging_steps=50,
    )
    trainer = CustomLossTrainer(
        model=model, args=training_args, data_collator=data_collator,
        train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["patience"])],
        loss_type=cfg["loss"], class_weights=class_weights, focal_gamma=cfg["focal_gamma"] or 2.0,
    )

    trainer.train()

    # ---------- Test evaluation ----------
    test_results = trainer.evaluate(eval_dataset=test_ds)
    test_pred    = trainer.predict(test_ds)
    y_pred = np.argmax(test_pred.predictions, axis=1)
    y_true = test_df["label_id"].values
    labels_idx = np.arange(num_labels)

    report = classification_report(y_true, y_pred, labels=labels_idx, target_names=sorted_names,
                                   zero_division=0, digits=4)
    print("\n=== Classification Report (Test) ===\n", report)
    (output_dir / "classification_report.txt").write_text(
        f"Mode: {args.mode}\nModel: {model_name}\nData: {data_path.name}\n"
        f"Date: {datetime.now():%Y-%m-%d %H:%M}\nClasses: {sorted_classes}\n"
        f"Test samples: {len(test_df)}\n\n{report}"
    )

    # Confusion matrix (counts + row %)
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    cm_pct = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    fig, axes = plt.subplots(1, 2, figsize=(max(12, num_labels * 1.5), max(5, num_labels * 0.6)))
    fig.suptitle(f"Confusion Matrix - PROMISE {args.mode} (Test n={len(test_df)})", fontweight="bold")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted_names,
                yticklabels=sorted_names, ax=axes[0], cbar=False)
    axes[0].set_title("Counts"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Greens", xticklabels=sorted_names,
                yticklabels=sorted_names, ax=axes[1], cbar=False)
    axes[1].set_title("Row % (recall per class)"); axes[1].set_xlabel("Predicted")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / "confusion_matrix_test.png", dpi=150, bbox_inches="tight"); plt.close()

    # Epoch metrics plot
    epoch_logs = [e for e in trainer.state.log_history if "eval_macro_f1" in e]
    if len(epoch_logs) >= 2:
        ep = [e["epoch"] for e in epoch_logs]
        plt.figure(figsize=(10, 5))
        for key, lbl in [("eval_accuracy", "Accuracy"), ("eval_macro_f1", "Macro F1"),
                         ("eval_weighted_f1", "Weighted F1"), ("eval_macro_precision", "Macro Precision"),
                         ("eval_macro_recall", "Macro Recall")]:
            plt.plot(ep, [e.get(key, 0) for e in epoch_logs], marker="o", label=lbl)
        plt.ylim(0, 1.05); plt.xlabel("Epoch"); plt.ylabel("Score"); plt.grid(alpha=0.3); plt.legend()
        plt.title(f"Validation Metrics per Epoch - PROMISE {args.mode}")
        plt.tight_layout(); plt.savefig(output_dir / "epoch_metrics.png", dpi=150); plt.close()

    # ---------- Save model ----------
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    # ---------- Test predictions sheet ----------
    probs = torch.softmax(torch.tensor(test_pred.predictions), dim=1).numpy()
    true_dec = label_encoder.inverse_transform(y_true)
    pred_dec = label_encoder.inverse_transform(y_pred)
    pd.DataFrame({
        "Requirement":     test_df[TEXT_COL].values,
        "True_Label":      true_dec,
        "Predicted_Label": pred_dec,
        "Correct":         np.where(true_dec == pred_dec, "YES", "NO"),
        "Confidence_%":    (probs.max(axis=1) * 100).round(2),
    }).sort_values("Correct").to_excel(output_dir / "test_predictions.xlsx", index=False)

    # ---------- Results log ----------
    report_dict = classification_report(y_true, y_pred, labels=labels_idx, target_names=sorted_names,
                                        zero_division=0, output_dict=True)
    row = {
        "Timestamp": f"{datetime.now():%Y-%m-%d %H:%M}", "Mode": args.mode, "Model": model_name,
        "Num_Classes": num_labels, "Classes": str(sorted_classes), "Split": cfg["split"],
        "Train_Size": len(train_df), "Val_Size": len(val_df), "Test_Size": len(test_df),
        "Epochs_Trained": len(epoch_logs), "LR": cfg["lr"], "Loss": cfg["loss"],
        "Oversampling_Min": cfg["oversample_min"],
    }
    for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1",
              "weighted_precision", "weighted_recall", "weighted_f1"]:
        row[k] = round(float(test_results.get(f"eval_{k}", 0)), 4)
    for n in sorted_names:
        row[f"F1_{n.replace(' ', '_')}"] = round(report_dict[n]["f1-score"], 4)
    append_to_excel(row, output_dir.parent / "promise_results.xlsx")

    print(f"\nDone. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()