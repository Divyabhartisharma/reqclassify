"""
==================
BERT fine-tuning on the User Story dataset (1600).
Model: Bert Base Uncased(FR, NFR)
Expected CSV columns: "Requirement" (text), "Type" (label)
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
#from Hugging face
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
# Project root:
#   <root>/1_data/raw/labeled/...
#   <root>/2_experiments/bert/scripts/...py
#   <root>/3_models/bert/...
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR     = PROJECT_ROOT / "1_data" / "raw" / "labeled"
MODELS_DIR   = PROJECT_ROOT / "3_models" / "bert"

# =========================================================
# EXPERIMENT CONFIGS
# =========================================================
CONFIGS = {
    "binary": dict(
        data=DATA_DIR / "User_story" / "UserStory_FR_vs_NFR.csv",
        run_name="UserStory_FR_vs_NFR",
        split="80_20", max_length=128, epochs=4,
        dropout=None, weighted_loss=False, patience=None,
        weight_decay=0.0, fp16=False,
    ),
    "8class": dict(
        data=DATA_DIR / "User_story" / "UserStory_Multi_8classes.csv",
        run_name="UserStory_Multi_8classes",
        split="60_20_20", max_length=384, epochs=20,
        dropout=0.2, weighted_loss=True, patience=3,
        weight_decay=0.01, fp16=True,
    ),
    "nfr": dict(
        data=DATA_DIR / "10006_dataset_Multi_NFRclass.csv",
        run_name="UserStory_NFRclass",
        split="60_20_20", max_length=384, epochs=20,
        dropout=0.2, weighted_loss=True, patience=3,
        weight_decay=0.01, fp16=True,
    ),
}

MODEL_NAME = "bert-base-uncased"
TEXT_COL   = "Requirement"
LABEL_COL  = "Type"
LR         = 2e-5
TRAIN_BS   = 8
EVAL_BS    = 8
SEED       = 42


# =========================================================
# generating token and batches
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


class WeightedTrainer(Trainer):
    """Trainer with class-weighted."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        loss    = nn.CrossEntropyLoss(weight=self.class_weights)(outputs.get("logits"), labels)
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
    parser = argparse.ArgumentParser(description="BERT - User Story dataset")
    parser.add_argument("--mode", choices=CONFIGS.keys(), required=True,
                        help="binary = FR vs NFR | 8class | nfr = NFR subclasses")
    parser.add_argument("--data", default=None, help="Override dataset CSV path")
    parser.add_argument("--output_dir", default=None, help="Override output folder")
    parser.add_argument("--model_name", default=MODEL_NAME, help="HF model (default bert-base-uncased)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    cfg        = CONFIGS[args.mode]
    data_path  = Path(args.data) if args.data else cfg["data"]
    output_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR / "bert_runs" / cfg["run_name"]
    epochs     = args.epochs or cfg["epochs"]
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: {args.mode} | Device: {device}\nData: {data_path}\nOutput: {output_dir}\n")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}\nUse --data to point to the CSV.")

    # ---------- Load ----------
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    df[TEXT_COL]  = df[TEXT_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    print("Dataset distribution:\n", df[LABEL_COL].value_counts(), "\n")

    # ---------- Labels ----------
    # labels text into numbers
    label_encoder = LabelEncoder().fit(sorted(df[LABEL_COL].unique()))
    classes       = label_encoder.classes_.tolist()
    num_labels    = len(classes)
    df["label_id"] = label_encoder.transform(df[LABEL_COL])
    print("Classes:", classes)

    # ---------- Split ----------
    if cfg["split"] == "80_20":
        train_df, test_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["label_id"])
        val_df = test_df
    else:
        train_df, temp_df = train_test_split(df, test_size=0.40, random_state=SEED, stratify=df["label_id"])
        val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label_id"])
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}\n")

    # ---------- Class weights ----------
    class_weights = None
    if cfg["weighted_loss"]:
        w = compute_class_weight("balanced", classes=np.arange(num_labels), y=train_df["label_id"].values)
        class_weights = torch.tensor(w, dtype=torch.float).to(device)

    # ---------- Tokenizer / datasets ----------
    tokenizer     = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    ml = cfg["max_length"]
    train_ds = ReqDataset(train_df[TEXT_COL], train_df["label_id"], tokenizer, ml)
    val_ds   = ReqDataset(val_df[TEXT_COL],   val_df["label_id"],   tokenizer, ml)
    test_ds  = ReqDataset(test_df[TEXT_COL],  test_df["label_id"],  tokenizer, ml)

    # ---------- Model ----------
    model_kwargs = {"num_labels": num_labels}
    if cfg["dropout"] is not None:
        model_kwargs.update(hidden_dropout_prob=cfg["dropout"], attention_probs_dropout_prob=cfg["dropout"])
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, **model_kwargs).to(device)

    # ---------- Trainer ----------
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        learning_rate=LR,
        weight_decay=cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_only_model=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=SEED,
        fp16=cfg["fp16"] and torch.cuda.is_available(),
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg["patience"])] if cfg["patience"] else []

    trainer = WeightedTrainer(
        model=model, args=training_args, data_collator=data_collator,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics, callbacks=callbacks,
        class_weights=class_weights,   # None -> plain CrossEntropy
    )

    trainer.train()

    # ---------- Test evaluation ----------
    test_results = trainer.evaluate(eval_dataset=test_ds)
    y_pred = np.argmax(trainer.predict(test_ds).predictions, axis=1)
    y_true = test_df["label_id"].values

    report = classification_report(y_true, y_pred, labels=np.arange(num_labels),
                                   target_names=classes, zero_division=0, digits=4)
    print("\n=== Classification Report (Test) ===\n", report)
    (output_dir / "classification_report.txt").write_text(
        f"Mode: {args.mode}\nModel: {args.model_name}\nData: {data_path.name}\n"
        f"Date: {datetime.now():%Y-%m-%d %H:%M}\n\n{report}"
    )

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_labels))
    plt.figure(figsize=(max(8, num_labels), max(6, num_labels - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - UserStory {args.mode} (Test)")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_test.png", dpi=150); plt.close()

    # ---------- Save ----------
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    pred_df = test_df[[TEXT_COL, LABEL_COL]].copy()
    pred_df["Predicted"] = label_encoder.inverse_transform(y_pred)
    pred_df["Correct"]   = pred_df["Predicted"] == pred_df[LABEL_COL]
    pred_df.to_excel(output_dir / "test_predictions.xlsx", index=False)

    append_to_excel({
        "Timestamp": f"{datetime.now():%Y-%m-%d %H:%M}",
        "Mode": args.mode, "Model": args.model_name, "Num_Classes": num_labels,
        "Split": cfg["split"], "Epochs": epochs, "LR": LR,
        "Train_Size": len(train_df), "Val_Size": len(val_df), "Test_Size": len(test_df),
        **{f"Test_{k.replace('eval_', '')}": round(float(v), 4)
           for k, v in test_results.items() if k.startswith("eval_") and "runtime" not in k
           and "per_second" not in k and k != "eval_epoch"},
    }, output_dir.parent / "userstory_results.xlsx")

    print(f"\nDone. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()