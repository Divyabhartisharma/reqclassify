"""
Usage
-----
    # Train
    python train_autosar.py --mode binary
    python train_autosar.py --mode fr          # FR subtypes only
    python train_autosar.py --mode nfr         # NFR subtypes only
    python train_autosar.py --mode all         # all subtypes together

CSV columns:
    binary : "Requirement Text", "Type" (FR / NFR)
    fr/nfr/all : "Requirement Text", "Type", "Subtype"
"""

import argparse
import warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_fscore_support, classification_report, confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)

# =========================================================
# PATHS- relative to project root
#   <root>/1_data/raw/labeled/Autosar/...
#   <root>/2_experiments/bert/scripts/..
#   <root>/3_models/bert/...
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR     = PROJECT_ROOT / "1_data" / "raw" / "labeled" / "Autosar"
MODELS_DIR   = PROJECT_ROOT / "3_models" / "bert"

BINARY_CSV  = DATA_DIR / "Autosar.csv"
SUBTYPE_CSV = DATA_DIR / "Autosar_labels.csv"

COL_TEXT    = "Requirement Text"
COL_TYPE    = "Type"
COL_SUBTYPE = "Subtype"

MODEL_NAME = "bert-base-uncased"
SEED       = 42

# Binary (FR vs NFR) - custom training loop, same as original Autosar.py
BIN = dict(max_len=128, batch_size=8, epochs=4, lr=2e-5)
LABEL2ID = {"FR": 0, "NFR": 1}
ID2LABEL = {0: "FR", 1: "NFR"}

# Subtypes - HF Trainer, same as original Autosar_subtype.py
SUB = dict(max_len=128, batch_size=8, epochs=20, lr=2e-5, weight_decay=0.01, patience=3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_output_dir(mode):
    return MODELS_DIR / "bert_runs" / ("autosar_binary" if mode == "binary" else f"autosar_subtype_{mode}")


def append_to_excel(row: dict, path: Path):
    df_new = pd.DataFrame([row])
    if path.exists():
        df_new = pd.concat([pd.read_excel(path), df_new], ignore_index=True)
    df_new.to_excel(path, index=False)


def read_csv_checked(csv_path: Path, required_cols):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}\nUse --data to point to the CSV.")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")
    return df


def save_confusion_matrix(y_true, y_pred, class_names, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    n = len(class_names)
    plt.figure(figsize=(max(6, n), max(5, n - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def metric_row(y_true, y_pred):
    return {
        "accuracy":           accuracy_score(y_true, y_pred),
        "macro_precision":    precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "macro_recall":       recall_score   (y_true, y_pred, average="macro",    zero_division=0),
        "macro_f1":           f1_score       (y_true, y_pred, average="macro",    zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall":    recall_score   (y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1":        f1_score       (y_true, y_pred, average="weighted", zero_division=0),
    }


class ReqDataset(torch.utils.data.Dataset):
    """Returns dicts with input_ids / attention_mask / labels (dynamic padding via collator)."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = list(texts), [int(x) for x in labels]
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_len)
        enc["labels"] = self.labels[idx]
        return enc


# =========================================================
# 1) BINARY  FR vs NFR   (custom PyTorch loop)
# =========================================================
def load_binary(csv_path: Path):
    df = read_csv_checked(csv_path, [COL_TEXT, COL_TYPE])
    df[COL_TYPE] = df[COL_TYPE].astype(str).str.strip().str.upper()
    bad = df[~df[COL_TYPE].isin(LABEL2ID)]
    if len(bad):
        print(f"  Warning: {len(bad)} rows with unknown labels removed: {bad[COL_TYPE].unique()}")
    df = df[df[COL_TYPE].isin(LABEL2ID)].dropna(subset=[COL_TEXT]).reset_index(drop=True)
    df[COL_TEXT] = df[COL_TEXT].astype(str).str.strip()
    df = df[df[COL_TEXT] != ""].reset_index(drop=True)
    df["label_id"] = df[COL_TYPE].map(LABEL2ID)
    print(f"  Total: {len(df)} | FR: {(df[COL_TYPE]=='FR').sum()} | NFR: {(df[COL_TYPE]=='NFR').sum()}")
    return train_test_split(df[COL_TEXT].tolist(), df["label_id"].tolist(),
                            test_size=0.20, random_state=SEED, stratify=df["label_id"].tolist())


def evaluate_loader(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch.pop("labels")
            out = model(**batch)
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return np.array(labels), np.array(preds)


def train_binary(csv_path: Path, output_dir: Path, model_name: str, epochs: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID).to(device)

    X_train, X_test, y_train, y_test = load_binary(csv_path)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(ReqDataset(X_train, y_train, tokenizer, BIN["max_len"]),
                              batch_size=BIN["batch_size"], shuffle=True, collate_fn=collator)
    test_loader  = DataLoader(ReqDataset(X_test, y_test, tokenizer, BIN["max_len"]),
                              batch_size=BIN["batch_size"], shuffle=False, collate_fn=collator)

    optimizer   = AdamW(model.parameters(), lr=BIN["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item()

        y_true, y_pred = evaluate_loader(model, test_loader)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        saved = ""
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(output_dir); tokenizer.save_pretrained(output_dir)
            saved = "  <- best model saved"
        print(f"  Epoch {epoch}/{epochs} | Loss {total_loss/len(train_loader):.4f} | "
              f"Acc {accuracy_score(y_true, y_pred):.4f} | P {p:.4f} | R {r:.4f} | F1 {f1:.4f}{saved}")

    # Final evaluation with best checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    y_true, y_pred = evaluate_loader(model, test_loader)
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["FR", "NFR"],
                                   digits=4, zero_division=0)
    print("\n  Classification Report (Test):\n", report)
    (output_dir / "classification_report.txt").write_text(report)
    save_confusion_matrix(y_true, y_pred, ["FR", "NFR"], "Confusion Matrix - AUTOSAR FR vs NFR",
                          output_dir / "confusion_matrix_test.png")

    append_to_excel({"Timestamp": f"{datetime.now():%Y-%m-%d %H:%M}", "Mode": "binary",
                     "Model": model_name, "Num_Classes": 2, "Train_Size": len(X_train),
                     "Test_Size": len(X_test), "Epochs": epochs,
                     **{k: round(v, 4) for k, v in metric_row(y_true, y_pred).items()}},
                    output_dir.parent / "autosar_results.xlsx")
    print(f"\n  Model saved to: {output_dir}")


# =========================================================
# 2) SUBTYPES  (FR / NFR / all)  - HF Trainer + class weights
# =========================================================
class WeightedTrainer(Trainer):
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
    return metric_row(labels, np.argmax(logits, axis=1))


def load_subtypes(csv_path: Path, type_filter):
    df = read_csv_checked(csv_path, [COL_TEXT, COL_TYPE, COL_SUBTYPE])
    df = df.dropna(subset=[COL_TEXT, COL_SUBTYPE]).reset_index(drop=True)
    df[COL_TEXT]    = df[COL_TEXT].astype(str).str.strip()
    df[COL_TYPE]    = df[COL_TYPE].astype(str).str.strip().str.upper()
    df[COL_SUBTYPE] = df[COL_SUBTYPE].astype(str).str.strip()
    df = df[(df[COL_TEXT] != "") & (df[COL_SUBTYPE] != "") &
            (df[COL_SUBTYPE].str.lower() != "nan")].reset_index(drop=True)
    if type_filter:
        df = df[df[COL_TYPE] == type_filter].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows with {COL_TYPE} = '{type_filter}'")
    print(f"  Total: {len(df)}\n  Subtype distribution:\n{df[COL_SUBTYPE].value_counts().to_string()}\n")
    return df


def train_subtypes(csv_path: Path, output_dir: Path, mode: str, model_name: str, epochs: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    type_filter = {"fr": "FR", "nfr": "NFR", "all": None}[mode]
    df = load_subtypes(csv_path, type_filter)

    rare = df[COL_SUBTYPE].value_counts()
    rare = rare[rare < 5]
    if len(rare):
        print(f"  WARNING - very few samples: {rare.to_dict()}")

    label_encoder = LabelEncoder().fit(sorted(df[COL_SUBTYPE].unique()))
    classes    = label_encoder.classes_.tolist()
    num_labels = len(classes)
    df["label_id"] = label_encoder.transform(df[COL_SUBTYPE])
    print(f"  Classes ({num_labels}): {classes}\n")

    strat = df["label_id"].value_counts().min() >= 2
    if not strat:
        print("  Warning: some classes have 1 sample - stratified split disabled.")
    train_df, temp_df = train_test_split(df, test_size=0.40, random_state=SEED,
                                         stratify=df["label_id"] if strat else None)
    try:
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED,
                                           stratify=temp_df["label_id"] if strat else None)
    except ValueError:   # a class has only 1 sample in temp_df
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}\n")

    # Balanced class weights; classes missing from train get weight 1.0
    counts = np.bincount(train_df["label_id"].values, minlength=num_labels)
    w = np.where(counts == 0, 1.0, counts.sum() / (num_labels * np.maximum(counts, 1)))
    class_weights = torch.tensor(w, dtype=torch.float).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ml = SUB["max_len"]
    train_ds = ReqDataset(train_df[COL_TEXT], train_df["label_id"], tokenizer, ml)
    val_ds   = ReqDataset(val_df[COL_TEXT],   val_df["label_id"],   tokenizer, ml)
    test_ds  = ReqDataset(test_df[COL_TEXT],  test_df["label_id"],  tokenizer, ml)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
        hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2).to(device)

    trainer = WeightedTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=SUB["batch_size"],
            per_device_eval_batch_size=SUB["batch_size"],
            learning_rate=SUB["lr"], weight_decay=SUB["weight_decay"],
            eval_strategy="epoch", save_strategy="epoch",
            save_only_model=True, save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
            report_to="none", seed=SEED, fp16=torch.cuda.is_available(),
        ),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=SUB["patience"])],
        class_weights=class_weights,
    )
    trainer.train()

    test_results = trainer.evaluate(eval_dataset=test_ds)
    y_pred = np.argmax(trainer.predict(test_ds).predictions, axis=1)
    y_true = test_df["label_id"].values

    report = classification_report(y_true, y_pred, labels=np.arange(num_labels),
                                   target_names=classes, zero_division=0, digits=4)
    print("\n  Classification Report (Test):\n", report)
    (output_dir / "classification_report.txt").write_text(report)
    save_confusion_matrix(y_true, y_pred, classes, f"Confusion Matrix - AUTOSAR {mode.upper()} subtypes",
                          output_dir / "confusion_matrix_test.png")

    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    append_to_excel({"Timestamp": f"{datetime.now():%Y-%m-%d %H:%M}", "Mode": f"subtype_{mode}",
                     "Model": model_name, "Num_Classes": num_labels, "Train_Size": len(train_df),
                     "Val_Size": len(val_df), "Test_Size": len(test_df), "Epochs": epochs,
                     **{k.replace("eval_", ""): round(float(v), 4) for k, v in test_results.items()
                        if k.replace("eval_", "") in metric_row([0], [0])}},
                    output_dir.parent / "autosar_results.xlsx")
    print(f"\n  Model saved to: {output_dir}")


# =========================================================
# 3) PREDICTION  (works for all modes)
# =========================================================
def load_trained(mode: str, output_dir: Path):
    if mode == "binary":
        model_dir, tok_dir, names = output_dir, output_dir, [ID2LABEL[0], ID2LABEL[1]]
    else:
        model_dir, tok_dir = output_dir / "model", output_dir / "tokenizer"
        names = None
    if not model_dir.exists():
        raise FileNotFoundError(f"No trained model at {model_dir}. Train first: --mode {mode}")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    if names is None:
        names = joblib.load(output_dir / "label_encoder.joblib").classes_.tolist()
    return tokenizer, model, names


def predict_probs(texts, tokenizer, model, max_len=128, batch_size=32):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            all_probs.append(torch.softmax(model(**enc).logits, dim=1).cpu().numpy())
    return np.vstack(all_probs)


def predict_one(text, mode, output_dir):
    tokenizer, model, names = load_trained(mode, output_dir)
    probs = predict_probs([text], tokenizer, model)[0]
    order = np.argsort(probs)[::-1]
    print(f"\n  Requirement : {text}\n  Prediction  : {names[order[0]]} ({probs[order[0]]*100:.2f}%)\n")
    for i in order:
        print(f"    {names[i]:<28} {probs[i]*100:6.2f}%")


def predict_csv(csv_path, mode, output_dir, out_path=None):
    df = read_csv_checked(Path(csv_path), [COL_TEXT])
    tokenizer, model, names = load_trained(mode, output_dir)
    probs = predict_probs(df[COL_TEXT].astype(str).tolist(), tokenizer, model)
    df["Predicted"]    = [names[i] for i in probs.argmax(axis=1)]
    df["Confidence %"] = (probs.max(axis=1) * 100).round(2)
    out_path = Path(out_path) if out_path else Path(csv_path).with_name(Path(csv_path).stem + f"_pred_{mode}.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}\n{df['Predicted'].value_counts().to_string()}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT - AUTOSAR dataset",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--mode", choices=["binary", "fr", "nfr", "all"], required=True,
                        help="binary = FR vs NFR\nfr = FR subtypes\nnfr = NFR subtypes\nall = all subtypes")
    parser.add_argument("--data", default=None, help="Override training CSV path")
    parser.add_argument("--output_dir", default=None, help="Override model/output folder")
    parser.add_argument("--model_name", default=MODEL_NAME, help="HF model (default bert-base-uncased)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--predict", metavar="TEXT", default=None, help="Classify one requirement")
    parser.add_argument("--batch_predict", metavar="CSV", default=None, help="Classify all rows of a CSV")
    parser.add_argument("--output", default=None, help="Output CSV path for --batch_predict")
    args = parser.parse_args()

    set_seed(SEED)
    torch.manual_seed(SEED)
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.mode)
    print(f"Mode: {args.mode} | Device: {device} | Output: {out_dir}\n")

    if args.predict:
        predict_one(args.predict, args.mode, out_dir)
    elif args.batch_predict:
        predict_csv(args.batch_predict, args.mode, out_dir, args.output)
    elif args.mode == "binary":
        train_binary(Path(args.data) if args.data else BINARY_CSV, out_dir,
                     args.model_name, args.epochs or BIN["epochs"])
    else:
        train_subtypes(Path(args.data) if args.data else SUBTYPE_CSV, out_dir, args.mode,
                       args.model_name, args.epochs or SUB["epochs"])