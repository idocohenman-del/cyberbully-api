import os
import re
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_all_datasets

# ── Config ─────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
OUTPUT_DIR   = "model/distilbert"
MAX_LEN      = 128
BATCH_SIZE   = 32
EPOCHS       = 3
# Set to 1.0 to use all data; reduce for faster CPU runs (0.2 ≈ 80k rows, ~15 h on CPU)
SAMPLE_FRAC  = 0.2

# ── Clean text ─────────────────────────────────────────────────
def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Metrics ────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

# ── Main ───────────────────────────────────────────────────────
def main():
    print("\n[ Step 1 ] Loading all datasets...")
    df = load_all_datasets()

    if SAMPLE_FRAC < 1.0:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(frac=SAMPLE_FRAC, random_state=42)
        ).reset_index(drop=True)
        print(f"  Sampled {SAMPLE_FRAC:.0%} → {len(df):,} rows (stratified by label)")

    print("\n[ Step 2 ] Cleaning text...")
    df["text"] = df["text"].apply(clean)
    df = df[df["text"].str.len() > 3].reset_index(drop=True)
    print(f"  Rows after cleaning: {len(df):,}")

    print("\n[ Step 3 ] Splitting (70/15/15)...")
    X = df["text"].values
    y = df["label"].values
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\n[ Step 4 ] Tokenizing...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize(texts, labels):
        enc = tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LEN)
        enc["labels"] = list(labels)
        return Dataset.from_dict(enc)

    train_dataset = tokenize(X_train, y_train)
    val_dataset   = tokenize(X_val,   y_val)
    test_dataset  = tokenize(X_test,  y_test)

    print("\n[ Step 5 ] Loading DistilBERT model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using: {device.upper()}")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    print("\n[ Step 6 ] Training...")
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=500,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),   # faster if GPU
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    print("\n[ Step 7 ] Test results...")
    preds_output = trainer.predict(test_dataset)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    print(classification_report(y_test, y_pred, labels=[0,1], target_names=["Safe", "Bullying"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\n[ Step 8 ] Saving model to {OUTPUT_DIR} ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("  ✅ Model saved!")

    print("\n[ Ready ] Type a sentence to classify (or 'quit' to exit):")
    model.eval()
    while True:
        text = input(">>> ").strip()
        if text.lower() == "quit":
            break
        cleaned = clean(text)
        inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
        prob = torch.softmax(logits, dim=-1)[0][pred].item()
        label = "🚨 BULLYING" if pred == 1 else "✅ SAFE"
        print(f"  {label}  (confidence: {prob:.1%})")

if __name__ == "__main__":
    main()