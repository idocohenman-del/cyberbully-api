import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import re

from data_loader import load_all_datasets

# ── Text cleaning ──────────────────────────────────────────────
def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)                  # remove mentions
    text = re.sub(r"[^a-z\s]", "", text)              # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Main ───────────────────────────────────────────────────────
def main():
    print("\n[ Step 1 ] Loading all datasets...")
    df = load_all_datasets()

    print("\n[ Step 2 ] Cleaning text...")
    df["text"] = df["text"].apply(clean)
    df = df[df["text"].str.len() > 3].reset_index(drop=True)
    print(f"  Rows after cleaning: {len(df)}")

    print("\n[ Step 3 ] Splitting (70% train / 15% val / 15% test)...")
    X = df["text"].values
    y = df["label"].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Check imbalance
    ratio = np.sum(y_train == 1) / np.sum(y_train == 0)
    use_smote = ratio < 0.67 or ratio > 1.5
    print(f"  Class ratio (bullying/safe): {ratio:.2f} → SMOTE={'ON' if use_smote else 'OFF'}")

    print("\n[ Step 4 ] Building model pipeline...")
    steps = [
        ("tfidf", TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2
        )),
    ]

    if use_smote:
        pipeline = ImbPipeline(steps + [
            ("smote", SMOTE(random_state=42)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", n_jobs=-1))
        ])
    else:
        pipeline = Pipeline(steps + [
            ("clf", LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1))
        ])

    print("\n[ Step 5 ] Training...")
    pipeline.fit(X_train, y_train)

    print("\n[ Step 6 ] Validation results:")
    y_val_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_val_pred, labels=[0,1], target_names=["Safe", "Bullying"]))

    print("[ Step 7 ] Test results:")
    y_test_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_test_pred, labels=[0,1], target_names=["Safe", "Bullying"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    print("\n[ Step 8 ] Saving model to model/classifier.pkl ...")
    import os; os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/classifier.pkl")
    print("  ✅ Model saved!")

    # Interactive prediction
    print("\n[ Ready ] Type a sentence to classify (or 'quit' to exit):")
    while True:
        text = input(">>> ").strip()
        if text.lower() == "quit":
            break
        cleaned = clean(text)
        pred = pipeline.predict([cleaned])[0]
        prob = pipeline.predict_proba([cleaned])[0][pred]
        label = "🚨 BULLYING" if pred == 1 else "✅ SAFE"
        print(f"  {label}  (confidence: {prob:.1%})")

if __name__ == "__main__":
    main()