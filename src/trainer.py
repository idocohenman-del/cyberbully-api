"""
trainer.py
──────────
Builds and trains a TF-IDF + classifier sklearn Pipeline.

Supported classifiers (pass as `model_type` string):
  "logreg"      → Logistic Regression  (fast, strong baseline, interpretable)
  "svm"         → Linear SVC           (often best for text classification)
  "naive_bayes" → Multinomial NB       (very fast, decent baseline)

Class imbalance is handled via two complementary strategies:
  1. class_weight='balanced'  — always on; re-weights the loss per sample
  2. SMOTE oversampling       — optional; adds synthetic minority-class examples
     between the TF-IDF step and the classifier using an imblearn Pipeline.
     Requires: pip install imbalanced-learn
"""

import joblib
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV


def build_pipeline(model_type: str = "logreg", use_smote: bool = False) -> object:
    """
    Return an untrained Pipeline ready for .fit().

    When use_smote=True the pipeline is an imblearn Pipeline that inserts a
    SMOTE resampler between TF-IDF and the classifier.  This is the correct
    place for SMOTE because it operates on the numerical TF-IDF feature
    vectors, not on raw text strings.

    Steps:
      tfidf  → (smote) → clf
    """
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),      # unigrams + bigrams capture short phrases
        max_features=100_000,    # cap vocabulary to keep memory predictable
        sublinear_tf=True,       # log(1+tf) dampens very high-frequency terms
        min_df=2,                # ignore terms appearing in only one document
        strip_accents="unicode",
        decode_error="replace",
    )

    if model_type == "logreg":
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="saga",       # saga handles large sparse matrices well
            n_jobs=-1,
        )
    elif model_type == "svm":
        # LinearSVC has no predict_proba; wrap with calibration to enable it
        base = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")
        clf  = CalibratedClassifierCV(base, cv=3)
    elif model_type == "naive_bayes":
        # ComplementNB is designed for imbalanced text classification
        clf = ComplementNB(alpha=0.1)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose from: 'logreg', 'svm', 'naive_bayes'."
        )

    if use_smote:
        try:
            from imblearn.pipeline import Pipeline as ImbPipeline
            from imblearn.over_sampling import SMOTE

            # imblearn's Pipeline supports Resampler steps between transforms
            pipeline = ImbPipeline([
                ("tfidf", tfidf),
                ("smote", SMOTE(random_state=42)),
                ("clf",   clf),
            ])
            print("  SMOTE oversampling enabled (imbalanced-learn pipeline).")
        except ImportError:
            print(
                "  WARNING: imbalanced-learn not installed; SMOTE skipped.\n"
                "  Run:  pip install imbalanced-learn"
            )
            pipeline = SklearnPipeline([("tfidf", tfidf), ("clf", clf)])
    else:
        pipeline = SklearnPipeline([("tfidf", tfidf), ("clf", clf)])

    return pipeline


def train(pipeline, X_train, y_train):
    """Fit the pipeline on training data and return it."""
    clf_name = pipeline.named_steps["clf"].__class__.__name__
    print(f"  Training {clf_name} ...")
    pipeline.fit(X_train, y_train)
    print("  Training complete.")
    return pipeline


def save_model(pipeline, path: str) -> None:
    joblib.dump(pipeline, path)
    print(f"  Model saved → {path}")


def load_model(path: str):
    return joblib.load(path)
