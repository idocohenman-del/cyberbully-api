"""
bert_trainer.py
───────────────
Fine-tunes distilbert-base-uncased for binary cyberbullying classification.

Key design decisions:
  - Weighted cross-entropy loss handles class imbalance without resampling.
    SMOTE on raw text doesn't make sense (there's no meaningful interpolation
    between token sequences); weighted loss achieves the same goal at the
    gradient level without modifying the data.
  - EarlyStoppingCallback monitors validation F1 and halts training when it
    stops improving, preventing overfitting on the training set.
  - fp16=True is enabled automatically on CUDA to halve VRAM usage and speed
    up training with no accuracy cost on modern GPUs.
  - dataloader_num_workers=0 is required on Windows to avoid multiprocessing
    issues with the PyTorch DataLoader.

Requirements:
    pip install torch transformers>=4.40.0 accelerate>=0.24.0
    (For GPU: install PyTorch with CUDA from https://pytorch.org/get-started/locally/)
"""

import logging
import re
import html
import unicodedata
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Default hyperparameters — all overridable via **kwargs in train()
_DEFAULTS = dict(
    model_name              = "distilbert-base-uncased",
    max_length              = 128,
    batch_size              = 16,
    num_epochs              = 5,
    learning_rate           = 2e-5,
    weight_decay            = 0.01,
    warmup_ratio            = 0.06,
    early_stopping_patience = 2,
)


def is_available() -> bool:
    """Return True when torch and transformers are importable."""
    try:
        import torch          # noqa: F401
        import transformers   # noqa: F401
        return True
    except ImportError:
        return False


# ── simple text cleaner (mirrors TextPreprocessor without lemmatization) ──────
# Kept inline so bert_trainer.py is self-contained for the BertPredictor class.

def _clean(text: str) -> str:
    """Normalise a social-media post the same way training data was cleaned."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"@\w+", " ", text)                     # @mentions
    text = re.sub(r"#(\w+)", r"\1", text)                 # #hashtag → hashtag
    text = re.sub(r"[^a-z0-9\s]", " ", text)             # special chars
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)           # loooove → loove
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip()


# ── training ──────────────────────────────────────────────────────────────────

def train(
    X_train: list,
    y_train: list,
    X_val:   list,
    y_val:   list,
    X_test:  list,
    y_test:  list,
    save_dir: Path,
    **kwargs,
):
    """
    Fine-tune DistilBERT on the supplied text/label lists.

    Parameters
    ----------
    X_train, y_train : training texts and integer labels (0/1)
    X_val,   y_val   : validation texts and labels (used for early stopping)
    X_test,  y_test  : held-out test texts and labels (final evaluation only)
    save_dir         : directory that will receive checkpoints/ and the
                       final distilbert_cyberbullying/ sub-directory
    **kwargs         : override any key from _DEFAULTS (e.g. num_epochs=3)
    """
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, accuracy_score,
    )
    from sklearn.utils.class_weight import compute_class_weight
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    cfg    = {**_DEFAULTS, **kwargs}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device.upper())
    if device == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        log.warning(
            "No GPU detected — DistilBERT on CPU can take several hours on large datasets. "
            "Consider --model logreg for a CPU-friendly alternative."
        )

    # ── class weights for imbalanced data ─────────────────────────────────────
    # Weight the loss so that each bullying example counts more when it is
    # the minority class, without duplicating or removing any data.
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.array(y_train))
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(device)
    log.info("Class weights  →  safe=%.3f  bullying=%.3f", cw[0], cw[1])

    # ── tokenise ──────────────────────────────────────────────────────────────
    log.info("Loading tokenizer: %s", cfg["model_name"])
    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg["model_name"])

    class BullyDataset(Dataset):
        """Lazy-tokenising dataset — tokenises one text at a time in __getitem__
        so that 400k+ texts never need to be tokenised all at once in RAM."""
        def __init__(self, texts: list, labels: list):
            self.texts  = texts
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            enc  = tokenizer(
                self.texts[i],
                truncation=True,
                padding="max_length",
                max_length=cfg["max_length"],
            )
            item = {k: torch.tensor(v) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
            return item

    train_ds = BullyDataset(X_train, y_train)
    val_ds   = BullyDataset(X_val,   y_val)
    test_ds  = BullyDataset(X_test,  y_test)

    # ── model ─────────────────────────────────────────────────────────────────
    log.info("Loading %s …", cfg["model_name"])
    model = DistilBertForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2
    ).to(device)

    # Trainer subclass that replaces the default cross-entropy with a weighted version
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out    = model(**inputs)
            loss   = torch.nn.CrossEntropyLoss(weight=cw_tensor)(out.logits, labels)
            return (loss, out) if return_outputs else loss

    def compute_metrics(ep):
        preds = np.argmax(ep.predictions, axis=1)
        return {
            "accuracy":  accuracy_score( ep.label_ids, preds),
            "f1":        f1_score(       ep.label_ids, preds),
            "precision": precision_score(ep.label_ids, preds, zero_division=0),
            "recall":    recall_score(   ep.label_ids, preds, zero_division=0),
        }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = str(save_dir / "checkpoints"),
        num_train_epochs            = cfg["num_epochs"],
        per_device_train_batch_size = cfg["batch_size"],
        per_device_eval_batch_size  = cfg["batch_size"] * 2,
        learning_rate               = cfg["learning_rate"],
        weight_decay                = cfg["weight_decay"],
        warmup_ratio                = cfg["warmup_ratio"],
        eval_strategy               = "epoch",  # requires transformers>=4.40
        save_strategy               = "epoch",
        load_best_model_at_end      = True,     # restore best checkpoint after training
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        logging_steps               = 100,
        report_to                   = "none",   # disable wandb / tensorboard
        fp16                        = (device == "cuda"),
        dataloader_num_workers      = 0,        # must be 0 on Windows
    )

    trainer = WeightedTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
        callbacks       = [
            EarlyStoppingCallback(
                early_stopping_patience=cfg["early_stopping_patience"]
            )
        ],
    )

    log.info("Starting DistilBERT training …")
    trainer.train()

    # ── final evaluation on the held-out test set ─────────────────────────────
    log.info("Evaluating on held-out test set …")
    pred_out   = trainer.predict(test_ds)
    test_preds = np.argmax(pred_out.predictions, axis=1)

    print(f"\n{'─'*52}")
    print("  Test Evaluation (DistilBERT)")
    print(f"{'─'*52}")
    print(classification_report(
        y_test, test_preds,
        target_names=["Safe (0)", "Bullying (1)"],
        digits=4,
    ))

    # Save confusion matrix
    cm  = confusion_matrix(y_test, test_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Safe", "Bullying"]).plot(
        ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix — Test (DistilBERT)")
    cm_path = save_dir / "confusion_matrix_bert_test.png"
    fig.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info("Confusion matrix → %s", cm_path)

    # ── persist final model + tokenizer ───────────────────────────────────────
    model_path = save_dir / "distilbert_cyberbullying"
    model.save_pretrained(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    log.info("Model saved → %s", model_path)

    return model, tokenizer


# ── inference ─────────────────────────────────────────────────────────────────

class BertPredictor:
    """
    Load a saved DistilBERT model and run single-text inference.

    Usage
    -----
        p = BertPredictor("models/distilbert_cyberbullying")
        result = p.predict("you are so stupid and worthless")
        # {'label': 'BULLYING', 'is_bullying': True, 'confidence': 0.97, 'prob_bully': 0.97}
    """

    def __init__(self, model_dir, max_length: int = _DEFAULTS["max_length"]):
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

        model_dir       = str(model_dir)
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self._model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self._device    = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device).eval()
        self._max_len   = max_length

    def predict(self, text: str) -> dict:
        """
        Classify a single text string.

        Returns a dict with keys:
            label        — "BULLYING" or "SAFE"
            is_bullying  — bool
            confidence   — probability of the predicted class
            prob_bully   — P(cyberbullying)
        """
        import torch

        cleaned = _clean(text)
        if not cleaned:
            return {"label": "SAFE", "is_bullying": False, "confidence": 1.0, "prob_bully": 0.0}

        enc   = self._tokenizer(
            cleaned, return_tensors="pt", truncation=True, max_length=self._max_len
        )
        enc   = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred  = int(probs.argmax())

        return {
            "label":       "BULLYING" if pred else "SAFE",
            "is_bullying": bool(pred),
            "confidence":  float(probs[pred]),
            "prob_bully":  float(probs[1]),
        }
