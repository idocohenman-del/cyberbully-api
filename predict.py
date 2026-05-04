"""
predict.py
──────────
Run inference with a saved model (TF-IDF pipeline or DistilBERT).

The script auto-detects which model to load: DistilBERT takes priority if the
models/distilbert_cyberbullying/ directory exists, otherwise it falls back to
the most recently saved TF-IDF .joblib file.

Usage
-----
    python predict.py                              # interactive loop
    python predict.py "you are so stupid"         # single prediction + exit
    python predict.py --model-dir models/ "text"  # explicit directory

Programmatic usage after training
----------------------------------
    from predict import Predictor
    p = Predictor()
    result = p.predict("some message here")
    # {'label': 'BULLYING', 'is_bullying': True, 'confidence': 0.94, 'prob_bully': 0.94}
"""

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessor import TextPreprocessor

_MODEL_DIR = Path("models")

# TF-IDF path uses lemmatized text; BERT path uses non-lemmatized text.
# These match the preprocessing applied during training in train.py.
_preprocessor_tfidf = TextPreprocessor(remove_stopwords=False, lemmatize=True)
_preprocessor_bert  = TextPreprocessor(remove_stopwords=False, lemmatize=False)


class Predictor:
    """
    Unified predictor. Loads the best available model from model_dir
    (DistilBERT if present, otherwise the most recently saved joblib pipeline).

    Parameters
    ----------
    model_dir : directory that contains either
                  distilbert_cyberbullying/  (created by --model bert)
                  cyberbullying_*.joblib     (created by --model logreg/svm/naive_bayes)
    """

    def __init__(self, model_dir: Path = _MODEL_DIR):
        bert_path  = model_dir / "distilbert_cyberbullying"
        if bert_path.exists():
            self._load_bert(bert_path)
        else:
            self._load_tfidf(model_dir)

    # ── loaders ───────────────────────────────────────────────────────────────

    def _load_bert(self, path: Path):
        from src.bert_trainer import BertPredictor
        self._backend = BertPredictor(path)
        self._kind    = "bert"
        print(f"  Loaded DistilBERT from {path}")

    def _load_tfidf(self, model_dir: Path):
        import joblib

        matches = sorted(glob.glob(str(model_dir / "cyberbullying_*.joblib")))
        if not matches:
            raise FileNotFoundError(
                f"No saved model found in '{model_dir}'.\n"
                "Train one first:  python train.py"
            )
        path           = matches[-1]   # use the most recently saved file
        self._pipeline = joblib.load(path)
        self._kind     = "tfidf"
        print(f"  Loaded TF-IDF model from {path}")

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Classify a single piece of text.

        Returns
        -------
        dict with keys:
            label        — "BULLYING" or "SAFE"
            is_bullying  — bool
            confidence   — probability of the predicted class (float 0-1)
            prob_bully   — P(cyberbullying) (float 0-1)
        """
        if self._kind == "bert":
            return self._backend.predict(text)   # BertPredictor cleans internally
        else:
            return self._predict_tfidf(text)

    def _predict_tfidf(self, text: str) -> dict:
        cleaned = _preprocessor_tfidf.clean(text)
        if not cleaned:
            return {"label": "SAFE", "is_bullying": False, "confidence": 1.0, "prob_bully": 0.0}

        label = int(self._pipeline.predict([cleaned])[0])
        try:
            probs = self._pipeline.predict_proba([cleaned])[0]
            conf  = float(probs[label])
            p_b   = float(probs[1])
        except AttributeError:
            # LinearSVC without calibration has no predict_proba
            conf, p_b = None, None

        return {
            "label":       "BULLYING" if label else "SAFE",
            "is_bullying": bool(label),
            "confidence":  conf,
            "prob_bully":  p_b,
        }


# ── interactive terminal loop ─────────────────────────────────────────────────

def repl(predictor: Predictor):
    """Run an interactive classification loop in the terminal."""
    print("\n" + "═" * 58)
    print("  Cyberbullying Classifier  —  Interactive Mode")
    print("  Type a message to classify.  Enter 'quit' to exit.")
    print("═" * 58)

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            break

        r    = predictor.predict(raw)
        tag  = "BULLYING" if r["is_bullying"] else "SAFE"
        conf = f"  Confidence : {r['confidence']:.1%}" if r["confidence"] is not None else ""
        pb   = f"  P(bully)   : {r['prob_bully']:.1%}" if r["prob_bully"]  is not None else ""
        print(f"  Verdict    : {tag}")
        if conf:
            print(conf)
        if pb:
            print(pb)

    print("Goodbye.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Classify text for cyberbullying.")
    p.add_argument(
        "text", nargs="?", default=None,
        help="Text to classify (omit to enter interactive mode).",
    )
    p.add_argument(
        "--model-dir", type=Path, default=_MODEL_DIR,
        help=f"Directory containing the saved model (default: {_MODEL_DIR})",
    )
    args = p.parse_args()

    predictor = Predictor(args.model_dir)

    if args.text:
        r = predictor.predict(args.text)
        print(f"\n  Input      : {args.text}")
        print(f"  Verdict    : {r['label']}")
        if r["confidence"] is not None:
            print(f"  Confidence : {r['confidence']:.1%}")
        if r["prob_bully"] is not None:
            print(f"  P(bully)   : {r['prob_bully']:.1%}")
        print()
    else:
        repl(predictor)


if __name__ == "__main__":
    main()
