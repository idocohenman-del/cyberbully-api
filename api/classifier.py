import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_predictor  = None
_load_error = None

_MODEL_DIR = Path(
    os.getenv(
        "MODEL_DIR",
        str(Path(__file__).parent.parent / "models" / "distilbert_colab" / "distilbert_cyberbullying"),
    )
)


def _get_predictor():
    global _predictor, _load_error

    if _predictor is not None:
        return _predictor
    if _load_error is not None:
        raise _load_error

    src_dir = Path(__file__).parent.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    if not _MODEL_DIR.exists():
        _load_error = FileNotFoundError(
            f"Model not found at {_MODEL_DIR}. "
            "Download models/distilbert_colab/distilbert_cyberbullying/ from Google Drive "
            "and place it in the project's models/ folder."
        )
        raise _load_error

    try:
        from bert_trainer import BertPredictor
        _predictor = BertPredictor(str(_MODEL_DIR))
        log.info("BertPredictor loaded from %s", _MODEL_DIR)
    except Exception as exc:
        _load_error = exc
        raise

    return _predictor


def classify(text: str) -> dict:
    return _get_predictor().predict(text)


def is_model_ready() -> bool:
    try:
        _get_predictor()
        return True
    except Exception:
        return False
