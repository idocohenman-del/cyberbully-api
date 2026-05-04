from fastapi import APIRouter, HTTPException
from ..schemas import ClassifyRequest, ClassifyResponse
from .. import classifier

router = APIRouter(prefix="/classify", tags=["classify"])

BULLY_THRESHOLD = 0.85


@router.post("", response_model=ClassifyResponse)
def classify_text(body: ClassifyRequest):
    try:
        result = classifier.classify(body.text)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Model error: {exc}")

    result["is_bullying"] = result["prob_bully"] >= BULLY_THRESHOLD
    result["label"] = "BULLYING" if result["is_bullying"] else "SAFE"
    return ClassifyResponse(**result)
