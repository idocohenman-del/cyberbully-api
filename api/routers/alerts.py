from datetime import datetime, timezone, timedelta
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import Alert, Family, FamilyMember, User
from ..schemas import AlertCreateRequest, AlertResponse
from ..dependencies import get_current_user
from ..fcm import send_push

router = APIRouter(prefix="/alerts", tags=["alerts"])

# In-memory FCM cooldown: (child_id, source_app) → last push datetime
_last_push: dict = {}


@router.post("", status_code=201)
def create_alert(
    body: AlertCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    import logging
    logging.getLogger(__name__).warning(
        "POST /alerts — user=%s role=%s", current_user.email, current_user.role
    )
    if current_user.role != "child":
        raise HTTPException(403, "only child apps can send alerts")

    member = db.query(FamilyMember).filter(FamilyMember.child_id == current_user.id).first()
    if not member:
        raise HTTPException(400, "child is not linked to any family — use POST /family/join first")

    alert = Alert(
        family_id    = member.family_id,
        child_id     = current_user.id,
        message_text = body.message_text,
        source_app   = body.source_app,
        prob_bully   = body.prob_bully,
        is_bullying  = body.is_bullying,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)

    # Only push FCM once per (child, app) per 5 minutes to avoid flooding guardian
    cooldown_key = (current_user.id, body.source_app or "unknown")
    now = datetime.now(timezone.utc)
    last = _last_push.get(cooldown_key)
    if not last or (now - last) >= timedelta(minutes=5):
        _last_push[cooldown_key] = now
        family_obj = db.query(Family).filter(Family.id == member.family_id).first()
        if family_obj:
            guardian = db.query(User).filter(User.id == family_obj.guardian_id).first()
            if guardian and guardian.fcm_token:
                child_name = current_user.first_name or "Your child"
                conf = int(body.prob_bully * 100)
                send_push(
                    guardian.fcm_token,
                    "Cyberbullying Alert",
                    f"{child_name} — detected in {body.source_app or 'unknown app'} ({conf}% confidence)",
                )

    return {"alert_id": alert.id, "status": "created"}


@router.get("", response_model=List[AlertResponse])
def get_alerts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "guardian":
        raise HTTPException(403, "guardians only")

    family = db.query(Family).filter(Family.guardian_id == current_user.id).first()
    if not family:
        return []

    alerts = (
        db.query(Alert)
        .filter(Alert.family_id == family.id, Alert.is_bullying == True)
        .order_by(Alert.timestamp.desc())
        .all()
    )

    result = []
    for alert in alerts:
        child = db.query(User).filter(User.id == alert.child_id).first()
        resp = AlertResponse.model_validate(alert)
        resp.child_first_name = (child.first_name if child and child.first_name else "Child")
        result.append(resp)
    return result


@router.put("/{alert_id}/read", status_code=200)
def mark_read(
    alert_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "guardian":
        raise HTTPException(403, "guardians only")

    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(404, "alert not found")

    alert.is_read = True
    db.commit()
    return {"status": "marked_read"}
