import random
import string
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import Family, FamilyMember, RegisteredChild, User
from ..schemas import FamilyCreateResponse, FamilyJoinRequest, RegisteredChildRequest, RegisteredChildResponse
from ..dependencies import get_current_user

router = APIRouter(prefix="/family", tags=["family"])


def _gen_code(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


@router.post("/create", response_model=FamilyCreateResponse, status_code=201)
def create_family(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "guardian":
        raise HTTPException(403, "only guardians can create a family")

    existing = db.query(Family).filter(Family.guardian_id == current_user.id).first()
    if existing:
        return FamilyCreateResponse(family_id=existing.id, invite_code=existing.invite_code)

    for _ in range(10):
        code = _gen_code()
        if not db.query(Family).filter(Family.invite_code == code).first():
            break

    family = Family(guardian_id=current_user.id, invite_code=code)
    db.add(family)
    db.commit()
    db.refresh(family)
    return FamilyCreateResponse(family_id=family.id, invite_code=family.invite_code)


@router.post("/join", status_code=200)
def join_family(
    body: FamilyJoinRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "child":
        raise HTTPException(403, "only child accounts can join a family")

    family = db.query(Family).filter(Family.invite_code == body.invite_code).first()
    if not family:
        raise HTTPException(404, "invite code not found")

    already = db.query(FamilyMember).filter(
        FamilyMember.family_id == family.id,
        FamilyMember.child_id == current_user.id,
    ).first()
    if already:
        return {"status": "already_joined", "family_id": family.id}

    if not body.skip_verification:
        registered = db.query(RegisteredChild).filter(
            RegisteredChild.family_id == family.id
        ).all()
        if registered:
            child_id_number = current_user.id_number or ""
            match = next((r for r in registered if r.id_number == child_id_number), None)
            if not match:
                raise HTTPException(
                    403,
                    "Your ID number does not match any child registered by the guardian. "
                    "Ask your guardian to add your profile first."
                )
            match.linked_child_id = current_user.id
            db.commit()

    db.add(FamilyMember(family_id=family.id, child_id=current_user.id))
    db.commit()
    return {"status": "joined", "family_id": family.id}


@router.get("/members")
def list_members(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "guardian":
        raise HTTPException(403, "guardians only")

    family = db.query(Family).filter(Family.guardian_id == current_user.id).first()
    if not family:
        raise HTTPException(404, "no family created yet")

    members = db.query(FamilyMember).filter(FamilyMember.family_id == family.id).all()
    children = []
    for m in members:
        child = db.query(User).filter(User.id == m.child_id).first()
        if child:
            children.append({
                "id":         child.id,
                "email":      child.email,
                "first_name": child.first_name or "Unknown",
                "last_name":  child.last_name  or "",
            })

    return {
        "family_id":   family.id,
        "invite_code": family.invite_code,
        "children":    children,
    }


@router.delete("/members/{child_id}", status_code=200)
def remove_child(
    child_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "guardian":
        raise HTTPException(403, "guardians only")

    family = db.query(Family).filter(Family.guardian_id == current_user.id).first()
    if not family:
        raise HTTPException(404, "no family found")

    member = db.query(FamilyMember).filter(
        FamilyMember.family_id == family.id,
        FamilyMember.child_id == child_id,
    ).first()
    if not member:
        raise HTTPException(404, "child not found in your family")

    db.delete(member)
    rc = db.query(RegisteredChild).filter(RegisteredChild.linked_child_id == child_id).first()
    if rc:
        rc.linked_child_id = None
    db.commit()
    return {"status": "removed"}


@router.get("/child-status")
def child_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "child":
        raise HTTPException(403, "children only")

    member = db.query(FamilyMember).filter(FamilyMember.child_id == current_user.id).first()
    if not member:
        raise HTTPException(404, "not linked to any family")

    return {"family_id": member.family_id, "linked": True}


@router.post("/registered-children", response_model=RegisteredChildResponse, status_code=201)
def add_registered_child(
    body: RegisteredChildRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "guardian":
        raise HTTPException(403, "guardians only")

    family = db.query(Family).filter(Family.guardian_id == current_user.id).first()
    if not family:
        raise HTTPException(400, "create a family first")

    rc = RegisteredChild(
        family_id=family.id,
        first_name=body.first_name.strip(),
        last_name=body.last_name.strip(),
        id_number=body.id_number.strip(),
    )
    db.add(rc)
    db.commit()
    db.refresh(rc)
    return rc


@router.get("/registered-children")
def list_registered_children(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "guardian":
        raise HTTPException(403, "guardians only")

    family = db.query(Family).filter(Family.guardian_id == current_user.id).first()
    if not family:
        return []

    children = db.query(RegisteredChild).filter(RegisteredChild.family_id == family.id).all()
    return [
        {
            "id":         rc.id,
            "first_name": rc.first_name,
            "last_name":  rc.last_name,
            "id_number":  rc.id_number,
            "linked":     rc.linked_child_id is not None,
        }
        for rc in children
    ]


@router.delete("/registered-children/{rc_id}", status_code=200)
def remove_registered_child(
    rc_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "guardian":
        raise HTTPException(403, "guardians only")

    family = db.query(Family).filter(Family.guardian_id == current_user.id).first()
    if not family:
        raise HTTPException(404, "no family found")

    rc = db.query(RegisteredChild).filter(
        RegisteredChild.id == rc_id,
        RegisteredChild.family_id == family.id,
    ).first()
    if not rc:
        raise HTTPException(404, "registered child not found")

    db.delete(rc)
    db.commit()
    return {"status": "removed"}
