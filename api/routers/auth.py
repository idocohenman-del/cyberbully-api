from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import User
from ..schemas import RegisterRequest, LoginRequest, TokenResponse, FCMTokenUpdate
from ..auth_utils import hash_password, verify_password, create_token
from ..dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if body.role not in ("guardian", "child"):
        raise HTTPException(400, "role must be 'guardian' or 'child'")
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(409, "email already registered")

    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        role=body.role,
        first_name=body.first_name.strip(),
        last_name=body.last_name.strip(),
        id_number=body.id_number.strip(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return TokenResponse(
        access_token=create_token(user.id, user.role),
        role=user.role,
        user_id=user.id,
    )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "invalid credentials")

    return TokenResponse(
        access_token=create_token(user.id, user.role),
        role=user.role,
        user_id=user.id,
    )


@router.put("/fcm-token", status_code=200)
def update_fcm_token(
    body: FCMTokenUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    db.query(User).filter(User.id == current_user.id).update({"fcm_token": body.fcm_token})
    db.commit()
    return {"status": "updated"}
