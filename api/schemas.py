from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:      str
    password:   str
    role:       str
    first_name: str
    last_name:  str
    id_number:  str

class LoginRequest(BaseModel):
    email:    str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    role:         str
    user_id:      str

class FCMTokenUpdate(BaseModel):
    fcm_token: str


# ── Classify ──────────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    label:       str
    is_bullying: bool
    prob_bully:  float
    confidence:  float


# ── Family ────────────────────────────────────────────────────────────────────

class FamilyCreateResponse(BaseModel):
    family_id:   str
    invite_code: str

class FamilyJoinRequest(BaseModel):
    invite_code:       str
    skip_verification: bool = False

class RegisteredChildRequest(BaseModel):
    first_name: str
    last_name:  str
    id_number:  str

class RegisteredChildResponse(BaseModel):
    id:              str
    first_name:      str
    last_name:       str
    id_number:       str
    linked_child_id: Optional[str] = None
    model_config = {"from_attributes": True}


# ── Alerts ────────────────────────────────────────────────────────────────────

class AlertCreateRequest(BaseModel):
    message_text: str
    source_app:   Optional[str] = None
    prob_bully:   float
    is_bullying:  bool

class AlertResponse(BaseModel):
    id:               str
    child_id:         str
    child_first_name: Optional[str] = None
    message_text:     str
    source_app:       Optional[str]
    prob_bully:       float
    is_bullying:      bool
    timestamp:        datetime
    is_read:          bool
    model_config = {"from_attributes": True}
