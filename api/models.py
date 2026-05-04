import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Boolean, DateTime, ForeignKey
from .database import Base


def _uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id            = Column(String, primary_key=True, default=_uuid)
    email         = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    role          = Column(String, nullable=False)
    first_name    = Column(String, nullable=True)
    last_name     = Column(String, nullable=True)
    id_number     = Column(String, nullable=True)
    fcm_token     = Column(String, nullable=True)
    created_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Family(Base):
    __tablename__ = "families"

    id          = Column(String, primary_key=True, default=_uuid)
    guardian_id = Column(String, ForeignKey("users.id"), nullable=False)
    invite_code = Column(String, unique=True, nullable=False, index=True)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RegisteredChild(Base):
    """Guardian pre-registers a child's identity before they join."""
    __tablename__ = "registered_children"

    id              = Column(String, primary_key=True, default=_uuid)
    family_id       = Column(String, ForeignKey("families.id"), nullable=False)
    first_name      = Column(String, nullable=False)
    last_name       = Column(String, nullable=False)
    id_number       = Column(String, nullable=False)
    linked_child_id = Column(String, ForeignKey("users.id"), nullable=True)


class FamilyMember(Base):
    __tablename__ = "family_members"

    id        = Column(String, primary_key=True, default=_uuid)
    family_id = Column(String, ForeignKey("families.id"), nullable=False)
    child_id  = Column(String, ForeignKey("users.id"), nullable=False)


class Alert(Base):
    __tablename__ = "alerts"

    id           = Column(String, primary_key=True, default=_uuid)
    family_id    = Column(String, ForeignKey("families.id"), nullable=False)
    child_id     = Column(String, ForeignKey("users.id"), nullable=False)
    message_text = Column(String, nullable=False)
    source_app   = Column(String, nullable=True)
    prob_bully   = Column(Float, nullable=False)
    is_bullying  = Column(Boolean, nullable=False)
    timestamp    = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_read      = Column(Boolean, default=False)
