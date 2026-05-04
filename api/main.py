import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import text, inspect
from .database import engine, Base
from .routers import auth, classify, family, alerts
from . import classifier as _classifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def _migrate_db():
    """Safely add new columns to existing tables without dropping data."""
    with engine.connect() as conn:
        inspector = inspect(engine)
        existing = {col["name"] for col in inspector.get_columns("users")}
        for col, type_ in [("first_name", "VARCHAR"), ("last_name", "VARCHAR"), ("id_number", "VARCHAR")]:
            if col not in existing:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {col} {type_}"))
                conn.commit()
                log.info("Migration: added column users.%s", col)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    _migrate_db()
    log.info("Database tables ready.")

    try:
        _classifier._get_predictor()
        log.info("DistilBERT model loaded and ready.")
    except FileNotFoundError as exc:
        log.warning("Model not loaded at startup: %s", exc)

    yield


app = FastAPI(
    title="CyberBullying Guard API",
    description="Backend for the child/guardian cyberbullying detection app",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(auth.router)
app.include_router(classify.router)
app.include_router(family.router)
app.include_router(alerts.router)


@app.get("/health", tags=["meta"])
def health():
    return {
        "status":      "ok",
        "model_ready": _classifier.is_model_ready(),
    }
