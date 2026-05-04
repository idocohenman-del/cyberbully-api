import base64
import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_app = None
_init_failed = False

_CRED_PATH = Path(__file__).parent.parent / "firebase_service_account.json"


def _resolve_credentials() -> str | None:
    """Return a file path to the service account JSON, or None if unavailable."""
    env_val = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if env_val:
        try:
            decoded = base64.b64decode(env_val).decode("utf-8")
            json.loads(decoded)  # validate it's real JSON
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            tmp.write(decoded)
            tmp.flush()
            return tmp.name
        except Exception as exc:
            logger.warning("FIREBASE_SERVICE_ACCOUNT env var is invalid: %s", exc)

    if _CRED_PATH.exists():
        return str(_CRED_PATH)

    return None


def _get_app():
    global _app, _init_failed
    if _app is not None:
        return _app
    if _init_failed:
        return None

    cred_path = _resolve_credentials()
    if not cred_path:
        logger.warning("FCM disabled: no credentials found (set FIREBASE_SERVICE_ACCOUNT env var)")
        _init_failed = True
        return None

    try:
        import firebase_admin
        from firebase_admin import credentials
        try:
            _app = firebase_admin.get_app()
        except ValueError:
            cred = credentials.Certificate(cred_path)
            _app = firebase_admin.initialize_app(cred)
        return _app
    except Exception as exc:
        logger.warning("FCM init failed: %s", exc)
        _init_failed = True
        return None


def send_push(token: str, title: str, body: str) -> None:
    if not _get_app():
        return
    try:
        from firebase_admin import messaging
        msg = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            android=messaging.AndroidConfig(priority="high"),
            token=token,
        )
        messaging.send(msg)
    except Exception as exc:
        logger.warning("FCM send failed: %s", exc)
