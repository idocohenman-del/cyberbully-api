# CyberBully Guard — API Server

Real-time cyberbullying detection backend for the CyberBully Guard mobile system.  
Classifies notification text using a fine-tuned DistilBERT model and alerts guardians via Firebase push notifications.

**Live API:** https://cyberbully-api-pnmm.onrender.com  
**Interactive docs:** https://cyberbully-api-pnmm.onrender.com/docs

---

## Overview

CyberBully Guard is a B.Sc final project in Information Systems. The system monitors incoming notifications on a child's Android device, classifies them in real time using NLP, and notifies the parent instantly when bullying content is detected.

This repository contains the **FastAPI backend** only. The Flutter mobile apps (child + guardian) are in a separate repository.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI |
| ML Model | DistilBERT (fine-tuned, HuggingFace Transformers) |
| Database | SQLite + SQLAlchemy ORM |
| Authentication | JWT (python-jose) + bcrypt (passlib) |
| Push Notifications | Firebase Cloud Messaging (FCM) |
| Containerization | Docker |
| Cloud Deployment | Render (with persistent disk) |
| Model Storage | Google Drive (downloaded via gdown on startup) |

---

## Model Performance

Trained on **92,883 labeled messages** (bullying / not bullying):

| Metric | Score |
|--------|-------|
| Accuracy | 91.5% |
| F1-Score | 0.844 |
| Recall (bullying class) | 87.1% |
| Precision | 90.2% |
| AUC-ROC | 0.962 |

Classification threshold: `prob_bully >= 0.85` → labeled as BULLYING.

---

## API Endpoints

### Auth
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/auth/register` | Public | Register as guardian or child |
| POST | `/auth/login` | Public | Login and receive JWT token |
| PUT | `/auth/fcm-token` | JWT | Update Firebase push token |

### Classification
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/classify` | JWT | Classify text using DistilBERT |

### Alerts
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/alerts` | JWT (child) | Submit a detected bullying alert |
| GET | `/alerts` | JWT (guardian) | Get all alerts for the family |
| PUT | `/alerts/{id}/read` | JWT (guardian) | Mark alert as read |

### Family
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/family/create` | JWT (guardian) | Create family and get invite code |
| POST | `/family/join` | JWT (child) | Join family using invite code |
| GET | `/family/members` | JWT (guardian) | List linked children |
| DELETE | `/family/members/{child_id}` | JWT (guardian) | Remove a child from family |
| POST | `/family/registered-children` | JWT (guardian) | Pre-register a child by ID number |
| GET | `/family/registered-children` | JWT (guardian) | List pre-registered children |
| DELETE | `/family/registered-children/{id}` | JWT (guardian) | Remove pre-registered child |
| GET | `/family/child-status` | JWT (child) | Check if child is linked to a family |

### Health
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | Public | Server and model status check |

---

## Example Requests

**Register as guardian:**
```json
POST /auth/register
{
  "email": "parent@example.com",
  "password": "SecurePass123",
  "role": "guardian",
  "first_name": "Ido",
  "last_name": "Cohen",
  "id_number": "123456789"
}
```

**Classify a message:**
```json
POST /classify
{ "text": "nobody likes you, you should disappear forever" }
```
```json
Response:
{
  "label": "BULLYING",
  "is_bullying": true,
  "prob_bully": 0.961,
  "confidence": 0.961
}
```

---

## Project Structure

```
api/
├── main.py               # App entry point, router registration
├── models.py             # SQLAlchemy ORM models
├── schemas.py            # Pydantic request/response schemas
├── database.py           # DB session and engine
├── auth_utils.py         # JWT creation, password hashing
├── dependencies.py       # get_current_user dependency
├── classifier.py         # DistilBERT inference logic
├── fcm.py                # Firebase push notification sender
├── requirements.txt
└── routers/
    ├── auth.py
    ├── alerts.py
    ├── classify.py
    └── family.py
src/
└── (model download scripts)
Dockerfile
start.sh
```

---

## Running Locally

### Prerequisites
- Python 3.10+
- Docker (optional)

### Without Docker

```bash
pip install -r api/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers numpy gdown

uvicorn api.main:app --reload
```

### With Docker

```bash
docker build -t cyberbully-api .
docker run -p 8000:8000 cyberbully-api
```

The model is downloaded automatically from Google Drive on first startup via `gdown`. Make sure the `MODEL_GDRIVE_ID` environment variable is set.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SECRET_KEY` | JWT signing secret |
| `MODEL_GDRIVE_ID` | Google Drive file ID for the DistilBERT model |
| `FIREBASE_CREDENTIALS` | Firebase service account JSON (as string or file path) |

---

## Database Schema

SQLite database with 5 tables: `users`, `families`, `registered_children`, `family_members`, `alerts`.  
All primary keys are UUID strings. See `api/models.py` for full schema.

---

## Authors

- **Ido Cohen** — Backend, ML integration, deployment
- **Shiraz Mor Yosef** — Frontend (Flutter), UI/UX

B.Sc in Information Systems — The Academic Center of Law and Business  
Supervisor: Dr. Saeid Asli
