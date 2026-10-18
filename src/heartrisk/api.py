"""FastAPI service exposing the heart-disease risk model.

Endpoints: GET /  ·  GET /health  ·  POST /predict  ·  GET /metrics (Prometheus)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from heartrisk import __version__
from heartrisk.data import FEATURES, ROOT

DEFAULT_MODEL_PATH = ROOT / "models" / "model.joblib"


# ---------- logging: one JSON object per line (easy to ship to Loki/ELK/CloudWatch) ----------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


logger = logging.getLogger("heartrisk.api")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(JsonFormatter())
    logger.addHandler(_handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    logger.propagate = False


# ---------- Prometheus metrics ----------
REQUESTS = Counter("heartrisk_http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram(
    "heartrisk_http_request_duration_seconds",
    "Request latency",
    ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
PREDICTIONS = Counter("heartrisk_predictions_total", "Predictions by class", ["label"])
CONFIDENCE = Histogram(
    "heartrisk_prediction_confidence",
    "Confidence of returned predictions",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
MODEL_INFO = Gauge("heartrisk_model_loaded", "1 if a model is loaded", ["model_name", "version"])


# ---------- request / response schemas ----------
class PatientRecord(BaseModel):
    """One patient's measurements, using the UCI Cleveland coding."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 1,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0,
                "thal": 6,
            }  # fmt: skip
        }
    )

    age: float = Field(..., ge=18, le=100, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=70, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=650, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=60, le=230, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=7, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: int | None = Field(None, ge=0, le=3, description="Major vessels coloured (0-3)")
    thal: int | None = Field(None, description="3 = normal, 6 = fixed, 7 = reversible defect")


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int
    label: str
    confidence: float
    probability_disease: float
    model_name: str
    model_version: str


# ---------- app ----------
def create_app(model_path: str | Path | None = None) -> FastAPI:
    path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = joblib.load(path)
        meta_file = path.with_name("metadata.json")
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        app.state.model_name = meta.get("model_name", type(app.state.model[-1]).__name__)
        app.state.model_version = meta.get("model_version", __version__)
        MODEL_INFO.labels(app.state.model_name, app.state.model_version).set(1)
        logger.info(
            "model loaded", extra={"extra_fields": {"path": str(path), **meta.get("metrics", {})}}
        )
        yield

    app = FastAPI(
        title="Heart Disease Risk API",
        version=__version__,
        description="Predicts presence of heart disease from patient health data.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            route = request.scope.get("route")
            fallback = "unmatched" if status == 404 else request.url.path
            path_label = route.path if route else fallback
            REQUESTS.labels(request.method, path_label, str(status)).inc()
            LATENCY.labels(path_label).observe(elapsed)
            if request.url.path != "/metrics":  # don't flood logs with scrapes
                logger.info(
                    "request",
                    extra={
                        "extra_fields": {
                            "request_id": request_id,
                            "method": request.method,
                            "path": request.url.path,
                            "status": status,
                            "latency_ms": round(elapsed * 1000, 2),
                            "client": request.client.host if request.client else None,
                        }
                    },
                )

    @app.get("/")
    def root():
        return {"service": "heart-disease-risk-api", "version": __version__, "docs": "/docs"}

    @app.get("/health")
    def health(request: Request):
        loaded = getattr(request.app.state, "model", None) is not None
        if not loaded:
            raise HTTPException(503, "model not loaded")
        return {"status": "ok", "model_name": request.app.state.model_name}

    @app.post("/predict", response_model=Prediction)
    def predict(record: PatientRecord, request: Request) -> Prediction:
        state = request.app.state
        X = pd.DataFrame([record.model_dump()], columns=FEATURES).astype(float)
        proba = float(state.model.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)
        confidence = proba if pred else 1 - proba
        label = "heart_disease" if pred else "no_heart_disease"
        PREDICTIONS.labels(label).inc()
        CONFIDENCE.observe(confidence)
        logger.info(
            "prediction",
            extra={"extra_fields": {"label": label, "confidence": round(confidence, 4)}},
        )
        return Prediction(
            prediction=pred,
            label=label,
            confidence=round(confidence, 4),
            probability_disease=round(proba, 4),
            model_name=state.model_name,
            model_version=state.model_version,
        )

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
