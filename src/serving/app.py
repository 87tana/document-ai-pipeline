# src/serving/app.py
"""
FastAPI serving layer for Document AI Pipeline.
Uses ONNX Runtime instead of PyTorch for fast cold start on Lambda.

Endpoints:
    GET  /health     - health check
    POST /classify   - classify document type from image
    POST /pipeline   - classify + extract text
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path

import boto3
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

import json, logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def log_prediction(predicted_class, confidence):
    logger.info(json.dumps({'event': 'prediction', 'class': predicted_class, 'confidence': confidence}))


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = Path("/tmp/efficientnet_b0.onnx")
S3_BUCKET = "document-ai-pipeline-models"
S3_KEY = "models/exp3_efficientnet_b0/efficientnet_b0.onnx"
IMAGE_SIZE = 224

LABELS = [
    "letter", "form", "email", "handwritten", "advertisement",

    "scientific_report", "scientific_publication", "specification",
    "file_folder", "news_article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo"
]

# ImageNet normalization values
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Model loading ─────────────────────────────────────────────────────────────

_session = None


def load_model():
    global _session

    # Download ONNX model from S3 if not present
    if not MODEL_PATH.exists():
        print("Downloading ONNX model from S3...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3").download_file(S3_BUCKET, S3_KEY, str(MODEL_PATH))
        print("ONNX model downloaded.")

    _session = ort.InferenceSession(str(MODEL_PATH))
    print("ONNX model loaded successfully.")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    load_model()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Document AI Pipeline",
    description="Document classification and text extraction for clinic document triage.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response models ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


class ClassifyResponse(BaseModel):
    predicted_class: str
    class_id: int
    confidence: float
    all_scores: dict


class PipelineResponse(BaseModel):
    classification: ClassifyResponse
    extracted_text: str
    word_count: int


# ── Helper ────────────────────────────────────────────────────────────────────

def preprocess(image: Image.Image) -> np.ndarray:
    """Preprocess image for ONNX inference."""
    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr[np.newaxis, :]     # add batch dimension


def predict(image: Image.Image) -> ClassifyResponse:
    input_array = preprocess(image)
    outputs = _session.run(None, {"input": input_array})
    logits = outputs[0][0]

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    pred_id = int(np.argmax(probs))
    return ClassifyResponse(
        predicted_class=LABELS[pred_id],
        class_id=pred_id,
        confidence=round(float(probs[pred_id]), 4),
        all_scores={label: round(float(score), 4)
                    for label, score in zip(LABELS, probs)}
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model="efficientnet_b0_onnx",
        version="0.2.0"
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify(file: UploadFile = File(...)):
    """Classify document type from uploaded image."""
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    result = predict(image)
    log_prediction(result.predicted_class, result.confidence)
    return result


@app.post("/pipeline", response_model=PipelineResponse)
async def pipeline(file: UploadFile = File(...)):
    """Classify document and extract text."""
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    classification = predict(image)

    try:
        import pytesseract
        text = pytesseract.image_to_string(image, lang="eng")
    except Exception:
        text = ""

    words = text.split()
    return PipelineResponse(
        classification=classification,
        extracted_text=text.strip(),
        word_count=len(words)
    )

from mangum import Mangum
handler = Mangum(app)
