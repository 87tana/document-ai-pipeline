# src/serving/app.py
"""
FastAPI serving layer for Document AI Pipeline.

Endpoints:
    GET  /health     - health check
    POST /classify   - classify document type from image
    POST /pipeline   - classify + extract text
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path

import boto3
import timm
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from src.monitoring.monitor import log_prediction


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = Path("/tmp/best_model.pt")
S3_BUCKET = "document-ai-pipeline-models"
S3_KEY = "models/exp3_efficientnet_b0/best_model.pt"
ARCHITECTURE = "efficientnet_b0"
NUM_CLASSES = 16
IMAGE_SIZE = 224

LABELS = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific_report", "scientific_publication", "specification",
    "file_folder", "news_article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo"
]


# ── Model loading ─────────────────────────────────────────────────────────────

_model = None
_device = None
_transform = None


def load_model():
    global _model, _device, _transform

    # Download model from S3 if not present locally
    if not MODEL_PATH.exists():
        print("Downloading model from S3...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3").download_file(S3_BUCKET, S3_KEY, str(MODEL_PATH))
        print("Model downloaded from S3.")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = timm.create_model(
        ARCHITECTURE,
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    _model.to(_device)
    _model.eval()

    _transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    print(f"Model loaded: {ARCHITECTURE} on {_device}")


# ── Lifespan — runs load_model() at startup ───────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    load_model()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Document AI Pipeline",
    description="Document classification and text extraction for clinic document triage.",
    version="0.1.0",
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

def predict(image: Image.Image) -> ClassifyResponse:
    img = image.convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    pred_id = int(torch.argmax(torch.tensor(probs)))
    return ClassifyResponse(
        predicted_class=LABELS[pred_id],
        class_id=pred_id,
        confidence=round(probs[pred_id], 4),
        all_scores={label: round(score, 4)
                    for label, score in zip(LABELS, probs)}
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model=ARCHITECTURE,
        version="0.1.0"
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
