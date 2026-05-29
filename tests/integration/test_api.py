import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from src.serving.app import app, load_model

@pytest.fixture(scope="session", autouse=True)
def setup_model():
    load_model()

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

def make_dummy_image() -> bytes:
    img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_health_returns_model_name(client):
    response = client.get("/health")
    assert response.json()["model"] == "efficientnet_b0"

def test_classify_returns_200(client):
    response = client.post("/classify", files={"file": ("test.png", make_dummy_image(), "image/png")})
    assert response.status_code == 200

def test_classify_response_structure(client):
    data = client.post("/classify", files={"file": ("test.png", make_dummy_image(), "image/png")}).json()
    assert "predicted_class" in data
    assert "class_id" in data
    assert "confidence" in data
    assert "all_scores" in data

def test_classify_confidence_between_0_and_1(client):
    confidence = client.post("/classify", files={"file": ("test.png", make_dummy_image(), "image/png")}).json()["confidence"]
    assert 0.0 <= confidence <= 1.0

def test_classify_predicted_class_is_valid(client):
    valid_classes = ["letter","form","email","handwritten","advertisement","scientific_report","scientific_publication","specification","file_folder","news_article","budget","invoice","presentation","questionnaire","resume","memo"]
    result = client.post("/classify", files={"file": ("test.png", make_dummy_image(), "image/png")}).json()
    assert result["predicted_class"] in valid_classes

def test_classify_all_scores_sum_to_one(client):
    scores = client.post("/classify", files={"file": ("test.png", make_dummy_image(), "image/png")}).json()["all_scores"]
    assert abs(sum(scores.values()) - 1.0) < 0.01

def test_classify_rejects_invalid_file(client):
    response = client.post("/classify", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert response.status_code == 400

def test_pipeline_returns_200(client):
    response = client.post("/pipeline", files={"file": ("test.png", make_dummy_image(), "image/png")})
    assert response.status_code == 200

def test_pipeline_response_structure(client):
    data = client.post("/pipeline", files={"file": ("test.png", make_dummy_image(), "image/png")}).json()
    assert "classification" in data
    assert "extracted_text" in data
    assert "word_count" in data
