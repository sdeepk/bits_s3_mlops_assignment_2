import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.inference import app


def get_test_client():
    return TestClient(app)

def test_health_endpoint():
    with get_test_client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

def test_predict_endpoint_accepts_image():

    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    with get_test_client() as client:
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", buf, "image/jpeg")}
        )

    assert response.status_code == 200
    assert "label" in response.json()
    assert "probability" in response.json()

