# ---- Imports ----
import io
import logging
import time
from contextlib import asynccontextmanager

import mlflow.pytorch
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.inference_utils import prepare_image
from src.schemas import PredictResponse

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cats-dogs-api")

CLASS_NAMES = ["Cat", "Dog"]

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests"
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds"
)

# ---- Lifespan ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    model = mlflow.pytorch.load_model("models/champion")
    model.eval()
    app.state.model = model
    logger.info("Champion model loaded")
    yield

# ---- Create App ----
app = FastAPI(
    title="Cats vs Dogs Inference Service",
    lifespan=lifespan
)

# ---- Middleware ----
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(duration)

    logger.info(
        f"{request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Latency: {duration:.4f}s"
    )

    return response


# ---- Routes ----
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    x = prepare_image(image)

    with torch.no_grad():
        logits = app.state.model(x)
        probs = torch.softmax(logits, dim=1).squeeze()
        pred_idx = int(torch.argmax(probs))
        confidence = float(probs[pred_idx])

    return PredictResponse(
        label=CLASS_NAMES[pred_idx],
        probability=confidence
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "inference:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )