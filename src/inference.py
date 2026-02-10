# ---- Imports ----
import time
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from contextlib import asynccontextmanager
import torch
import mlflow.pytorch
from src.inference_utils import prepare_image
from src.schemas import PredictResponse

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cats-dogs-api")

# ---- Metrics Globals ----
request_count = 0
total_latency = 0.0

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
    global request_count, total_latency

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    request_count += 1
    total_latency += duration

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
    avg_latency = (
        total_latency / request_count if request_count > 0 else 0
    )
    return {
        "request_count": request_count,
        "average_latency": avg_latency,
    }

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