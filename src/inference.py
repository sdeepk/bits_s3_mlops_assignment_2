import mlflow
import mlflow.pytorch
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
from contextlib import asynccontextmanager

from src.schemas import PredictResponse
from src.inference_utils import prepare_image


MODEL_NAME = "cats-dogs-resnet18"
STAGE = "Production"
CLASS_NAMES = ["Cat", "Dog"]


@asynccontextmanager
async def lifespan(app):
    model = mlflow.pytorch.load_model("models/champion")
    model.eval()
    app.state.model = model
    print("Champion model loaded")
    yield

app = FastAPI(
    title="Cats vs Dogs Inference Service",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {"status": "ok"}

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