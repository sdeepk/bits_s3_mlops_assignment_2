import torch
import uvicorn
from fastapi import FastAPI
import numpy as np

from schemas import PredictRequest, PredictResponse
from load_model import load_champion_model

app = FastAPI(title="MNIST CNN Inference Service")

model = load_champion_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Convert input to tensor
    x = np.array(request.pixels, dtype=np.float32)
    x = x.reshape(1, 1, 28, 28)
    x = torch.tensor(x)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        pred = int(torch.argmax(logits, dim=1).item())

    return PredictResponse(
        predicted_label=pred,
        probabilities=probs
    )

if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8000)
