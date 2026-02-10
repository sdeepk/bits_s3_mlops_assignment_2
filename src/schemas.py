from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    pixels: List[float]  # flattened 28x28 image = 784 values

class PredictResponse(BaseModel):
    predicted_label: int
    probabilities: List[float]
