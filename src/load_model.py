import mlflow
import mlflow.pytorch
import torch
from model import SimpleCNN

MODEL_NAME = "cnn-mnist"
STAGE = "Production"

def load_champion_model():
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    return model
