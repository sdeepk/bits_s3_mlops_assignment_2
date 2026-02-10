import mlflow
import shutil
import os

MODEL_NAME = "cats-dogs-resnet18"
MODEL_URI = f"models:/{MODEL_NAME}/Production"

dst = "models/champion"
os.makedirs("models", exist_ok=True)

model = mlflow.pytorch.load_model(MODEL_URI)
mlflow.pytorch.save_model(model, dst)

print("Champion model exported to models/champion")