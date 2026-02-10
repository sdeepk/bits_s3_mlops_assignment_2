import mlflow
import mlflow.pytorch

MODEL_NAME = "cnn-mnist"
STAGE = "Production"

def load_champion_model():
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    return model
