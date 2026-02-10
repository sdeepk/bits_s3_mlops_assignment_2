import torch
from torchvision import transforms
from PIL import Image

# Same normalization as training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def prepare_image(image: Image.Image):
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    return tensor
