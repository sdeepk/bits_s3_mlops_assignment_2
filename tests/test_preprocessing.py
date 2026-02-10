from PIL import Image
import numpy as np
from src.inference_utils import prepare_image

def test_prepare_image_shape():
    # Create dummy RGB image
    img = Image.fromarray(
        np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    )

    tensor = prepare_image(img)

    # Expect shape: (1, 3, 224, 224)
    assert tensor.shape == (1, 3, 224, 224)
