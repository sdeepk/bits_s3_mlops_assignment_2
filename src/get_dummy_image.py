import json
import random

payload = {
    "pixels": [random.random() for _ in range(784)]
}

print(json.dumps(payload))