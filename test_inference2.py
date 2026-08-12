import requests
import numpy as np
import base64
from io import BytesIO
from PIL import Image

url = "http://127.0.0.1:8000/predict"
payload = {
  "charges": [
    {
      "x": 50,
      "z": 50,
      "q": 5.0,
      "r": 5.0
    }
  ],
  "b": 1.0,
  "kappa": 1.0,
  "u": 0.0
}
response = requests.post(url, json=payload)
data = response.json()
img_data = base64.b64decode(data['image'])
img = Image.open(BytesIO(img_data))
img.save("test_out.png")
