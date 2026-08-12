import requests
import json

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
try:
    response = requests.post(url, json=payload)
    print(response.status_code)
    # Don't print full text as it's base64 images
    data = response.json()
    print("Keys in response:", data.keys())
    print("Metrics:", data.get("metrics"))
except Exception as e:
    print(e)
