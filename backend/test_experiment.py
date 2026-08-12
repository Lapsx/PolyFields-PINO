import requests
import json

payload = {
    "charges": [],
    "b": 1.0,
    "kappa": 1.0,
    "u": 0.0,
    "polymer_charge": 0,
    "polymer_charge_intensity": 1.0,
    "sweep_type": "isoelectric"
}

resp = requests.post("http://localhost:8000/experiment", json=payload)
print(resp.status_code)
if resp.status_code != 200:
    print(resp.text)
else:
    print("Success, length of image data:", len(resp.json().get("image", "")))
