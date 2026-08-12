import torch
import numpy as np
import os, sys

sys.path.append("/home/lucas/Documents/PINN/FNO_WebApp/backend/model_core")
from fno_parametric_architecture import ParametricFNO2d

device = torch.device('cpu')
model = ParametricFNO2d(modes1=16, modes2=16, width=64).to(device)
weights_path = "/home/lucas/Documents/PINN/FNO_WebApp/backend/model_core/fno_parametric_best_model.pth"
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.eval()

N = 100
L = 8.0
x = np.linspace(-L/2, L/2, N)
z = np.linspace(-L/2, L/2, N)
X, Z = np.meshgrid(x, z, indexing='ij')

def test_charge(cx_val, cz_val):
    V = np.zeros((N, N))
    epsilon = 0.1
    dist = np.sqrt((X - cx_val)**2 + (Z - cz_val)**2)
    V += 5.0 * np.exp(-1.0 * dist) / (dist + epsilon)
    
    # Do NOT put nanoparticle infinite potential yet, let's just see where it goes
    V_clean = np.copy(V)
    V_clean[np.sqrt(X**2 + Z**2) < 1.0] = 10.0
    
    inputs = torch.zeros(1, N, N, 6, dtype=torch.float32)
    inputs[0, :, :, 0] = torch.tensor(V_clean)
    inputs[0, :, :, 1] = torch.tensor(X)
    inputs[0, :, :, 2] = torch.tensor(Z)
    inputs[0, :, :, 3] = 1.0
    inputs[0, :, :, 4] = 1.0
    inputs[0, :, :, 5] = 0.0
    
    with torch.no_grad():
        out = model(inputs)
    density = out[0, :, :, 0].numpy()
    
    # Remove inside from max calculation
    density[np.sqrt(X**2 + Z**2) < 1.0] = 0.0
    
    max_idx = np.unravel_index(np.argmax(density), density.shape)
    print(f"Charge at ({cx_val}, {cz_val}) -> Max Density at ({X[max_idx]:.2f}, {Z[max_idx]:.2f})")

test_charge(0.0, -3.0)
test_charge(0.0, 3.0)
