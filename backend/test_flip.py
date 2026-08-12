import torch
import numpy as np
import sys

sys.path.append("/home/lucas/Documents/PINN/FNO_WebApp/backend/model_core")
from fno_parametric_architecture import ParametricFNO2d

device = torch.device('cpu')
model = ParametricFNO2d(modes1=16, modes2=16, width=64).to(device)
model.load_state_dict(torch.load("/home/lucas/Documents/PINN/FNO_WebApp/backend/model_core/fno_parametric_best_model.pth", map_location=device, weights_only=True))
model.eval()

N = 100
L = 8.0
x = np.linspace(-L/2, L/2, N)
z = np.linspace(-L/2, L/2, N)
X, Z = np.meshgrid(x, z, indexing='ij')

def query_fno(cx, cz):
    V = np.zeros((N, N))
    dist = np.sqrt((X - cx)**2 + (Z - cz)**2)
    V += 5.0 * np.exp(-1.0 * dist) / (dist + 0.1)
    V[np.sqrt(X**2 + Z**2) < 1.0] = 10.0
    
    # We will ALWAYS evaluate the FNO with Z-coordinates flipped if cz > 0
    flip = cz > 0
    if flip:
        # Flip V along the Z axis (axis=1)
        V = np.flip(V, axis=1).copy()
        
    inputs = torch.zeros(1, N, N, 6, dtype=torch.float32)
    inputs[0, :, :, 0] = torch.tensor(V)
    inputs[0, :, :, 1] = torch.tensor(X)
    inputs[0, :, :, 2] = torch.tensor(Z) # Always pass the ORIGINAL Z grid!
    inputs[0, :, :, 3] = 1.0
    inputs[0, :, :, 4] = 1.0
    inputs[0, :, :, 5] = 0.0
    
    with torch.no_grad():
        out = model(inputs)
    density = out[0, :, :, 0].numpy()
    
    if flip:
        # Flip the output back!
        density = np.flip(density, axis=1).copy()
        
    density[np.sqrt(X**2 + Z**2) < 1.0] = 0.0
    max_idx = np.unravel_index(np.argmax(density), density.shape)
    print(f"Charge at Z={cz} -> FNO Density Max at Z={Z[max_idx]:.2f}")

query_fno(0.0, -3.0)
query_fno(0.0, 3.0)
