import torch
import numpy as np
from backend.main import compute_density, Charge

charges = [Charge(x=50, z=50, q=5.0, r=5.0)]
b = 1.0
kappa = 1.0
u = 0.0

density, d1, d2 = compute_density(charges, b, kappa, u)
print("Success!")
