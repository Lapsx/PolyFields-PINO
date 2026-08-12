import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
X, Z = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
density = np.exp(-(X**2 + Z**2))

cax = ax.contourf(X, Z, density, levels=50, cmap='inferno')
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

plt.savefig("test.png", bbox_inches='tight')
print("Salvo test.png")
