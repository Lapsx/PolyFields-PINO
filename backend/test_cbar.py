import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
X, Z = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
density = np.exp(-(X**2 + Z**2))

cax = ax.contourf(X, Z, density, levels=50, cmap='inferno')
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(colors='white')
ax.set_aspect('equal')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

plt.savefig("test2.png", bbox_inches='tight', facecolor='black')
