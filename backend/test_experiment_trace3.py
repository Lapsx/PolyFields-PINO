from main import ExperimentRequest, Charge
import asyncio

async def main():
    req = ExperimentRequest(
        charges=[],
        b=1.0,
        kappa=1.0,
        u=0.0,
        polymer_charge=0,
        polymer_charge_intensity=1.0,
        sweep_type="isoelectric"
    )
    from main import compute_density, N
    import numpy as np
    
    values = np.linspace(-10.0, 10.0, 40)
    metrics = []
    
    for val in values:
        mod_charges = []
        mod_charges.append(Charge(x=int(N/2), z=int(N/2), q=float(val), r=5.0))
        density, _, _ = compute_density(mod_charges, req.b, req.kappa, req.u, req.polymer_charge, req.polymer_charge_intensity)
        
        valid_mask = ~np.isnan(density)
        valid_density = density[valid_mask]
        valid_density = np.clip(valid_density, 0, None)
        
        mass = float(np.sum(valid_density))
        if mass > 1e-6:
            com_x = float(np.sum(np.meshgrid(np.linspace(-4, 4, N), np.linspace(-4, 4, N), indexing='ij')[0][valid_mask] * valid_density) / mass)
            com_z = float(np.sum(np.meshgrid(np.linspace(-4, 4, N), np.linspace(-4, 4, N), indexing='ij')[1][valid_mask] * valid_density) / mass)
            dist_com = float(np.sqrt(com_x**2 + com_z**2))
        else:
            dist_com = 0.0
        metrics.append(dist_com)
    print("Metrics:", metrics)

asyncio.run(main())
