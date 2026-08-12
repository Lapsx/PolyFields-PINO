from main import app, ExperimentRequest, Charge
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
    from main import run_experiment
    try:
        await run_experiment(req)
        print("Success")
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
