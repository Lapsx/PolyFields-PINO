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
    from main import run_experiment
    try:
        res = await run_experiment(req)
        with open("test_iso.png", "wb") as f:
            import base64
            f.write(base64.b64decode(res["image"]))
        print("Success, saved test_iso.png")
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
