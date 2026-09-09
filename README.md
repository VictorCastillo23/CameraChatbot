# CameraChatbot

A computer-vision pipeline that ingests camera keyframes, runs person detection/re-identification/pose (plus a security subsystem: continuous tracking, zone intrusion/loitering detection, and person authorization) over them, and persists the result to Postgres.

See [`docs/README.md`](docs/README.md) for the full reference — start there for setup, architecture, and module-by-module docs. [`docs/05-running-the-pipeline.md`](docs/05-running-the-pipeline.md) is the fastest path to a working local run.

## Quick start

Create a virtual environment first — don't install into your system Python:

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate.bat

# macOS/Linux
source venv/bin/activate
```

Then, with the venv active:

```bash
pip install -r requirements.txt
```

Model weights are **not** in the repo (`models/` is gitignored). Download them from:

https://drive.google.com/drive/folders/1GL0_Td_-NVwmMqjASZNzko-FddC4wt6i?usp=drive_link

If you find better models, feel free to swap them in.

The runtime loads `.onnx` files, not the raw `.pt` weights — export them first with `python tools/export_to_onnx.py --all` (see [`docs/07-testing-and-dev-tools.md`](docs/07-testing-and-dev-tools.md)). Set the required `.env` vars (Postgres + optionally Supabase — see [`docs/05-running-the-pipeline.md`](docs/05-running-the-pipeline.md) for the full list), then:

```bash
python run_local.py
```

## Entry points

- `run_local.py` — offline smoke test: reads `keyFrames/`, runs the pipeline once, persists to Postgres.
- `run_webhook.py` — Flask app exposing `POST /webhook`, downloads frames from Supabase Storage and runs the same pipeline.
- `run_live_capture.py` — captures frames live from a camera and uploads keyframes to Supabase Storage.
