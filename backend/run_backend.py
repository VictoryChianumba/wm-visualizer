"""
Launcher for the WM Visualizer FastAPI backend.

Run with the iris venv Python (which has all IRIS deps):
    /path/to/iris/.venv310/bin/python run_backend.py

This sets the working directory and IRIS env vars before handing off to uvicorn.
"""
import os
import sys
from pathlib import Path

# Ensure cwd = backend/ so uvicorn can import main.py
os.chdir(Path(__file__).parent)

# Set env vars if not already provided
_iris_root = Path(__file__).parent.parent.parent / "iris"
os.environ.setdefault("IRIS_ROOT", str(_iris_root))
os.environ.setdefault("IRIS_SRC", str(_iris_root / "src"))
os.environ.setdefault("CHECKPOINT_DIR", str(_iris_root / "checkpoints"))

# Auto-select the best available compute device: MPS (Apple Silicon) > CUDA > CPU.
# Can be overridden by setting DEFAULT_DEVICE in the environment before launching.
if not os.environ.get("DEFAULT_DEVICE"):
    import torch
    if torch.backends.mps.is_available():
        _default_device = "mps"
    elif torch.cuda.is_available():
        _default_device = "cuda"
    else:
        _default_device = "cpu"
    os.environ["DEFAULT_DEVICE"] = _default_device
    print(f"[run_backend] Auto-selected device: {_default_device}")

import uvicorn
uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
