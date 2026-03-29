"""
Application-level configuration for MoldVision.

Config file location (platform-aware):
  Windows : %LOCALAPPDATA%\MoldVision\config.json
  macOS   : ~/Library/Application Support/MoldVision/config.json
  Linux   : $XDG_CONFIG_HOME/moldvision/config.json  (default: ~/.config/moldvision/config.json)

Environment variable overrides take precedence over config file values.

Available settings (config.json keys / env vars):
  default_dataset_root   MOLDVISION_DATASETS        Where 'dataset create' stores datasets
  default_num_workers    MOLDVISION_NUM_WORKERS      DataLoader worker count (0 on Windows)
  inference_backend      MOLDVISION_BACKEND          Default inference backend (auto/onnx/tensorrt/pytorch)
  export_format          MOLDVISION_EXPORT_FORMAT    Default export format (onnx/tensorrt/onnx_fp16/onnx_quantized)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Environment variable names
ENV_DATASETS = "MOLDVISION_DATASETS"
ENV_NUM_WORKERS = "MOLDVISION_NUM_WORKERS"
ENV_BACKEND = "MOLDVISION_BACKEND"
ENV_EXPORT_FORMAT = "MOLDVISION_EXPORT_FORMAT"

_APP_NAME_WIN = "MoldVision"
_APP_NAME_UNIX = "moldvision"

# Canonical allowed values — used for validation in CLI
VALID_BACKENDS = ("auto", "onnx", "tensorrt", "pytorch")
VALID_EXPORT_FORMATS = ("onnx", "tensorrt", "onnx_fp16", "onnx_quantized")


def config_dir() -> Path:
    """Return the platform-appropriate directory that holds config.json."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / _APP_NAME_WIN
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APP_NAME_WIN
    # Linux / other POSIX
    xdg = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(xdg) / _APP_NAME_UNIX


def config_path() -> Path:
    """Return the full path to the JSON config file."""
    return config_dir() / "config.json"


def load_config() -> Dict[str, Any]:
    """Load config from disk.  Returns an empty dict if the file is missing or unreadable."""
    p = config_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config(cfg: Dict[str, Any]) -> None:
    """Persist *cfg* to the config file, creating parent directories as needed."""
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Individual setting helpers
# ---------------------------------------------------------------------------

def get_default_dataset_root() -> str:
    """
    Resolve the default dataset root using the priority chain:
      1. Environment variable  MOLDVISION_DATASETS
      2. Config file           config.json  »  default_dataset_root
      3. Hard fallback         "datasets"   (relative to CWD — legacy behaviour)
    """
    env = os.environ.get(ENV_DATASETS)
    if env:
        return env
    cfg = load_config()
    root = cfg.get("default_dataset_root")
    if root and isinstance(root, str):
        return root
    return "datasets"


def set_default_dataset_root(path: str) -> None:
    """Persist a new default_dataset_root to the config file."""
    cfg = load_config()
    cfg["default_dataset_root"] = str(Path(path).expanduser())
    save_config(cfg)


# ---------------------------------------------------------------------------
# num_workers
# ---------------------------------------------------------------------------

def get_default_num_workers() -> int:
    """Return the default DataLoader worker count.

    Priority: env var ``MOLDVISION_NUM_WORKERS`` → config file → OS heuristic.
    On Windows multiprocessing with DataLoader is error-prone, so the OS
    heuristic defaults to 0 there and 4 on Linux/macOS.
    """
    env = os.environ.get(ENV_NUM_WORKERS)
    if env is not None:
        try:
            return int(env)
        except ValueError:
            pass
    cfg = load_config()
    val = cfg.get("default_num_workers")
    if val is not None:
        try:
            return int(val)
        except (TypeError, ValueError):
            pass
    # OS heuristic
    return 0 if sys.platform == "win32" else 4


def set_default_num_workers(value: int) -> None:
    """Persist the default DataLoader worker count."""
    cfg = load_config()
    cfg["default_num_workers"] = int(value)
    save_config(cfg)


# ---------------------------------------------------------------------------
# inference_backend
# ---------------------------------------------------------------------------

def get_default_inference_backend() -> str:
    """Return the preferred inference backend.

    Priority: env var ``MOLDVISION_BACKEND`` → config file → ``"auto"``.
    """
    env = os.environ.get(ENV_BACKEND)
    if env and env in VALID_BACKENDS:
        return env
    cfg = load_config()
    val = cfg.get("inference_backend")
    if val and val in VALID_BACKENDS:
        return str(val)
    return "auto"


def set_default_inference_backend(value: str) -> None:
    """Persist the preferred inference backend."""
    if value not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend {value!r}. Choose from: {', '.join(VALID_BACKENDS)}")
    cfg = load_config()
    cfg["inference_backend"] = value
    save_config(cfg)


# ---------------------------------------------------------------------------
# export_format
# ---------------------------------------------------------------------------

def get_default_export_format() -> str:
    """Return the preferred export format.

    Priority: env var ``MOLDVISION_EXPORT_FORMAT`` → config file → ``"onnx"``.
    """
    env = os.environ.get(ENV_EXPORT_FORMAT)
    if env and env in VALID_EXPORT_FORMATS:
        return env
    cfg = load_config()
    val = cfg.get("export_format")
    if val and val in VALID_EXPORT_FORMATS:
        return str(val)
    return "onnx"


def set_default_export_format(value: str) -> None:
    """Persist the preferred export format."""
    if value not in VALID_EXPORT_FORMATS:
        raise ValueError(f"Unknown format {value!r}. Choose from: {', '.join(VALID_EXPORT_FORMATS)}")
    cfg = load_config()
    cfg["export_format"] = value
    save_config(cfg)


def get_setting(key: str) -> Optional[Any]:
    """Generic getter for any config key."""
    return load_config().get(key)


def set_setting(key: str, value: Any) -> None:
    """Generic setter — persists a single key into the config file."""
    cfg = load_config()
    cfg[key] = value
    save_config(cfg)
