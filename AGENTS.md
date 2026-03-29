# Repository Guidelines

## Project Structure & Module Organization

- `moldvision/`: Primary Python package. Entry point is `python -m moldvision` (see `moldvision/__main__.py` and `moldvision/cli.py`).
- `scripts/`: Optional, standalone utilities for dataset prep/debugging and inference. Treat these as helpers, not a stable API surface.
- `datasets/`: Default working directory for local datasets created by the CLI (intentionally ignored by git).
- `docs/`: Notes on tooling and transfer/inference workflows.

## Build, Test, and Development Commands

Install dependencies (install PyTorch separately to match your CUDA/CPU setup):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Editable install (enables the `rfdetrw` console script):

```powershell
pip install -e .
rfdetrw --help
```

Common CLI commands:
- `python -m moldvision doctor`: Environment checks and fix hints.
- `python -m moldvision dataset --help`: Dataset creation/ingest/validation utilities.
- `python -m moldvision train ...`: Train RF-DETR (detect or seg).
- `python -m moldvision export ...`: Export to ONNX / TensorRT.

## Coding Style & Naming Conventions

- Python: 4-space indentation, PEP 8, and keep functions small and readable.
- Naming: `snake_case` for functions/files, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Prefer type hints on public functions and CLI-adjacent code paths.
- Keep CLI flags/backward compatibility stable; update `README.md` when changing user-facing behavior.

## Testing Guidelines

- There is no automated test suite in this repo yet.
- Minimum smoke checks for changes:
  - `python -m moldvision --help`
  - `python -m moldvision doctor`
  - For dataset changes, run `python -m moldvision dataset validate -d datasets/<UUID> --task detect|seg` on a small sample dataset.

## Commit & Pull Request Guidelines

- Commit subjects in existing history are short and descriptive (e.g., `repo initialization`, `general fixes`). Prefer an imperative, concise subject line.
- PRs should include: what/why, commands run, and any dataset UUIDs referenced. Add screenshots when changing visualization outputs.
- Do not commit local artifacts: `datasets/`, `runs/`, exported models, or large weights (most are ignored by `.gitignore`).

## Security & Configuration Tips

- Keep secrets in `.env` files (ignored by git). Never commit API keys or paths to private datasets.
- TensorRT export requires `trtexec` available on `PATH`; document any environment assumptions in `docs/` when adding new tooling.
