# dl-from-scratch

PyTorch modules and paper reproductions built from first principles - `nn.Module` + `nn.Parameter` + matmul + arithmetics. No `nn.Linear` or other high-level shortcuts.

## Structure

- `core_modules/` - reusable building blocks (attention, norm, positional, FFN)
- `papers/` - one file per paper reproduced (architecture assembly + smoke check)

## Setup

uv venv && source .venv/bin/activate
uv pip install -e .
