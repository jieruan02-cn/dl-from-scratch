# Roadmap & Project Strategy

This document captures the direction for `ml-from-scratch` and the broader
paper-reproduction / portfolio strategy it feeds into. It records *decisions and
rationale*, not implementation detail.

## 1. Guiding split — two distinct learning goals

| Goal | Where it lives | What it proves |
| --- | --- | --- |
| Understand the primitives deeply | **this repo** (`ml-from-scratch`) — rebuild `torch.nn` from `Parameter` + matmul | I know how the building blocks work |
| Reproduce & train real architectures | **separate reproduction repos** — real PyTorch, full data + training + eval | I can reproduce results in practice |

These are complementary, not redundant. The decisive reason to keep both:
if the reproduction repos use real `nn.*`, the from-scratch modules in this repo
would *never* be exercised inside a real architecture — they'd only ever be
unit-tested in isolation. Assembling them here is the only thing that proves they
**compose** into a model that can actually learn.

**Avoid:** making this repo's `papers/` use real PyTorch. That leaves
`core_modules/` permanently untested in composition, defeating the point.

## 2. This repo (`ml-from-scratch`)

### Status
- `core_modules/` transformer stack complete (embedding, attention, encoder/decoder
  layers, full `Transformer`), hand-verified against `nn.*` references at ~1e-15.
- 7/10 modules still lack unit tests.

### `papers/` convention (decided)
- **Flat: one file per paper** (e.g. `papers/transformer.py`), no subfolders.
- Each file contains the **architecture assembly from `core_modules`** plus a
  **small smoke check** guarded under `if __name__ == "__main__":`.
- Smoke check = instantiate full model → forward + backward on a dummy batch →
  assert output shape → **overfit one synthetic batch (copy/reverse) to ~0 loss**.
  The overfit step is the payoff: it proves the stack can *learn*, not just typecheck.
- **Architecture only** — no data pipeline, training infra, or evaluation here.
- CLAUDE.md already updated to reflect this convention.

### First task: `papers/transformer.py`
Assemble the paper's full model. Missing pieces beyond the existing `Transformer` body:
- input embedding scaled by √d_model
- sinusoidal positional encoding
- final output projection → softmax
- weight tying between embedding and pre-softmax projection
- the `__main__` smoke check (overfit a copy/reverse batch)

### Deferred (not dropped)
- Unit tests for the 7 uncovered modules — lower learning value right now, but the
  "each module has a test" rule stands. `attention.py` parity tests are the
  highest-value to add first (locks in the ~1e-15 checks already run by hand).

### Possible rename (undecided)
- Candidate: **`dl-from-scratch`** (recommended) — "deep learning" is more accurate
  than "ml" (no classical ML here); avoid "architecture" in the name since
  `core_modules` is building blocks, not architectures.
- If renamed: update the `# ml-from-scratch` H1 in CLAUDE.md and `name` in
  `pyproject.toml` to match.

## 3. Reproduction repos & portfolio strategy

### Principle
A profile is polished by **~4–6 substantial, documented, runnable repos**, not by
many thin ones (GitHub gives 6 pinned slots — that's the real budget).
One-repo-per-paper fragments the story and hurts the profile.

### Sharing mechanism — published package, not monorepo
Share common code via an **installable, versioned package** rather than a monorepo.
For a learning project the packaging ceremony is friction; for a *portfolio* it is
itself the engineering signal worth showing ("I extracted a reusable lib and have
downstream projects consume it"). This gives the multi-repo profile **without**
copy-paste drift.

### Proposed pinned lineup
1. **`dl-from-scratch`** — primitives rebuilt from matmul (this repo). *Already exists.*
2. **`ml-utils`** (a.k.a. `tinytrain`) — shared data loading, transforms, training
   loop, eval/metrics. The DRY backbone; a portfolio piece on its own.
3. **`vision-reproductions`** — vision papers, real PyTorch, trained + evaluated.
4. **`language-reproductions`** — language papers (a nanoGPT-style GPT is a strong anchor).
5. **`vla`** — vision-language-action; the robotics differentiator. Worth isolating
   and polishing with a demo video — this is the repo that makes the profile *mine*.

Group reproductions **by modality**. VLM and VLA are *compositions* of vision +
language (VLA adds action), so they sit together under a `multimodal`/`vla` umbrella,
sharing the vision + language input stack and differing mainly on the output side.
Repos 3–5 each depend on `ml-utils`.

### Fallback
If `ml-utils` packaging becomes too much overhead mid-learning: collapse
vision + language + multimodal into a single monorepo and keep `ml-utils` + `vla`
standalone — still 3 strong pins.

### Emphasis
Quality over count: real READMEs, reported results/benchmarks, and runnable demos
matter more than the number of repos.

## 4. Open decisions
- [ ] Rename this repo to `dl-from-scratch`? (and sync CLAUDE.md H1 + `pyproject.toml`)
- [ ] Confirm the `ml-utils` vs. per-repo boundary (what's shared vs. repo-local).
- [ ] When to circle back and add deferred unit tests (start with `attention.py`).

## 5. Immediate next step
Build `papers/transformer.py` (architecture assembly + `__main__` smoke check).
