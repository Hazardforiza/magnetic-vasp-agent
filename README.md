# Magnetic VASP Agent

Local control plane for magnetic VASP calculations. The application will plan
calculations locally, build validated VASP inputs from licensed local
pseudopotentials, and submit jobs to a Slurm cluster over SSH.

The first milestone intentionally excludes LLM calls and cluster submission.
It establishes the deterministic boundary that those components must use:

1. inventory a local POTCAR library without copying POTCAR contents;
2. parse POSCAR into a typed `Structure` whose species line fixes the order;
3. represent a calculation goal and potential selection as typed data;
4. validate every selected potential against the inventory;
5. assemble POTCAR in POSCAR species order and record hashes.

## Local setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\magvasp doctor
```

The committed `uv.lock` is the reproducible dependency source. On a machine
with `uv` installed, `uv sync --extra dev` replaces the first two setup lines.

## Inventory the licensed PBE.54 library

```powershell
.\.venv\Scripts\magvasp potcar scan `
  --root "C:\Users\18355\vasp实例\vasp.pot\vasp.pot\赝势库\potpaw_PBE\potpaw_PBE.54" `
  --family PBE_54 `
  --output var\potcar_registry.json
```

The generated registry contains only metadata, paths, and SHA256 hashes. The
runtime registry and all generated POTCAR files are ignored by Git.

## Create a baseline selection plan

Species order comes from the POSCAR species line, never from hand-typed
arguments, so POTCAR blocks cannot drift out of order:

```powershell
.\.venv\Scripts\magvasp structure show tests\fixtures\Mn3Al\POSCAR

.\.venv\Scripts\magvasp potcar select `
  --registry var\potcar_registry.json `
  --policy config\potential_policy.json `
  --poscar tests\fixtures\Mn3Al\POSCAR `
  --candidate-id Mn3Al `
  --task static `
  --magnetic-mode collinear `
  --output var\Mn3Al.potential-plan.json
```

`--element Mn --element Al` remains available for compositions without a
POSCAR yet; it is mutually exclusive with `--poscar`. Only VASP 5 POSCAR files
with an explicit species line are accepted.

`tests/fixtures/*/POSCAR` are the structures from the reviewed reference
calculations (Mn3Al collinear, CuMnAs SOC with alphabetical species order,
monoclinic SrRuO3 with +U). The policy reproduces the POTCAR of all 23
reference runs byte-for-byte after line-ending normalisation.

The policy selector is a deterministic fallback and a test oracle. A future
LLM selector must emit the same `PotentialPlan` schema and pass the same
validator before POTCAR assembly is allowed.

Reference-example evidence is not universal scientific guidance. Its
confidence is deliberately below 1.0, and later workflow policy will require
review for low-confidence or previously unseen selections.

## Security boundary

- POTCAR content is never sent to an LLM or committed to source control.
- A model may select only potential names present in the local registry.
- File assembly verifies source hashes immediately before reading each file.
- Cluster access will use a fixed SSH profile and typed Slurm operations, not
  arbitrary model-generated shell commands.
