# mCIF Research Workflows

`research.from-mcif` is the mCIF-first entry point for a magnetic-materials
study. It writes a `research-plan.json`, a typed `workflow.request.json`, and
hashes for generated POSCAR files. It never runs VASP or an external solver.

```powershell
uv run --frozen magvasp research from-mcif `
  --mcif tests\fixtures\generated_mcif\candidate_01_FeO_BNS2_7.mcif `
  --output var\feo-research `
  --task magnetic_order --task relax --task static --task dos `
  --task mae --task convergence
```

The default VASP stages are:

1. bounded collinear sign variants from the mCIF moments;
2. independent ionic relaxations for selected magnetic starts;
3. stationary calculations for energy and local-moment comparison;
4. DOS from a verified stationary `CHGCAR`;
5. self-consistent SOC directions for MAE;
6. independent ENCUT and k-point-spacing convergence runs.

The planner does not silently pass a relaxation `CONTCAR` into a static run.
After relaxation, re-import the chosen `CONTCAR`, regenerate a static workflow,
and then release its `CHGCAR` to DOS or band calculations after execution
verification.

For correlated materials, supply reviewed U/J values explicitly. The planner
does not infer U:

```json
{
  "tasks": ["magnetic_order", "static", "hubbard_sensitivity"],
  "hubbard_sets": {
    "u4": {"Fe": {"l": 2, "u": 4.0, "j": 0.0}},
    "u6": {"Fe": {"l": 2, "u": 6.0, "j": 0.0}}
  }
}
```

The following stages are represented as explicit plans with expected inputs,
outputs, and acceptance criteria because they need a different solver or a
scientific model beyond the VASP input compiler: `exchange`,
`finite_temperature`, `spin_dynamics`, `phonon`, `elastic`, `strain`,
`defect`, `dmi`, `neb`, `wannier`, `berry_transport`, `hse`, `gw`, and `dmft`.
They are never reported as completed calculations.

After importing result records, use the read-only analyses:

```powershell
uv run --frozen magvasp magnetic-analysis compare results.json
uv run --frozen magvasp magnetic-analysis mae mae-results.json
uv run --frozen magvasp magnetic-analysis exchange exchange-results.json
uv run --frozen magvasp magnetic-analysis sensitivity results.json --parameter U
```

Energy comparison prefers `energy_sigma0`. When per-formula-unit comparison is
requested but any candidate lacks formula-unit metadata, ranking is withheld
instead of mixing incompatible total-cell and normalized energies. The
analysis output is advisory and remains `scientific_converged=false` until
compatibility, convergence, final moments, and human review are established.

