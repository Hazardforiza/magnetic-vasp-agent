from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from magnetic_vasp_agent.cli import app
from magnetic_vasp_agent.incar import read_incar
from magnetic_vasp_agent.models import (
    PotentialPolicy,
    PotentialRecord,
    PotentialRegistry,
)
from magnetic_vasp_agent.poscar import read_poscar
from magnetic_vasp_agent.workflow import (
    digest,
    expand_steps,
    inspect_bundle,
    materialize_workflow,
    prepare_workflow,
)
from magnetic_vasp_agent.workflow_models import (
    CalculationStep,
    ExecutionCapabilities,
    Workflow,
)

ROOT = Path(__file__).parents[1]


@pytest.fixture
def context(tmp_path):
    records = []
    library = tmp_path / "synthetic-library"
    for element in ["Mn", "Al", "Cu", "As", "Sr", "Ru", "O"]:
        path = library / element / "POTCAR"
        path.parent.mkdir(parents=True)
        payload = f"SYNTHETIC TEST DATA, NOT A VASP POTENTIAL: {element}\n".encode()
        path.write_bytes(payload)
        records.append(
            PotentialRecord(
                element=element,
                potential=element,
                family="PBE_54",
                titel=f"Synthetic {element}",
                enmax=400,
                relative_path=f"{element}/POTCAR",
                sha256=digest(payload),
            )
        )
    registry = PotentialRegistry(root=str(library), family="PBE_54", records=records)
    policy = PotentialPolicy(family="PBE_54", preferred={})
    return registry, policy, tmp_path


def step(system="Mn3Al", name="scf", task="static"):
    suffix = {"Mn3Al": "static", "CuMnAs": "soc", "SrRuO3": "soc-u"}[system]
    incar = json.loads(
        (ROOT / f"config/incar_requests/{system}.{suffix}.json").read_text()
    )
    kpoints = json.loads(
        (ROOT / f"config/kpoints_requests/{system}.{suffix}.json").read_text()
    )
    incar["goal"]["task"] = kpoints["goal"]["task"] = task
    if task == "band":
        kpoints["sampling"] = json.loads(
            (ROOT / "config/kpoints_requests/Mn3Al.band.json").read_text()
        )["sampling"]
    return {
        "id": name,
        "candidate_id": system,
        "poscar": str(ROOT / f"tests/fixtures/{system}/POSCAR"),
        "incar": incar,
        "kpoints": kpoints,
    }


def chain(system="Mn3Al", task="band"):
    downstream = step(system, task, task)
    downstream["dependencies"] = [{"artifact": "CHGCAR", "from_step": "scf"}]
    return {"workflow_id": "test", "steps": [step(system), downstream]}


def compile_request(data, context):
    registry, policy, base = context
    return prepare_workflow(Workflow.model_validate(data), registry, policy, base)


@pytest.mark.parametrize("system", ["Mn3Al", "CuMnAs", "SrRuO3"])
def test_four_file_packages_mapping_and_full_context(system, context):
    prepared = compile_request(
        {"workflow_id": "single", "steps": [step(system)]}, context
    )
    result = prepared.steps["scf"]
    assert set(result.payloads) == {"POSCAR", "POTCAR", "INCAR", "KPOINTS"}
    report = result.manifest["validation"]
    assert report["valid"]
    assert "ENCUT_UNCHECKED" not in {
        x["code"] for x in report["incar_validation"]["diagnostics"]
    }
    mapping = result.manifest["atom_mapping"]
    assert [a["species"] for a in mapping] == result.structure.site_species
    assert [m for a in mapping for m in a["magmom"]] == result.incar.MAGMOM
    assert result.manifest["potcar"]["source_sha256"]
    out = context[2] / "bundle"
    materialize_workflow(prepared, out)
    assert inspect_bundle(out) == {
        "integrity_valid": True,
        "runtime_verified": False,
        "workflow_id": "single",
        "steps": 1,
        "waiting_for_upstream": [],
    }
    assert read_incar(out / "scf/INCAR").spec == result.incar
    assert read_poscar(out / "scf/POSCAR").site_species == result.structure.site_species
    manifest = json.loads((out / "workflow.manifest.json").read_text())
    assert manifest["steps"]["scf"]["status"]["runtime_verified"] is False
    assert "SYNTHETIC TEST DATA" not in json.dumps(manifest)


@pytest.mark.parametrize("system", ["Mn3Al", "CuMnAs", "SrRuO3"])
@pytest.mark.parametrize("task", ["dos", "band"])
def test_charge_dependency_with_different_sampling(system, task, context):
    data = chain(system, task)
    if task == "dos":
        data["steps"][1]["kpoints"]["sampling"]["mesh"] = [24, 24, 24]
    data["steps"].reverse()
    prepared = compile_request(data, context)
    assert prepared.manifest["order"] == ["scf", task]
    out = context[2] / "chain"
    materialize_workflow(prepared, out)
    assert not (out / task / "CHGCAR").exists()
    assert inspect_bundle(out)["waiting_for_upstream"] == [task]
    dep = prepared.steps[task].manifest["status"]["dependencies"][0]
    assert dep == {
        "artifact": "CHGCAR",
        "from_step": "scf",
        "sha256": None,
        "status": "pending_upstream",
    }


def test_branching_scf_dos_band(context):
    data = chain()
    dos = step(name="dos", task="dos")
    dos["dependencies"] = [{"artifact": "CHGCAR", "from_step": "scf"}]
    data["steps"].append(dos)
    result = compile_request(data, context)
    assert result.manifest["dependency_graph"] == {
        "scf": [],
        "band": ["scf"],
        "dos": ["scf"],
    }


@pytest.mark.parametrize(
    "name",
    ["../escape", "a/b", "a\\b", "a:b", "CON", "nul", "LPT1", ".", "", "x.", "x "],
)
def test_unsafe_names_rejected(name):
    data = step()
    data["id"] = name
    with pytest.raises(ValidationError):
        CalculationStep.model_validate(data)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda d: d["steps"][1]["dependencies"].clear(), "ARTIFACT_BINDING"),
        (
            lambda d: d["steps"][1]["dependencies"][0].update(from_step="absent"),
            "MISSING_STEP",
        ),
        (
            lambda d: d["steps"][0].update(
                dependencies=[{"artifact": "CHGCAR", "from_step": "band"}]
            ),
            "DEPENDENCY_CYCLE",
        ),
        (
            lambda d: d["steps"][0].update(incar_options={"LCHARG": False}),
            "ARTIFACT_OUTPUT_DISABLED",
        ),
        (lambda d: d["steps"][1]["incar"].update(encut=601), "ARTIFACT_COMPATIBILITY"),
        (
            lambda d: d["steps"][1]["incar"].update(magmom=[1, 1, -2, 0]),
            "ARTIFACT_COMPATIBILITY",
        ),
        (lambda d: d["steps"][1].update(candidate_id="other"), "ARTIFACT_STRUCTURE"),
        (lambda d: d["steps"][0]["incar"].update(encut=300), "ENCUT_BELOW_ENMAX"),
        (
            lambda d: d["steps"][1]["kpoints"]["goal"].update(task="static"),
            "goals must match",
        ),
        (lambda d: d["steps"].append(deepcopy(d["steps"][0])), "DUPLICATE_STEP"),
        (
            lambda d: d["steps"].append({**deepcopy(d["steps"][0]), "id": "SCF"}),
            "DUPLICATE_STEP",
        ),
        (lambda d: d["steps"][1].update(incar_options={"ENCUT": 999}), "incar_options"),
        (lambda d: d["steps"][1].update(incar_options={"LCHARG": None}), "non-null"),
    ],
)
def test_invalid_requests_fail_without_outputs(mutation, match, context):
    data = chain()
    mutation(data)
    with pytest.raises(ValueError, match=match):
        compile_request(data, context)
    assert not (context[2] / "bundle").exists()


def test_source_change_detected_by_potcar_hash(context):
    registry, _, base = context
    (Path(registry.root) / "Mn/POTCAR").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        compile_request({"workflow_id": "x", "steps": [step()]}, context)
    assert list(base.iterdir()) == [base / "synthetic-library"]


def test_species_order_plan_mismatch(context):
    result = compile_request({"workflow_id": "x", "steps": [step()]}, context)
    data = step()
    plan = deepcopy(result.steps["scf"].manifest["potential_plan"])
    plan["species_order"].reverse()
    plan["choices"].reverse()
    data["potential_plan"] = plan
    with pytest.raises(ValueError, match="species order"):
        compile_request({"workflow_id": "x", "steps": [data]}, context)


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_id", "other"),
        ("goal", {"task": "dos", "magnetic_mode": "collinear"}),
    ],
)
def test_plan_must_match_step(field, value, context):
    result = compile_request({"workflow_id": "x", "steps": [step()]}, context)
    data = step()
    data["potential_plan"] = deepcopy(result.steps["scf"].manifest["potential_plan"])
    data["potential_plan"][field] = value
    with pytest.raises(ValueError, match="must match"):
        CalculationStep.model_validate(data)


@pytest.mark.parametrize(
    "mutate", ["positions", "lattice", "atom_order", "nonfinite", "singular"]
)
def test_structure_changes_and_invalid_cells_rejected(mutate, context):
    from magnetic_vasp_agent.poscar import render_poscar

    data = chain()
    structure = read_poscar(Path(data["steps"][1]["poscar"]))
    if mutate == "positions":
        structure.positions[0][0] += 0.01
    elif mutate == "lattice":
        structure.lattice[0][0] += 0.01
    elif mutate == "atom_order":
        structure.positions[0], structure.positions[1] = (
            structure.positions[1],
            structure.positions[0],
        )
    elif mutate == "nonfinite":
        structure.positions[0][0] = float("nan")
    else:
        structure.lattice[1] = structure.lattice[0][:]
    path = context[2] / "changed.POSCAR"
    path.write_text(render_poscar(structure))
    data["steps"][1]["poscar"] = str(path)
    with pytest.raises(ValueError, match="ARTIFACT_STRUCTURE|INVALID_CELL"):
        compile_request(data, context)


def test_wavecar_restart_separate_rules(context):
    data = {"workflow_id": "wave", "steps": [step(), step(name="restart")]}
    data["steps"][0]["incar_options"] = {"LWAVE": True, "NBANDS": 48}
    data["steps"][1].update(
        incar_options={"ISTART": 1, "ICHARG": 0, "NBANDS": 48},
        dependencies=[{"artifact": "WAVECAR", "from_step": "scf"}],
    )
    result = compile_request(data, context)
    assert (
        result.manifest["steps"]["restart"]["status"]["input_state"]
        == "waiting_for_upstream"
    )
    data["steps"][1]["kpoints"]["sampling"]["mesh"] = [9, 9, 9]
    with pytest.raises(ValueError, match="WAVECAR_KPOINTS"):
        compile_request(data, context)
    data["steps"][1]["incar_options"].pop("NBANDS")
    with pytest.raises(ValueError, match="WAVECAR_NBANDS"):
        compile_request(data, context)


def test_charge_and_wavecar_two_sources_declared(context):
    data = {"workflow_id": "wave", "steps": [step(), step(name="restart")]}
    data["steps"][0]["incar_options"] = {"LWAVE": True, "NBANDS": 48}
    data["steps"][1].update(
        incar_options={"ISTART": 1, "ICHARG": 1, "NBANDS": 48},
        dependencies=[
            {"artifact": a, "from_step": "scf"} for a in ["WAVECAR", "CHGCAR"]
        ],
    )
    assert (
        len(
            compile_request(data, context)
            .steps["restart"]
            .manifest["status"]["dependencies"]
        )
        == 2
    )


def scan(kind="mae"):
    base = step(
        "CuMnAs", name="scan", task="magnetic_anisotropy" if kind == "mae" else "static"
    )
    variants = (
        {"axes": {"001": [0, 0, 1], "011": [0, 1, 1], "111": [1, 1, 1]}}
        if kind == "mae"
        else {"values": {"zero": 0, "half": 0.5, "full": 1}}
    )
    return {"workflow_id": "scan", "scans": [{"kind": kind, "base": base, **variants}]}


def test_mae_three_directions_only_saxis_changes(context):
    result = compile_request(scan(), context)
    assert list(result.steps) == ["scan-001", "scan-011", "scan-111"]
    specs = [s.incar.model_dump(exclude={"SAXIS"}) for s in result.steps.values()]
    assert specs[0] == specs[1] == specs[2]
    assert result.manifest["comparisons"][0]["method"] == "self_consistent_soc"
    assert not result.manifest["comparisons"][0]["results_available"]


def test_mae_requires_collinear_spin_frame_moments(context):
    data = scan()
    data["scans"][0]["base"]["incar"]["magmom"][0] = 1
    with pytest.raises(ValueError, match="MAE_MOMENTS"):
        compile_request(data, context)


@pytest.mark.parametrize(
    "axes", [{"a": [0, 0, 0], "b": [1, 0, 0]}, {"a": [1, 0, 0], "b": [2, 0, 0]}]
)
def test_mae_bad_axes_rejected(axes):
    data = scan()
    data["scans"][0]["axes"] = axes
    with pytest.raises(ValueError):
        Workflow.model_validate(data)


def test_socscale_requires_evidence_and_preserves_fractional_scale(context):
    data = scan("socscale")
    with pytest.raises(ValueError, match="SOCSCALE_CAPABILITY"):
        compile_request(data, context)
    with pytest.raises(ValueError, match="evidence"):
        ExecutionCapabilities(socscale=True)
    data["capabilities"] = {
        "socscale": True,
        "evidence": "Synthetic test capability, not a real VASP build",
    }
    result = compile_request(data, context)
    assert [s.incar.SOCSCALE for s in result.steps.values()] == [0, 0.5, 1]
    assert result.steps["scan-half"].payloads["INCAR"].find(b"SOCSCALE = 0.5") >= 0


def test_scan_rejects_restart_method(context):
    data = scan()
    data["scans"][0]["base"]["incar_options"] = {"ICHARG": 12}
    with pytest.raises(ValueError, match="SCAN_METHOD|RESTART_MODE"):
        compile_request(data, context)


def test_expand_revalidates_combined_ids():
    data = scan()
    data["steps"] = [step(name="scan-001")]
    with pytest.raises(ValueError, match="DUPLICATE_STEP"):
        expand_steps(Workflow.model_validate(data))


def test_deterministic_bundles_and_no_overwrite(context):
    prepared = compile_request(chain(), context)
    first, second = context[2] / "one", context[2] / "two"
    materialize_workflow(prepared, first)
    materialize_workflow(compile_request(chain(), context), second)
    original = {
        p.relative_to(first): p.read_bytes() for p in first.rglob("*") if p.is_file()
    }
    assert original == {
        p.relative_to(second): p.read_bytes() for p in second.rglob("*") if p.is_file()
    }
    with pytest.raises(FileExistsError):
        materialize_workflow(prepared, first)
    assert original == {
        p.relative_to(first): p.read_bytes() for p in first.rglob("*") if p.is_file()
    }


def test_partial_failure_rolls_back_owned_staging_only(context, monkeypatch):
    prepared = compile_request(chain(), context)
    target = context[2] / "failed"
    unrelated = context[2] / "keep.txt"
    unrelated.write_text("keep")
    original = Path.write_bytes

    def fail(path, payload):
        if path.name == "KPOINTS":
            raise OSError("simulated disk failure")
        return original(path, payload)

    monkeypatch.setattr(Path, "write_bytes", fail)
    with pytest.raises(OSError, match="disk failure"):
        materialize_workflow(prepared, target)
    assert not target.exists()
    assert not list(context[2].glob(".magvasp-stage-*"))
    assert unrelated.read_text() == "keep"


@pytest.mark.parametrize(
    "filename", ["POSCAR", "POTCAR", "INCAR", "KPOINTS", "manifest.json"]
)
def test_inspect_rejects_tampered_files(filename, context):
    root = context[2] / "bundle"
    materialize_workflow(compile_request(chain(), context), root)
    (root / "scf" / filename).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        inspect_bundle(root)


def test_materializer_rejects_mutated_prepared_payload(context):
    result = compile_request(chain(), context)
    result.steps["scf"].payloads["INCAR"] = b"ENCUT=1\n"
    with pytest.raises(ValueError, match="changed"):
        materialize_workflow(result, context[2] / "bad")
    assert not (context[2] / "bad").exists()


def test_cli_validate_build_inspect_and_relative_paths(context):
    registry, policy, root = context
    data = chain()
    for item in data["steps"]:
        item["poscar"] = "input.POSCAR"
    (root / "input.POSCAR").write_bytes(
        (ROOT / "tests/fixtures/Mn3Al/POSCAR").read_bytes()
    )
    request = root / "workflow.json"
    request.write_text(json.dumps(data))
    (root / "registry.json").write_text(registry.model_dump_json())
    (root / "policy.json").write_text(policy.model_dump_json())
    runner = CliRunner()
    args = [
        "--request",
        str(request),
        "--registry",
        str(root / "registry.json"),
        "--policy",
        str(root / "policy.json"),
    ]
    before = set(root.iterdir())
    checked = runner.invoke(app, ["workflow", "validate", *args])
    assert checked.exit_code == 0, checked.output
    assert json.loads(checked.output)["validation"]["valid"]
    assert set(root.iterdir()) == before
    built = runner.invoke(
        app, ["workflow", "build", *args, "--output", str(root / "bundle")]
    )
    assert built.exit_code == 0, built.output
    assert (
        runner.invoke(app, ["workflow", "inspect", str(root / "bundle")]).exit_code == 0
    )
    assert (
        runner.invoke(
            app, ["workflow", "build", *args, "--output", str(root / "bundle")]
        ).exit_code
        == 2
    )
    data["steps"][1]["dependencies"].clear()
    request.write_text(json.dumps(data))
    assert runner.invoke(app, ["workflow", "validate", *args]).exit_code != 0


@pytest.mark.parametrize(
    "filename",
    [
        "Mn3Al.scf-dos-band.json",
        "CuMnAs.soc.json",
        "SrRuO3.soc-u.json",
        "Mn3Al.mae.json",
    ],
)
def test_checked_in_workflows(filename, context):
    root = ROOT / "config/workflows"
    workflow = Workflow.model_validate_json((root / filename).read_bytes())
    registry, policy, _ = context
    result = prepare_workflow(workflow, registry, policy, root)
    assert result.manifest["validation"]["valid"]
    assert "pymatgen" in result.manifest["provenance"]["runtime"]


def test_checked_in_socscale_stays_blocked(context):
    root = ROOT / "config/workflows"
    workflow = Workflow.model_validate_json(
        (root / "CuMnAs.socscale.json").read_bytes()
    )
    with pytest.raises(ValueError, match="SOCSCALE_CAPABILITY"):
        prepare_workflow(workflow, context[0], context[1], root)


def test_upstream_lmaxmix_and_hubbard_mismatch(context):
    data = chain("SrRuO3")
    for s in data["steps"]:
        s["incar_options"] = {"LMAXMIX": 2}
    with pytest.raises(ValueError, match="ARTIFACT_LMAXMIX"):
        compile_request(data, context)
    for s in data["steps"]:
        s.pop("incar_options")
    data["steps"][1]["incar"]["hubbard"]["Ru"]["u"] = 2
    with pytest.raises(ValueError, match="LDAUU"):
        compile_request(data, context)


def test_non_scf_and_relaxed_charge_producers_rejected(context):
    data = chain()
    for req in ["incar", "kpoints"]:
        data["steps"][0][req]["goal"]["task"] = "relax"
    with pytest.raises(ValueError, match="ARTIFACT_PRODUCER"):
        compile_request(data, context)
    data = chain(task="dos")
    data["steps"][1]["incar_options"] = {"LCHARG": True}
    third = step(name="band", task="band")
    third["dependencies"] = [{"artifact": "CHGCAR", "from_step": "dos"}]
    data["steps"].append(third)
    with pytest.raises(ValueError, match="ARTIFACT_PRODUCER"):
        compile_request(data, context)


@pytest.mark.parametrize("variant", ["scale", "mode", "tail", "velocity"])
def test_poscar_boundary_rejects_unmodeled_layout(variant, context):
    data = step()
    lines = Path(data["poscar"]).read_text().splitlines()
    if variant == "scale":
        lines[1] = "5.8 6 7"
    elif variant == "mode":
        lines[7] = "typo"
    elif variant == "tail":
        lines.append("unmodeled predictor corrector data")
    else:
        lines[-1] = "1 0 0"
    path = context[2] / "source.POSCAR"
    path.write_text("\n".join(lines) + "\n")
    data["poscar"] = str(path)
    with pytest.raises(ValueError, match="POSCAR_LAYOUT|VELOCITIES_UNSUPPORTED"):
        compile_request({"workflow_id": "bad", "steps": [data]}, context)


def test_restart_zero_charge_requires_wave_start(context):
    data = step()
    data["incar_options"] = {"ICHARG": 0}
    with pytest.raises(ValueError, match="RESTART_MODE"):
        compile_request({"workflow_id": "bad", "steps": [data]}, context)


def test_duplicate_potentials_and_duplicate_artifacts_rejected(context):
    registry, _, _ = context
    registry.records.append(registry.records[0].model_copy())
    with pytest.raises(ValueError, match="DUPLICATE_POTENTIAL"):
        compile_request(chain(), context)
    data = chain()
    data["steps"][1]["dependencies"] *= 2
    with pytest.raises(ValueError, match="exactly one source"):
        Workflow.model_validate(data)


def test_equivalent_physical_cell_preserves_dependency(context):
    from magnetic_vasp_agent.poscar import render_poscar

    data = chain()
    structure = read_poscar(Path(data["steps"][1]["poscar"]))
    structure.scale /= 2
    structure.lattice = [[x * 2 for x in row] for row in structure.lattice]
    structure.positions[0][0] += 1
    path = context[2] / "equivalent.POSCAR"
    path.write_text(render_poscar(structure))
    data["steps"][1]["poscar"] = str(path)
    assert compile_request(data, context).manifest["validation"]["valid"]


def test_inspect_malformed_manifest_cli(context):
    root = context[2] / "bad-manifest"
    root.mkdir()
    (root / "workflow.manifest.json").write_text("[]")
    result = CliRunner().invoke(app, ["workflow", "inspect", str(root)])
    assert result.exit_code == 2
    assert "Unsupported workflow manifest" in result.output


def test_output_appears_during_publish_is_not_overwritten(context, monkeypatch):
    import magnetic_vasp_agent.workflow as module

    result = compile_request(chain(), context)
    target = context[2] / "race"
    original = module.inspect_bundle

    def race(path):
        target.mkdir()
        (target / "owned-by-other.txt").write_text("keep")
        return original(path)

    monkeypatch.setattr(module, "inspect_bundle", race)
    with pytest.raises(FileExistsError):
        materialize_workflow(result, target)
    assert (target / "owned-by-other.txt").read_text() == "keep"
    assert list(target.iterdir()) == [target / "owned-by-other.txt"]
    assert not list(context[2].glob(".magvasp-stage-*"))
