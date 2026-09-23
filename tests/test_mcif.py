from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import spglib
from pymatgen.electronic_structure.core import Magmom
from pymatgen.io.cif import CifBlock, CifParser
from typer.testing import CliRunner

from magnetic_vasp_agent.cli import app
from magnetic_vasp_agent.mcif import map_moments, parse_mcif, read_mcif
from magnetic_vasp_agent.mcif_models import CandidateStructure, McifIncarRequest
from magnetic_vasp_agent.models import (
    PotentialPolicy,
    PotentialRecord,
    PotentialRegistry,
)
from magnetic_vasp_agent.workflow import (
    inspect_bundle,
    materialize_workflow,
    prepare_workflow,
)
from magnetic_vasp_agent.mcif_workflow import plan_workflow_from_mcif
from magnetic_vasp_agent.workflow_models import CalculationStep, Workflow

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests/fixtures/generated_mcif"
PROVENANCE = json.loads((FIXTURES / "provenance.json").read_text())


def synthetic(
    *, rows=None, ops=None, centers=None, parameters=(4, 4, 6, 90, 90, 90), updates=None
):
    """Construct explicitly synthetic CIF data using the existing CIF writer."""
    rows = rows or [("Fe1", "Fe", 0.13, 0.21, 0.31, 1, 2, 3)]
    ops = ops or ["x,y,z,+1"]
    names = [
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_moment_label",
        "_atom_site_moment_crystalaxis_x",
        "_atom_site_moment_crystalaxis_y",
        "_atom_site_moment_crystalaxis_z",
    ]
    cells = [
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    ]
    data = {key: str(value) for key, value in zip(cells, parameters, strict=True)}
    op_names = [
        "_space_group_symop_magn_operation.id",
        "_space_group_symop_magn_operation.xyz",
    ]
    data[op_names[0]] = [str(i + 1) for i in range(len(ops))]
    data[op_names[1]] = ops
    loops = [op_names, names]
    expanded_rows = [(*r[:5], r[0], *r[5:]) for r in rows]
    data.update(
        {name: [str(row[i]) for row in expanded_rows] for i, name in enumerate(names)}
    )
    if centers:
        c_names = [
            "_space_group_symop_magn_centering.id",
            "_space_group_symop_magn_centering.xyz",
        ]
        data[c_names[0]] = [str(i + 1) for i in range(len(centers))]
        data[c_names[1]] = centers
        loops.append(c_names)
    data.update(updates or {})
    return (str(CifBlock(data, loops, "synthetic_not_experimental")) + "\n").encode()


def request(mode="noncollinear", **kwargs):
    return McifIncarRequest(
        goal={"task": "static", "magnetic_mode": mode},
        moment_source="mcif",
        encut=600,
        **kwargs,
    )


@pytest.mark.parametrize("record", PROVENANCE["files"], ids=lambda r: r["file"])
@pytest.mark.filterwarnings("ignore:Set OLD_ERROR_HANDLING to false:DeprecationWarning")
def test_real_candidates_hash_moments_and_independent_magnetic_identification(record):
    payload = (FIXTURES / record["file"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == record["source_sha256"]
    result = parse_mcif(payload, record["file"].removesuffix(".mcif"))
    assert result.source_sha256 == record["source_sha256"]
    assert result.structure.species_order == ["Fe", "O"]
    assert result.structure.counts == [2, 2]
    assert result.declared_bns == "2.7"
    assert result.collinear
    assert [s.fractional for s in result.sites] == [
        [0, 0, 0],
        [0, 0, 0.5],
        [0.5, 0.5, 0.25],
        [0.5, 0.5, 0.75],
    ]
    m = np.array([s.moment_cartesian for s in result.sites])
    np.testing.assert_allclose(m[0], record["fe_cartesian_moment"], atol=5e-7, rtol=0)
    np.testing.assert_allclose(m[1], -m[0], atol=1e-12)
    np.testing.assert_allclose(m[2:], 0, atol=1e-12)
    assert np.linalg.norm(m[0]) == pytest.approx(record["fe_moment_norm"], abs=1e-10)
    assert result.total_moment_cartesian == [0, 0, 0]
    assert [len(s.origins) for s in result.sites] == [2, 2, 2, 2]
    assert result.sites[1].origins[0].operation_id == "3"
    assert result.sites[1].origins[0].source_label == "Fe1"
    assert all(
        s.wyckoff_declared is None and not s.occupancy_explicit
        for s in result.source_sites
    )
    assert "OCCUPANCY_DEFAULT_ONE" in result.diagnostics[0]
    pmg = CifParser.from_str(
        payload.decode(), frac_tolerance=0, check_cif=False
    ).parse_structures(primitive=False)[0]
    np.testing.assert_allclose(pmg.lattice.matrix, result.structure.lattice, atol=1e-12)
    np.testing.assert_allclose(
        [v.global_moment for v in pmg.site_properties["magmom"]], m, atol=1e-10
    )
    detected = spglib.get_magnetic_symmetry_dataset(
        (result.structure.lattice, result.structure.positions, [26, 26, 8, 8], m),
        symprec=0.005,
    )
    assert spglib.get_magnetic_spacegroup_type(detected.uni_number).bns_number == "2.7"
    assert CandidateStructure.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "axis", [[0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, -1]]
)
@pytest.mark.parametrize("mode", ["noncollinear", "soc"])
def test_spin_basis_roundtrip_preserves_actual_direction(axis, mode):
    candidate = read_mcif(FIXTURES / PROVENANCE["files"][0]["file"])
    incar, mapping = map_moments(candidate, request(mode, saxis=axis))
    assert mapping.orientation_preserved
    assert mapping.maximum_roundtrip_error < 1e-7
    assert len(incar.magmom) == 12
    vectors = np.array(incar.magmom).reshape(-1, 3)
    restored = [
        Magmom(v, saxis=np.array(axis, dtype=float)).global_moment for v in vectors
    ]
    np.testing.assert_allclose(
        restored, [s.moment_cartesian for s in candidate.sites], atol=1e-7
    )
    if axis == [0, 0, 1]:
        assert not np.allclose(
            incar.magmom[:3], candidate.source_sites[0].moment_crystalaxis
        )


def test_collinear_projection_explicit_and_signed():
    candidate = read_mcif(FIXTURES / PROVENANCE["files"][0]["file"])
    with pytest.raises(ValueError, match="MCIF_PROJECTION"):
        map_moments(candidate, request("collinear"))
    incar, mapping = map_moments(
        candidate, request("collinear", moment_mode="collinear_projection")
    )
    np.testing.assert_allclose(
        incar.magmom, [-1.4177741788713274, 1.4177741788713274, 0, 0], atol=1e-8
    )
    assert not mapping.orientation_preserved
    assert mapping.projection_axis_cartesian is not None
    with pytest.raises(ValueError, match="MCIF_PROJECTION"):
        map_moments(candidate, request("soc", moment_mode="collinear_projection"))


def test_noncollinear_texture_rejected_by_scalar_route():
    data = synthetic(
        rows=[("Fe1", "Fe", 0, 0, 0, 1, 0, 0), ("Fe2", "Fe", 0.5, 0.5, 0.5, 0, 1, 0)]
    )
    result = parse_mcif(data, "texture")
    assert not result.collinear
    assert result.collinear_axis_cartesian is None
    with pytest.raises(ValueError, match="MCIF_NONCOLLINEAR"):
        map_moments(result, request("collinear", moment_mode="collinear_projection"))
    incar, _ = map_moments(result, request())
    np.testing.assert_allclose(incar.magmom, [1, 0, 0, 0, 1, 0], atol=1e-8)


def test_oblique_basis_rotation_does_not_act_directly_on_crystal_components():
    data = synthetic(
        parameters=(2, np.sqrt(2), 3, 90, 90, 45),
        ops=["x,y,z,+1", "x+y,-y,-z,+1"],
        rows=[("Fe1", "Fe", 0.13, 0.21, 0.31, 0, 1, 1)],
    )
    result = parse_mcif(data, "oblique")
    np.testing.assert_allclose(
        result.sites[0].moment_cartesian, [2**-0.5, 2**-0.5, 1], atol=1e-8
    )
    np.testing.assert_allclose(
        result.sites[1].moment_cartesian, [2**-0.5, -(2**-0.5), -1], atol=1e-8
    )
    assert np.linalg.norm(result.sites[0].moment_cartesian) == pytest.approx(
        np.linalg.norm(result.sites[1].moment_cartesian)
    )


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("-x,-y,-z,+1", [1, 2, 3]),
        ("-x,y,z,+1", [1, -2, -3]),
        ("x,y,z+1/2,-1", [-1, -2, -3]),
    ],
)
def test_axial_vector_and_time_reversal(operation, expected):
    result = parse_mcif(synthetic(ops=["x,y,z,+1", operation]), "axial")
    np.testing.assert_allclose(result.sites[1].moment_cartesian, expected, atol=1e-8)


def test_anti_centering_and_full_cell_deduplication():
    result = parse_mcif(synthetic(centers=["x,y,z,+1", "x,y,z+1/2,-1"]), "centered")
    assert len(result.sites) == 2
    assert result.sites[1].origins[0].centering_id == "2"
    np.testing.assert_allclose(result.total_moment_cartesian, 0)
    full = parse_mcif(
        synthetic(
            ops=["x,y,z,+1", "x,y,z+1/2,-1"],
            rows=[
                ("Fe1", "Fe", 0.13, 0.21, 0.31, 1, 2, 3),
                ("Fe2", "Fe", 0.13, 0.21, 0.81, -1, -2, -3),
            ],
        ),
        "full",
    )
    assert len(full.sites) == 2
    assert {o.source_label for o in full.sites[0].origins} == {"Fe1", "Fe2"}


def test_species_regrouping_keeps_moment_label_identity():
    rows = [
        ("O1", "O", 0, 0, 0, 0, 0, 0),
        ("Fe1", "Fe", 0.2, 0.2, 0.2, 1, 0, 0),
        ("O2", "O", 0.4, 0.4, 0.4, 0, 0, 0),
    ]
    result = parse_mcif(synthetic(rows=rows), "order")
    assert result.structure.site_species == ["O", "O", "Fe"]
    assert [s.origins[0].source_label for s in result.sites] == ["O1", "O2", "Fe1"]
    assert result.sites[-1].moment_cartesian == [1, 0, 0]


@pytest.mark.parametrize("value", ["?", ".", "NaN", "inf", "1e999", "2junk"])
def test_unknown_or_nonfinite_moment_never_defaults_to_zero(value):
    with pytest.raises(ValueError, match="MCIF_NUMBER"):
        parse_mcif(
            synthetic(updates={"_atom_site_moment_crystalaxis_x": [value]}), "bad"
        )


@pytest.mark.parametrize(
    "operation",
    [
        "x,y,z,0",
        "x,y,z",
        "x+x,y,z,+1",
        "x+1/2+1/2,y,z,+1",
        "x+__import__,y,z,+1",
        "x+1/0,y,z,+1",
        "4x,y,z,+1",
    ],
)
def test_invalid_operation_grammar(operation):
    with pytest.raises(ValueError, match="MCIF_OPERATION"):
        parse_mcif(synthetic(ops=[operation]), "bad")


@pytest.mark.parametrize(
    "ops,match",
    [
        (["x,y,z,+1", "x,y,z,+1"], "MCIF_GROUP"),
        (["x,y,z+1/2,-1"], "MCIF_GROUP"),
        (["x,y,z,+1", "x,y,z+1/3,+1"], "MCIF_GROUP"),
        (["x,y,z,+1", "z,y,x,+1"], "MCIF_METRIC"),
    ],
)
def test_incomplete_or_metric_incompatible_operations(ops, match):
    with pytest.raises(ValueError, match=match):
        parse_mcif(synthetic(ops=ops), "bad")


def test_conflicting_moments_and_species_at_shared_site():
    with pytest.raises(ValueError, match="MCIF_MOMENT_CONFLICT"):
        parse_mcif(
            synthetic(
                rows=[("Fe1", "Fe", 0, 0, 0, 1, 0, 0)], ops=["x,y,z,+1", "-x,-y,-z,-1"]
            ),
            "bad",
        )
    with pytest.raises(ValueError, match="MCIF_SITE_COLLISION"):
        parse_mcif(
            synthetic(
                rows=[("Fe1", "Fe", 0, 0, 0, 1, 0, 0), ("O1", "O", 0, 0, 0, 0, 0, 0)]
            ),
            "bad",
        )


def test_missing_moment_or_label_mismatch():
    text = synthetic().decode()
    with pytest.raises(ValueError, match="MCIF_COLUMN"):
        parse_mcif(
            text.replace(
                "_atom_site_moment_crystalaxis_x", "_unrelated_column"
            ).encode(),
            "bad",
        )
    with pytest.raises(ValueError, match="MCIF_MOMENTS"):
        parse_mcif(synthetic(updates={"_atom_site_moment_label": ["unmatched"]}), "bad")


def test_explicit_occupancy_and_wyckoff():
    data = synthetic()
    # Separate one-row loops are legal CIF and allow optional site metadata.
    data += b"loop_\n_atom_site_occupancy\n1\nloop_\n_atom_site_Wyckoff_symbol\n2a\n"
    result = parse_mcif(data, "metadata")
    assert result.source_sites[0].occupancy_explicit
    assert result.source_sites[0].wyckoff_declared == "2a"
    for occupancy in [b"0.5", b"0", b"?", b"1.1"]:
        with pytest.raises(ValueError, match="MCIF_OCCUPANCY|MCIF_NUMBER"):
            parse_mcif(
                data.replace(
                    b"_atom_site_occupancy\n1\n",
                    b"_atom_site_occupancy\n" + occupancy + b"\n",
                ),
                "partial",
            )


def test_duplicate_tags_blocks_and_malformed_loop_rejected():
    for payload in [
        synthetic() + b"\n_cell_length_a 7\n",
        synthetic() + synthetic(),
        synthetic() + b"loop_\n_empty\n",
    ]:
        with pytest.raises(ValueError):
            parse_mcif(payload, "bad")
    with pytest.raises(ValueError, match="MCIF_DUPLICATE_TAG"):
        parse_mcif(synthetic() + b"\n_atom_site_moment.crystalaxis_x 0\n", "alias")


def test_dot_aliases_uncertainties_and_crlf():
    data = synthetic(
        updates={
            "_cell_length_a": "4.0(2)",
            "_atom_site_moment_crystalaxis_x": ["1.2(1)D+1"],
        }
    )
    data = data.replace(b"_atom_site_moment_", b"_atom_site_moment.").replace(
        b"\n", b"\r\n"
    )
    result = parse_mcif(data, "aliases")
    assert result.sites[0].moment_cartesian[0] == pytest.approx(12)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"_cell_angle_gamma": "180"}, "MCIF_CELL"),
        (
            {
                "_cell_angle_alpha": "170",
                "_cell_angle_beta": "10",
                "_cell_angle_gamma": "10",
            },
            "MCIF_CELL",
        ),
        ({"_cell_length_a": "0"}, "MCIF_CELL"),
        ({"_cell_volume": "1"}, "MCIF_VOLUME"),
        ({"_space_group_magn.number_BNS": "bad"}, "MCIF_BNS"),
        ({"_space_group_magn_ssg_name": "modulated"}, "MCIF_MODULATION"),
        ({"_atom_site_type_symbol": ["Fe2+"]}, "MCIF_SPECIES"),
    ],
)
def test_rejected_metadata(updates, match):
    with pytest.raises(ValueError, match=match):
        parse_mcif(synthetic(updates=updates), "bad")


def test_magnetic_operations_required_without_database_fallback():
    payload = synthetic().replace(b"_space_group_symop_magn_operation", b"_ordinary")
    with pytest.raises(ValueError, match="MCIF_COLUMN"):
        parse_mcif(payload + b"\n_space_group_magn.number_BNS 2.7\n", "bad")


def test_candidate_id_injection_and_mutated_mapping():
    with pytest.raises(ValueError, match="single-line"):
        parse_mcif(synthetic(), "name\n1\n")
    candidate = parse_mcif(synthetic(), "good")
    candidate.sites[0].poscar_index = 2
    with pytest.raises(ValueError, match="mapping"):
        map_moments(candidate, request())


def workflow_step(mode="noncollinear"):
    return {
        "id": "scf",
        "candidate_id": "candidate_01",
        "mcif": str(FIXTURES / PROVENANCE["files"][0]["file"]),
        "incar": {
            "goal": {"task": "static", "magnetic_mode": mode},
            "encut": 600,
            "moment_source": "mcif",
        },
        "kpoints": {"goal": {"task": "static", "magnetic_mode": mode}, "spacing": 0.25},
    }


@pytest.fixture
def potential_context(tmp_path):
    records = []
    for element in ["Fe", "O"]:
        payload = f"Synthetic test potential, NOT VASP data: {element}\n".encode()
        path = tmp_path / element / "POTCAR"
        path.parent.mkdir()
        path.write_bytes(payload)
        records.append(
            PotentialRecord(
                element=element,
                potential=element,
                family="PBE_54",
                titel=f"Synthetic {element}",
                relative_path=f"{element}/POTCAR",
                sha256=hashlib.sha256(payload).hexdigest(),
                enmax=400,
            )
        )
    return PotentialRegistry(
        root=str(tmp_path), family="PBE_54", records=records
    ), PotentialPolicy(family="PBE_54", preferred={})


@pytest.mark.parametrize("mode", ["collinear", "noncollinear", "soc"])
def test_workflow_from_mcif_and_manifest_mapping(mode, potential_context, tmp_path):
    step = workflow_step(mode)
    if mode == "collinear":
        step["incar"]["moment_mode"] = "collinear_projection"
    registry, policy = potential_context
    result = prepare_workflow(
        Workflow.model_validate({"workflow_id": "mcif", "steps": [step]}),
        registry,
        policy,
        tmp_path,
    )
    prepared = result.steps["scf"]
    assert "poscar_source" not in prepared.manifest
    assert (
        prepared.manifest["mcif_source"]["sha256"]
        == PROVENANCE["files"][0]["source_sha256"]
    )
    assert prepared.manifest["candidate"]["declared_bns"] == "2.7"
    assert (
        prepared.manifest["atom_mapping"][1]["mcif_origins"][0]["source_label"] == "Fe1"
    )
    assert (
        prepared.manifest["atom_mapping"][1]["mcif_origins"][0]["operation_id"] == "3"
    )
    assert prepared.incar.MAGMOM == prepared.manifest["moment_mapping"]["magmom"]
    assert prepared.incar.ENCUT == 600
    assert prepared.incar.LSORBIT is (mode == "soc")
    materialize_workflow(result, tmp_path / "bundle")
    assert inspect_bundle(tmp_path / "bundle")["integrity_valid"]


def test_mcif_downstream_charge_dependency(potential_context, tmp_path):
    scf = workflow_step()
    dos = workflow_step()
    dos["id"] = "dos"
    dos["incar"]["goal"]["task"] = dos["kpoints"]["goal"]["task"] = "dos"
    dos["dependencies"] = [{"artifact": "CHGCAR", "from_step": "scf"}]
    dos["kpoints"]["spacing"] = 0.15
    result = prepare_workflow(
        Workflow.model_validate({"workflow_id": "mcif-dos", "steps": [scf, dos]}),
        *potential_context,
        tmp_path,
    )
    assert (
        result.steps["dos"].manifest["status"]["input_state"] == "waiting_for_upstream"
    )


def test_mcif_workflow_rejects_dual_sources_and_manual_moments():
    for mutation in [
        lambda s: s.update(poscar="POSCAR"),
        lambda s: s.pop("mcif"),
        lambda s: s["incar"].update(magmom=[1, -1, 0, 0]),
        lambda s: s["incar"].pop("moment_source"),
    ]:
        step = workflow_step()
        mutation(step)
        with pytest.raises(ValueError):
            CalculationStep.model_validate(step)


def test_mcif_mae_scan_rejected_instead_of_silently_preserving_direction():
    step = workflow_step("soc")
    step["incar"]["goal"]["task"] = step["kpoints"]["goal"]["task"] = (
        "magnetic_anisotropy"
    )
    with pytest.raises(ValueError, match="mCIF scan templates"):
        Workflow.model_validate(
            {
                "workflow_id": "scan",
                "scans": [
                    {
                        "kind": "mae",
                        "base": step,
                        "axes": {"a": [0, 0, 1], "b": [1, 0, 0]},
                    }
                ],
            }
        )


def test_mcif_cli_show_errors_and_workflow_build(potential_context, tmp_path):
    runner = CliRunner()
    source = FIXTURES / PROVENANCE["files"][0]["file"]
    shown = runner.invoke(app, ["mcif", "show", str(source)])
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.output)["structure"]["counts"] == [2, 2]
    bad = tmp_path / "bad.mcif"
    bad.write_bytes(synthetic(updates={"_atom_site_moment_crystalaxis_x": ["?"]}))
    assert runner.invoke(app, ["mcif", "show", str(bad)]).exit_code == 2
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps({"workflow_id": "cli", "steps": [workflow_step()]})
    )
    for name, model in zip(["registry", "policy"], potential_context, strict=True):
        (tmp_path / f"{name}.json").write_text(model.model_dump_json())
    args = [
        "workflow",
        "build",
        "--request",
        str(request_path),
        "--registry",
        str(tmp_path / "registry.json"),
        "--policy",
        str(tmp_path / "policy.json"),
        "--output",
        str(tmp_path / "bundle"),
    ]
    built = runner.invoke(app, args)
    assert built.exit_code == 0, built.output
    assert runner.invoke(app, args).exit_code == 2


@pytest.mark.parametrize("axis", [[1e200, 0, 0], [1e-200, 0, 0]])
def test_saxis_normalized_without_overflow_or_underflow(axis):
    candidate = parse_mcif(synthetic(), "axis")
    incar, mapping = map_moments(candidate, request("soc", saxis=axis))
    assert incar.saxis == mapping.saxis == [1, 0, 0]
    np.testing.assert_allclose(incar.magmom, [-3, 2, 1], atol=1e-8)


def test_zero_saxis_and_all_zero_candidate():
    candidate = parse_mcif(synthetic(rows=[("Fe1", "Fe", 0, 0, 0, 0, 0, 0)]), "zero")
    assert candidate.collinear_axis_cartesian is None
    with pytest.raises(ValueError, match="MCIF_SAXIS"):
        map_moments(candidate, request(saxis=[0, 0, 0]))
    incar, _ = map_moments(
        candidate, request("collinear", moment_mode="collinear_projection")
    )
    assert incar.magmom == [0]


def test_moments_matched_by_labels_not_row_position():
    rows = [("Fe1", "Fe", 0, 0, 0, 1, 0, 0), ("Fe2", "Fe", 0.5, 0.5, 0.5, 0, 1, 0)]
    result = parse_mcif(
        synthetic(rows=rows, updates={"_atom_site_moment_label": ["Fe2", "Fe1"]}),
        "labels",
    )
    assert result.sites[0].moment_cartesian == [0, 1, 0]
    assert result.sites[1].moment_cartesian == [1, 0, 0]


def test_nonzero_oxygen_on_real_forbidden_site_is_not_silently_projected():
    payload = (FIXTURES / PROVENANCE["files"][0]["file"]).read_bytes()
    payload = payload.replace(
        b"O1 0.000000 0.000000 0.000000 0.000000",
        b"O1 0.000000 0.000000 1.000000 1.000000",
    )
    with pytest.raises(ValueError, match="MCIF_MOMENT_CONFLICT"):
        parse_mcif(payload, "bad-oxygen")


def test_scalar_moment_uses_nonorthogonal_metric():
    payload = (FIXTURES / PROVENANCE["files"][0]["file"]).read_bytes()
    with pytest.raises(ValueError, match="MCIF_MAGNITUDE"):
        parse_mcif(payload.replace(b"1.417774", b"9.000000"), "wrong-norm")


def test_incomplete_centering_and_nontranslation_centering():
    data = synthetic() + b"loop_\n_space_group_symop_magn_centering.id\n1\n"
    with pytest.raises(ValueError, match="MCIF_CENTERING"):
        parse_mcif(data, "incomplete")
    with pytest.raises(ValueError, match="MCIF_CENTERING"):
        parse_mcif(synthetic(centers=["x,y,z,+1", "-x,-y,z,+1"]), "bad-center")


def test_closed_fractional_translations():
    result = parse_mcif(
        synthetic(ops=["x,y,z,+1", "x+1/3,y,z,+1", "x+2/3,y,z,+1"]), "thirds"
    )
    assert len(result.sites) == 3


def test_invalid_mcif_workflow_leaves_no_package(potential_context, tmp_path):
    bad = tmp_path / "unknown.mcif"
    bad.write_bytes(synthetic(updates={"_atom_site_moment_crystalaxis_x": ["?"]}))
    step = workflow_step()
    step["mcif"] = str(bad)
    with pytest.raises(ValueError, match="MCIF_NUMBER"):
        result = prepare_workflow(
            Workflow.model_validate({"workflow_id": "bad", "steps": [step]}),
            *potential_context,
            tmp_path,
        )
        materialize_workflow(result, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_checked_in_mcif_integration_workflow(potential_context, tmp_path):
    root = ROOT / "config/workflows"
    data = Workflow.model_validate_json((root / "FeO.generated.mcif.json").read_bytes())
    result = prepare_workflow(data, *potential_context, root)
    assert len(result.steps) == 3
    assert all(s.structure.counts == [2, 2] for s in result.steps.values())
    for step in result.steps.values():
        assert step.incar.LSORBIT is False
        assert step.incar.LNONCOLLINEAR is True
        assert step.incar.LDAU is None
    materialize_workflow(result, tmp_path / "generated")
    assert inspect_bundle(tmp_path / "generated")["steps"] == 3


def test_plan_workflow_from_mcif_auto_mode_and_dos(potential_context, tmp_path):
    source = FIXTURES / PROVENANCE["files"][0]["file"]
    request = tmp_path / "request.json"
    report = plan_workflow_from_mcif(
        [source],
        request,
        workflow_id="FeO-from-mcif",
        magnetic_mode="auto",
        include_dos=True,
        notes="unit test only",
    )
    assert report["preview"][0]["magnetic_mode"] == "collinear"
    assert report["preview"][0]["species_order"] == ["Fe", "O"]
    assert "magmom" in report["preview"][0]
    written = Workflow.model_validate_json(request.read_bytes())
    assert [step.id for step in written.steps] == ["scf", "dos"]
    assert written.steps[0].incar.moment_source == "mcif"
    assert "magmom" not in written.steps[0].incar.model_dump(exclude_none=True)
    result = prepare_workflow(written, *potential_context, request.parent)
    materialize_workflow(result, tmp_path / "bundle")
    inspect = inspect_bundle(tmp_path / "bundle")
    assert inspect["integrity_valid"]
    assert inspect["waiting_for_upstream"] == ["dos"]
    incar = (tmp_path / "bundle" / "scf" / "INCAR").read_text(encoding="utf-8")
    assert "ISPIN = 2" in incar
    assert "LNONCOLLINEAR" not in incar or "LNONCOLLINEAR = .FALSE." in incar
    assert "MAGMOM" in incar
    with pytest.raises(FileExistsError):
        plan_workflow_from_mcif([source], request)


def test_plan_workflow_from_mcif_refuses_collinear_and_band_guess(tmp_path):
    source = tmp_path / "canted.mcif"
    source.write_bytes(
        synthetic(rows=[("Fe1", "Fe", 0, 0, 0, 1, 0, 0), ("Fe2", "Fe", 0.5, 0.5, 0.5, 0, 1, 0)])
    )
    with pytest.raises(ValueError, match="MCIF_NONCOLLINEAR"):
        plan_workflow_from_mcif(
            [source],
            tmp_path / "unused.json",
            magnetic_mode="collinear",
        )
    with pytest.raises(ValueError, match="band"):
        plan_workflow_from_mcif(
            [FIXTURES / PROVENANCE["files"][0]["file"]],
            tmp_path / "unused-band.json",
            task="band",
        )
