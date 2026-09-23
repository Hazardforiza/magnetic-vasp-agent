"""Strict mCIF boundary around pymatgen parsing and magnetic operations.

Coordinates use fractional lattice vectors; moments use unit crystal axes.
Magnetic rotations therefore act in Cartesian space, not directly on the
crystal-axis moment triples for a general nonorthogonal cell.
"""

from __future__ import annotations

import hashlib
import math
import re
import warnings
from pathlib import Path

import numpy as np
from pymatgen.core import Element, Lattice
from pymatgen.core.operations import MagSymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.io.cif import CifBlock, CifFile, str2float

from magnetic_vasp_agent.incar_models import IncarRecipeRequest
from magnetic_vasp_agent.mcif_models import (
    CandidateStructure,
    MagneticSite,
    McifIncarRequest,
    MomentMapping,
    SiteOrigin,
)
from magnetic_vasp_agent.models import Structure

SITE_TOLERANCE = 1e-5  # Angstrom, only for symmetry-image merging.
MOMENT_TOLERANCE = 1e-6  # Bohr magnetons, absolute Cartesian residual.
MAX_OPERATIONS = 384
MAX_EXPANDED_SITES = 4096


def _tag(value: str) -> str:
    return value.lower().replace(".", "_")


def _block(text: str) -> tuple[str, dict]:
    if len(text) > 5_000_000:
        raise ValueError("MCIF_SIZE: input exceeds 5 MB")
    # Use pymatgen's tokenizer to audit quoted values correctly before its
    # permissive dictionary parser can overwrite duplicate data names/blocks.
    tokens = CifBlock._process_string(text)
    keys = [_tag(t[0]) for t in tokens if t[0].startswith("_")]
    if len(set(keys)) != len(keys):
        raise ValueError("MCIF_DUPLICATE_TAG: duplicate or aliased CIF data name")
    if sum(t[0].lower().startswith("data_") for t in tokens) != 1:
        raise ValueError("MCIF_BLOCKS: exactly one structure data block is required")
    if any(t[0].lower().startswith(("save_", "global_", "stop_")) for t in tokens):
        raise ValueError("MCIF_LAYOUT: save/global/stop frames are not supported")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parsed = CifFile.from_str(text)
        if caught:
            raise ValueError(
                "MCIF_SYNTAX: " + "; ".join(str(w.message) for w in caught)
            )
    except (ZeroDivisionError, IndexError, KeyError) as exc:
        raise ValueError("MCIF_SYNTAX: malformed CIF loop") from exc
    if len(parsed.data) != 1:
        raise ValueError("MCIF_BLOCKS: missing supported structure block")
    block = next(iter(parsed.data.values()))
    return block.header, {_tag(k): v for k, v in block.data.items()}


def _number(value: str, field: str) -> float:
    if not isinstance(value, str) or not re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\(\d+\))?(?:[eEdD][+-]?\d+)?", value
    ):
        raise ValueError(f"MCIF_NUMBER: {field} needs a finite number, not {value!r}")
    value = value.replace("D", "e").replace("d", "e")
    # CIF standard uncertainties precede the exponent; pymatgen's converter
    # truncates at the uncertainty, so remove only that parenthesized suffix.
    number = str2float(re.sub(r"\(\d+\)", "", value))
    if not math.isfinite(number):
        raise ValueError(f"MCIF_NUMBER: {field} is nonfinite")
    return number


def _column(data: dict, name: str, count: int | None = None) -> list[str]:
    value = data.get(name)
    if (
        not isinstance(value, list)
        or not value
        or (count is not None and len(value) != count)
    ):
        raise ValueError(f"MCIF_COLUMN: missing or inconsistent loop column {name}")
    return value


def _optional_scalar(data: dict, name: str) -> str | None:
    value = data.get(name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"MCIF_COLUMN: {name} must be a scalar")
    return value


def _operation(text: str) -> MagSymmOp:
    fields = text.lower().replace(" ", "").split(",")
    if len(fields) != 4 or fields[-1] not in {"1", "+1", "-1"}:
        raise ValueError("MCIF_OPERATION: expected x,y,z,+1/-1")
    term = r"(?:[1-3]?[xyz]|\d+(?:/\d+|\.\d+)?)"
    for expression in fields[:3]:
        if not re.fullmatch(rf"[+-]?{term}(?:[+-]{term})*", expression):
            raise ValueError(
                f"MCIF_OPERATION: unsupported coordinate expression {expression!r}"
            )
        pieces = re.findall(rf"[+-]?{term}", expression)
        variables = [p[-1] for p in pieces if p[-1] in "xyz"]
        if len(variables) != len(set(variables)) or len(pieces) - len(variables) > 1:
            raise ValueError("MCIF_OPERATION: repeated variables or translation terms")
    try:
        op = MagSymmOp.from_xyzt_str(",".join(fields))
    except (ValueError, ZeroDivisionError, RuntimeError) as exc:
        raise ValueError(f"MCIF_OPERATION: invalid operation {text!r}") from exc
    if not np.isfinite(op.affine_matrix).all() or not np.isclose(
        abs(np.linalg.det(op.rotation_matrix)), 1, atol=1e-10
    ):
        raise ValueError("MCIF_OPERATION: rotation must be finite and unimodular")
    return op


def _op_key(op: MagSymmOp) -> tuple:
    translation = np.round(np.mod(op.translation_vector, 1), 8)
    translation[translation == 1] = 0
    return (
        *np.rint(op.rotation_matrix).astype(int).ravel(),
        *translation,
        op.time_reversal,
    )


def _operations(data: dict, lattice: Lattice) -> tuple[list[MagSymmOp], list[dict]]:
    prefix = "_space_group_symop_magn_operation"
    strings = _column(data, prefix + "_xyz")
    ids = (
        _column(data, prefix + "_id", len(strings))
        if prefix + "_id" in data
        else [str(i + 1) for i in range(len(strings))]
    )
    cp = "_space_group_symop_magn_centering"
    if cp + "_id" in data and cp + "_xyz" not in data:
        raise ValueError("MCIF_CENTERING: centering IDs require explicit operations")
    centers = _column(data, cp + "_xyz") if cp + "_xyz" in data else ["x,y,z,+1"]
    cids = (
        _column(data, cp + "_id", len(centers))
        if cp + "_id" in data
        else [str(i + 1) for i in range(len(centers))]
    )
    if len(strings) * len(centers) > MAX_OPERATIONS:
        raise ValueError("MCIF_SIZE: too many magnetic operations")
    if len(set(ids)) != len(ids) or len(set(cids)) != len(cids):
        raise ValueError("MCIF_OPERATION: duplicate operation IDs")
    operations, metadata = [], []
    basis = lattice.matrix.T
    for oid, string in zip(ids, strings, strict=True):
        op = _operation(string)
        cart_rotation = basis @ op.rotation_matrix @ np.linalg.inv(basis)
        if not np.allclose(
            cart_rotation.T @ cart_rotation, np.eye(3), atol=2e-5, rtol=0
        ):
            raise ValueError(
                "MCIF_METRIC: magnetic rotation does not preserve the cell metric"
            )
        for cid, center_string in zip(cids, centers, strict=True):
            center = _operation(center_string)
            if not np.array_equal(center.rotation_matrix, np.eye(3)):
                raise ValueError(
                    "MCIF_CENTERING: centering operations must be pure translations with optional time reversal"
                )
            combined = MagSymmOp.from_rotation_and_translation_and_time_reversal(
                op.rotation_matrix,
                np.mod(op.translation_vector + center.translation_vector, 1),
                op.time_reversal * center.time_reversal,
            )
            operations.append(combined)
            metadata.append(
                {
                    "operation_id": oid,
                    "centering_id": cid,
                    "operation": string,
                    "centering": center_string,
                    "combined": combined.as_xyzt_str(),
                }
            )
    keys = {_op_key(op) for op in operations}
    if len(keys) != len(operations) or _op_key(_operation("x,y,z,+1")) not in keys:
        raise ValueError(
            "MCIF_GROUP: magnetic operations must be unique and include identity"
        )
    for a in operations:
        for b in operations:
            product = MagSymmOp.from_rotation_and_translation_and_time_reversal(
                a.rotation_matrix @ b.rotation_matrix,
                a.rotation_matrix @ b.translation_vector + a.translation_vector,
                a.time_reversal * b.time_reversal,
            )
            if _op_key(product) not in keys:
                raise ValueError(
                    "MCIF_GROUP: magnetic operations are not closed under composition"
                )
    return operations, metadata


def _axis(moments: np.ndarray) -> tuple[bool, list[float] | None]:
    norms = np.linalg.norm(moments, axis=1)
    if np.max(norms) <= MOMENT_TOLERANCE:
        return True, None
    axis = moments[int(np.argmax(norms))] / np.max(norms)
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis *= -1
    residual = moments - np.outer(moments @ axis, axis)
    return bool(
        np.max(np.linalg.norm(residual, axis=1)) <= MOMENT_TOLERANCE
    ), axis.tolist()


def parse_mcif(payload: bytes, candidate_id: str) -> CandidateStructure:
    header, data = _block(payload.decode("utf-8-sig"))
    if any(
        "_ssg" in k or "_modulation" in k or "_wave_vector" in k or "fourier" in k
        for k in data
    ):
        raise ValueError(
            "MCIF_MODULATION: only commensurate, explicit-cell magnetic structures are supported"
        )
    parameters = [
        _number(data.get(f"_cell_{kind}_{axis}"), f"cell {axis}")
        for kind, axes in [
            ("length", ("a", "b", "c")),
            ("angle", ("alpha", "beta", "gamma")),
        ]
        for axis in axes
    ]
    if any(x <= 0 for x in parameters[:3]) or any(
        not 0 < x < 180 for x in parameters[3:]
    ):
        raise ValueError("MCIF_CELL: invalid lattice lengths or angles")
    ca, cb, cg = np.cos(np.radians(parameters[3:]))
    if 1 + 2 * ca * cb * cg - ca * ca - cb * cb - cg * cg <= 1e-12:
        raise ValueError("MCIF_CELL: singular or impossible lattice angles")
    lattice = Lattice.from_parameters(*parameters)
    if not math.isfinite(lattice.volume) or lattice.volume <= 1e-10:
        raise ValueError("MCIF_CELL: finite positive volume required")
    operations, operation_metadata = _operations(data, lattice)
    diagnostics = []
    if "_cell_volume" in data:
        declared_volume = _number(data["_cell_volume"], "cell volume")
        if not math.isclose(
            declared_volume, lattice.volume, rel_tol=1e-5, abs_tol=1e-5
        ):
            raise ValueError(
                "MCIF_VOLUME: declared volume disagrees with lattice parameters"
            )
    labels = _column(data, "_atom_site_label")
    if (
        len(labels) > MAX_EXPANDED_SITES
        or len(set(labels)) != len(labels)
        or any(not s or s in {"?", "."} for s in labels)
    ):
        raise ValueError(
            "MCIF_LABELS: site labels must be unique, nonempty, and within the site limit"
        )
    elements = _column(data, "_atom_site_type_symbol", len(labels))
    for element in elements:
        if not Element.is_valid_symbol(element):
            raise ValueError(
                f"MCIF_SPECIES: explicit pure element symbols required, got {element!r}"
            )
    coords = [_column(data, f"_atom_site_fract_{a}", len(labels)) for a in "xyz"]
    moment_labels = _column(data, "_atom_site_moment_label")
    if len(set(moment_labels)) != len(moment_labels) or set(moment_labels) != set(
        labels
    ):
        raise ValueError(
            "MCIF_MOMENTS: every site needs one explicit moment, including zero moments"
        )
    moment_cols = [
        _column(data, f"_atom_site_moment_crystalaxis_{a}", len(moment_labels))
        for a in "xyz"
    ]
    moment_dict = {
        label: [_number(col[i], f"moment {label}") for col in moment_cols]
        for i, label in enumerate(moment_labels)
    }
    if "_atom_site_occupancy" in data:
        occupancy = [
            _number(x, "occupancy")
            for x in _column(data, "_atom_site_occupancy", len(labels))
        ]
        if any(x != 1 for x in occupancy):
            raise ValueError(
                "MCIF_OCCUPANCY: partial, mixed, or vacant sites require explicit structural enumeration"
            )
    else:
        occupancy = [1.0] * len(labels)
        diagnostics.append(
            "OCCUPANCY_DEFAULT_ONE: occupancy omitted; full occupancy assumed and recorded"
        )
    wyckoff_key = next(
        (
            k
            for k in ("_atom_site_wyckoff_symbol", "_atom_site_wyckoff_label")
            if k in data
        ),
        None,
    )
    wyckoff = (
        _column(data, wyckoff_key, len(labels)) if wyckoff_key else [None] * len(labels)
    )
    if wyckoff_key is None:
        diagnostics.append(
            "WYCKOFF_NOT_PROVIDED: no Wyckoff labels inferred from filenames or source comments"
        )
    if any(
        k.startswith("_symmetry_equiv_pos")
        or k.startswith("_space_group_symop_operation")
        for k in data
    ):
        diagnostics.append(
            "MAGNETIC_OPERATIONS_USED: ordinary symmetry loop not used for magnetic expansion"
        )
    if "_atom_site_moment" in data:
        magnitudes = _column(data, "_atom_site_moment", len(labels))
    else:
        magnitudes = None
    basis = lattice.matrix.T
    cart_ops = [
        MagSymmOp.from_rotation_and_translation_and_time_reversal(
            basis @ op.rotation_matrix @ np.linalg.inv(basis),
            [0, 0, 0],
            op.time_reversal,
        )
        for op in operations
    ]
    expanded, source_sites = [], []
    for row, (label, element) in enumerate(zip(labels, elements, strict=True)):
        position = np.array([_number(col[row], f"position {label}") for col in coords])
        moment = Magmom.from_moment_relative_to_crystal_axes(
            moment_dict[label], lattice
        )
        if magnitudes is not None and not math.isclose(
            _number(magnitudes[row], "moment magnitude"),
            np.linalg.norm(moment.global_moment),
            abs_tol=2e-5,
            rel_tol=1e-5,
        ):
            raise ValueError(
                "MCIF_MAGNITUDE: scalar moment disagrees with Cartesian norm"
            )
        source_sites.append(
            {
                "label": label,
                "row": row + 1,
                "species": element,
                "fractional": position.tolist(),
                "moment_crystalaxis": moment_dict[label],
                "moment_cartesian": moment.global_moment.tolist(),
                "occupancy": occupancy[row],
                "occupancy_explicit": "_atom_site_occupancy" in data,
                "wyckoff_declared": wyckoff[row],
            }
        )
        for op, cart_op, metadata in zip(
            operations, cart_ops, operation_metadata, strict=True
        ):
            fractional = np.mod(op.operate(position), 1)
            transformed = cart_op.operate_magmom(moment).global_moment
            origin = SiteOrigin(
                source_label=label,
                source_row=row + 1,
                operation_id=metadata["operation_id"],
                centering_id=metadata["centering_id"],
            )
            matches = [
                s
                for s in expanded
                if lattice.get_distance_and_image(s.fractional, fractional)[0]
                < SITE_TOLERANCE
            ]
            if matches:
                site = matches[0]
                if site.species != element:
                    raise ValueError(
                        "MCIF_SITE_COLLISION: different species occupy the same expanded site"
                    )
                if (
                    np.linalg.norm(np.array(site.moment_cartesian) - transformed)
                    > MOMENT_TOLERANCE
                ):
                    raise ValueError(
                        "MCIF_MOMENT_CONFLICT: symmetry-equivalent site has incompatible magnetic moments"
                    )
                site.origins.append(origin)
            else:
                if len(expanded) >= MAX_EXPANDED_SITES:
                    raise ValueError("MCIF_SIZE: expanded structure exceeds site limit")
                expanded.append(
                    MagneticSite(
                        poscar_index=0,
                        species=element,
                        fractional=fractional.tolist(),
                        moment_cartesian=transformed.tolist(),
                        origins=[origin],
                    )
                )
    species_order = list(dict.fromkeys(elements))
    expanded.sort(key=lambda s: species_order.index(s.species))
    for index, site in enumerate(expanded):
        site.poscar_index = index + 1
    structure = Structure(
        comment=candidate_id,
        scale=1.0,
        lattice=lattice.matrix.tolist(),
        species_order=species_order,
        counts=[sum(s.species == e for s in expanded) for e in species_order],
        coordinate_mode="direct",
        positions=[s.fractional for s in expanded],
    )
    moments = np.array([s.moment_cartesian for s in expanded])
    collinear, axis = _axis(moments)
    bns = _optional_scalar(data, "_space_group_magn_number_bns")
    if bns is not None and not re.fullmatch(r"\d+\.\d+", bns):
        raise ValueError("MCIF_BNS: BNS number must have the form n.m")
    return CandidateStructure(
        candidate_id=candidate_id,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        source_block=header,
        declared_bns=bns,
        declared_space_group=_optional_scalar(data, "_symmetry_space_group_name_h-m"),
        structure=structure,
        source_sites=source_sites,
        magnetic_operations=operation_metadata,
        sites=expanded,
        collinear=collinear,
        collinear_axis_cartesian=axis if collinear else None,
        total_moment_cartesian=moments.sum(axis=0).tolist(),
        diagnostics=diagnostics,
    )


def read_mcif(path: Path, candidate_id: str | None = None) -> CandidateStructure:
    return parse_mcif(path.read_bytes(), candidate_id or path.stem)


def map_moments(
    candidate: CandidateStructure, request: McifIncarRequest
) -> tuple[IncarRecipeRequest, MomentMapping]:
    candidate = CandidateStructure.model_validate_json(candidate.model_dump_json())
    request = McifIncarRequest.model_validate_json(request.model_dump_json())
    moments = np.array([s.moment_cartesian for s in candidate.sites])
    axis = np.array(request.saxis, dtype=float)
    if not math.isfinite(math.hypot(*axis)) or math.hypot(*axis) == 0:
        raise ValueError("MCIF_SAXIS: finite nonzero spin axis required")
    axis /= math.hypot(*axis)
    projection = None
    if request.goal.magnetic_mode.value == "collinear":
        if request.moment_mode != "collinear_projection" or request.saxis != [0, 0, 1]:
            raise ValueError(
                "MCIF_PROJECTION: collinear calculation requires explicit collinear_projection and default SAXIS"
            )
        collinear, projection = _axis(moments)
        if not collinear:
            raise ValueError(
                "MCIF_NONCOLLINEAR: refusing to discard transverse magnetic components"
            )
        projection = projection or [0.0, 0.0, 1.0]
        values = moments @ np.array(projection)
        error = float(
            np.max(np.linalg.norm(moments - np.outer(values, projection), axis=1))
        )
    else:
        if request.moment_mode != "preserve_cartesian":
            raise ValueError(
                "MCIF_PROJECTION: spinor/SOC calculations must preserve Cartesian moments"
            )
        vectors = np.array([Magmom(m).get_moment(saxis=axis.copy()) for m in moments])
        restored = np.array(
            [Magmom(v, saxis=axis.copy()).global_moment for v in vectors]
        )
        error = float(np.max(np.linalg.norm(restored - moments, axis=1)))
        values = vectors.ravel()
    if error > MOMENT_TOLERANCE:
        raise ValueError(
            "MCIF_ROUNDTRIP: moment coordinate conversion exceeded tolerance"
        )
    mapping = MomentMapping(
        mode=request.moment_mode,
        magmom=values.tolist(),
        saxis=axis.tolist(),
        projection_axis_cartesian=projection,
        orientation_preserved=projection is None,
        maximum_roundtrip_error=error,
    )
    settings = request.model_dump(exclude={"moment_source", "moment_mode"})
    settings["saxis"] = axis.tolist()
    return IncarRecipeRequest(**settings, magmom=mapping.magmom), mapping
