"""Deterministic parsing of VASP output files.

Own parsers (in the spirit of the input parsers) keep results auditable and
provenance-stamped. ``vasprun.xml`` is read with streaming ``iterparse`` because
real files can reach hundreds of MB; only compact summaries are kept, never full
DOS or eigenvalue arrays. OUTCAR carries the scientifically critical magnetic
quantities (per-ion local moments), so it is the primary source; OSZICAR and
vasprun.xml serve as cross-checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from magnetic_vasp_agent.postprocess_models import (
    CalculationResult,
    CrossCheck,
    DetectedMagneticMode,
    DosSummary,
    IonMagnetization,
    OszicarSummary,
    OutcarParameters,
    OutcarSummary,
    VasprunSummary,
)

_NUM = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"

# Energy agreement tolerances (eV) and magnetic/Fermi tolerances.
_ENERGY_TOL = 1.0e-2
_MAG_TOL = 5.0e-2
_FERMI_TOL = 1.0e-1


class VaspOutputError(ValueError):
    """Raised when an output file cannot be parsed into a typed result."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ffloat(token: str) -> float | None:
    """Parse a Fortran-style float; return None on overflow markers like ``***``."""

    token = token.strip()
    if not token or "*" in token:
        return None
    try:
        return float(token.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# OSZICAR                                                                       #
# --------------------------------------------------------------------------- #

_OSZICAR_STEP = re.compile(
    rf"^\s*[A-Za-z]+:\s*(\d+)\s+({_NUM})\s+({_NUM})"
)
_OSZICAR_IONIC = re.compile(
    rf"^\s*(\d+)\s+F=\s*({_NUM})\s+E0=\s*({_NUM})\s+d E\s*=\s*({_NUM})"
    rf"(?:\s+mag=\s*(.*))?$"
)


def parse_oszicar(text: str, *, source_sha256: str) -> OszicarSummary:
    ionic_steps = 0
    electronic_total = 0
    electronic_since_ionic = 0
    electronic_last = 0
    final_f: float | None = None
    final_e0: float | None = None
    final_de: float | None = None
    total_mag: float | None = None
    mag_vector: list[float] | None = None

    for line in text.splitlines():
        step = _OSZICAR_STEP.match(line)
        if step:
            electronic_total += 1
            electronic_since_ionic += 1
            continue
        ionic = _OSZICAR_IONIC.match(line)
        if ionic:
            ionic_steps += 1
            final_f = _ffloat(ionic.group(2))
            final_e0 = _ffloat(ionic.group(3))
            final_de = _ffloat(ionic.group(4))
            electronic_last = electronic_since_ionic
            electronic_since_ionic = 0
            mag_field = ionic.group(5)
            total_mag, mag_vector = _parse_mag_field(mag_field)

    if ionic_steps == 0:
        # Incomplete run (no ionic summary line yet): report what we have.
        electronic_last = electronic_since_ionic

    return OszicarSummary(
        ionic_steps=ionic_steps,
        electronic_steps_total=electronic_total,
        electronic_steps_last=electronic_last,
        final_free_energy=final_f,
        final_e0=final_e0,
        final_delta_e=final_de,
        total_magnetization=total_mag,
        magnetization_vector=mag_vector,
        source_sha256=source_sha256,
    )


def _parse_mag_field(mag_field: str | None) -> tuple[float | None, list[float] | None]:
    if not mag_field:
        return None, None
    values = [v for v in (_ffloat(tok) for tok in mag_field.split()) if v is not None]
    if len(values) >= 3:
        return None, values[:3]
    if len(values) == 1:
        return values[0], None
    return None, None


# --------------------------------------------------------------------------- #
# OUTCAR                                                                        #
# --------------------------------------------------------------------------- #

_ORBITAL_LABELS = ["s", "p", "d", "f", "g"]


def _search_last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text)
    return _ffloat(matches[-1]) if matches else None


def _outcar_parameters(text: str) -> OutcarParameters:
    def _int(pattern: str, default: int | None = None) -> int:
        match = re.search(pattern, text)
        if match is None:
            if default is None:
                raise VaspOutputError(f"OUTCAR missing required field: {pattern!r}")
            return default
        return int(match.group(1))

    def _flag(pattern: str) -> bool:
        match = re.search(pattern, text)
        return bool(match) and match.group(1).upper().startswith("T")

    ispin = _int(r"ISPIN\s*=\s*(\d+)", default=1)
    lnoncollinear = _flag(r"LNONCOLLINEAR\s*=\s*([TF])")
    lsorbit = _flag(r"LSORBIT\s*=\s*([TF])")
    nelm = _int(r"NELM\s*=\s*(\d+)", default=60)
    nsw = _int(r"NSW\s*=\s*(\d+)", default=0)
    nions = _int(r"NIONS\s*=\s*(\d+)")
    nelect = _search_last_float(rf"NELECT\s*=\s*({_NUM})", text)

    return OutcarParameters(
        ispin=ispin,
        lnoncollinear=lnoncollinear,
        lsorbit=lsorbit,
        nelm=nelm,
        nsw=nsw,
        nions=nions,
        nelect=nelect,
    )


def _magnetic_mode(params: OutcarParameters) -> DetectedMagneticMode:
    if params.lsorbit:
        return DetectedMagneticMode.SOC
    if params.lnoncollinear:
        return DetectedMagneticMode.NONCOLLINEAR
    if params.ispin == 2:
        return DetectedMagneticMode.COLLINEAR
    return DetectedMagneticMode.NONMAGNETIC


def _parse_magnetization_block(lines: list[str], start: int) -> list[list[float]]:
    """Parse rows of a ``magnetization (x)`` block into [ion, ...orbitals, tot]."""

    rows: list[list[float]] = []
    index = start + 1
    seen_header = False
    while index < len(lines):
        raw = lines[index].strip()
        index += 1
        if not raw:
            if seen_header and rows:
                break
            continue
        if raw.startswith("#"):  # "# of ion   s   p   d   tot"
            seen_header = True
            continue
        if set(raw) <= {"-"}:  # dashed separator
            if rows:  # the closing separator before the "tot" summary line
                break
            continue
        tokens = raw.split()
        if not tokens[0].lstrip("-").isdigit():
            break
        values = [_ffloat(tok) for tok in tokens]
        if any(v is None for v in values) or len(values) < 2:
            break
        rows.append(values)  # [ion_index, orb1, ..., tot]
    return rows


def _last_block_starts(lines: list[str], needle: str) -> int | None:
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip().startswith(needle):
            return index
    return None


def _ion_moments_collinear(lines: list[str]) -> tuple[list[IonMagnetization], float | None]:
    start = _last_block_starts(lines, "magnetization (x)")
    if start is None:
        return [], None
    rows = _parse_magnetization_block(lines, start)
    moments: list[IonMagnetization] = []
    for row in rows:
        idx = int(row[0])
        orbitals = row[1:-1]
        tot = row[-1]
        labels = _ORBITAL_LABELS[: len(orbitals)]
        moments.append(
            IonMagnetization(
                index=idx,
                moment=tot,
                orbital_moments={
                    label: value for label, value in zip(labels, orbitals, strict=True)
                }
                or None,
            )
        )
    total = sum(m.moment for m in moments) if moments else None
    return moments, total


def _ion_moments_noncollinear(
    lines: list[str],
) -> tuple[list[IonMagnetization], list[float] | None]:
    components: dict[str, list[list[float]]] = {}
    for axis in ("x", "y", "z"):
        start = _last_block_starts(lines, f"magnetization ({axis})")
        if start is None:
            continue
        components[axis] = _parse_magnetization_block(lines, start)
    if "x" not in components:
        return [], None

    num_ions = len(components["x"])
    moments: list[IonMagnetization] = []
    totals = [0.0, 0.0, 0.0]
    for i in range(num_ions):
        vector = []
        for axis in ("x", "y", "z"):
            rows = components.get(axis)
            vector.append(rows[i][-1] if rows and i < len(rows) else 0.0)
        idx = int(components["x"][i][0])
        for a in range(3):
            totals[a] += vector[a]
        moments.append(
            IonMagnetization(
                index=idx,
                moment=math.sqrt(sum(v * v for v in vector)),
                vector=vector,
            )
        )
    return moments, totals


def parse_outcar(text: str, *, source_sha256: str) -> OutcarSummary:
    params = _outcar_parameters(text)
    mode = _magnetic_mode(params)
    lines = text.splitlines()

    electronic_converged = bool(
        re.search(
            r"(?:aborting loop because EDIFF is reached|reached required accuracy)",
            text,
            re.IGNORECASE,
        )
    )
    free_energy = _search_last_float(rf"free\s+energy\s+TOTEN\s*=\s*({_NUM})", text)
    energy_wo = _search_last_float(rf"energy\s+without\s+entropy\s*=\s*({_NUM})", text)
    energy_sigma0 = _search_last_float(rf"energy\(sigma->0\)\s*=\s*({_NUM})", text)
    fermi = _search_last_float(rf"E-fermi\s*:\s*({_NUM})", text)
    if fermi is None:
        fermi = _search_last_float(rf"Fermi energy:\s*({_NUM})", text)

    total_vector: list[float] | None = None
    if mode in (DetectedMagneticMode.NONCOLLINEAR, DetectedMagneticMode.SOC):
        moments, total_vector = _ion_moments_noncollinear(lines)
        total_scalar = None
    else:
        moments, total_scalar = _ion_moments_collinear(lines)

    return OutcarSummary(
        parameters=params,
        magnetic_mode=mode,
        electronic_converged=electronic_converged,
        free_energy_toten=free_energy,
        energy_without_entropy=energy_wo,
        energy_sigma0=energy_sigma0,
        fermi_energy=fermi,
        ion_magnetizations=moments,
        total_magnetization=total_scalar,
        total_magnetization_vector=total_vector,
        max_force=_max_force(lines),
        source_sha256=source_sha256,
    )


def _max_force(lines: list[str]) -> float | None:
    start = _last_block_starts(lines, "POSITION")
    if start is None or "TOTAL-FORCE" not in lines[start]:
        # "POSITION ... TOTAL-FORCE (eV/Angst)" is a single header line.
        start = _last_block_starts(lines, "TOTAL-FORCE")
    if start is None:
        return None
    max_force: float | None = None
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if set(stripped) <= {"-"} and stripped:
            if max_force is not None:
                break
            continue
        tokens = stripped.split()
        if len(tokens) < 6:
            break
        values = [_ffloat(tok) for tok in tokens[:6]]
        if any(v is None for v in values):
            break
        fx, fy, fz = values[3], values[4], values[5]
        magnitude = math.sqrt(fx * fx + fy * fy + fz * fz)
        max_force = magnitude if max_force is None else max(max_force, magnitude)
    return max_force


# --------------------------------------------------------------------------- #
# vasprun.xml (streaming)                                                       #
# --------------------------------------------------------------------------- #


def parse_vasprun(path: Path, *, source_sha256: str) -> VasprunSummary:
    version: str | None = None
    ispin = 1
    noncollinear = False
    soc = False
    nelm = 60
    nsw = 0
    calc_count = 0
    last_scstep_count = 0
    scstep_in_calc = 0
    final_free: float | None = None
    final_sigma0: float | None = None
    dos: DosSummary | None = None

    try:
        context = ET.iterparse(path, events=("start", "end"))
        for event, elem in context:
            tag = elem.tag
            if event == "start":
                if tag == "calculation":
                    scstep_in_calc = 0
                continue
            # end events
            if tag == "i" and elem.get("name") == "version" and version is None:
                version = (elem.text or "").strip() or None
            elif tag == "i" and elem.get("name") == "ISPIN":
                ispin = int(float((elem.text or "1").strip()))
            elif tag == "i" and elem.get("name") == "LNONCOLLINEAR":
                noncollinear = (elem.text or "F").strip().upper().startswith("T")
            elif tag == "i" and elem.get("name") == "LSORBIT":
                soc = (elem.text or "F").strip().upper().startswith("T")
            elif tag == "i" and elem.get("name") == "NELM":
                nelm = int(float((elem.text or "60").strip()))
            elif tag == "i" and elem.get("name") == "NSW":
                nsw = int(float((elem.text or "0").strip()))
            elif tag == "scstep":
                scstep_in_calc += 1
                elem.clear()
            elif tag == "energy":
                fr = elem.find("i[@name='e_fr_energy']")
                s0 = elem.find("i[@name='e_0_energy']")
                if fr is not None and fr.text:
                    final_free = _ffloat(fr.text)
                if s0 is not None and s0.text:
                    final_sigma0 = _ffloat(s0.text)
            elif tag == "dos":
                dos = _parse_vasprun_dos(elem)
                elem.clear()
            elif tag == "calculation":
                calc_count += 1
                last_scstep_count = scstep_in_calc
                elem.clear()
    except ET.ParseError as exc:
        raise VaspOutputError(f"vasprun.xml is not well-formed: {exc}") from exc

    electronic_converged = last_scstep_count < nelm if last_scstep_count else None
    if nsw == 0:
        ionic_converged = electronic_converged
    else:
        ionic_converged = calc_count < nsw if calc_count else None

    return VasprunSummary(
        vasp_version=version,
        ispin=ispin,
        noncollinear=noncollinear,
        soc=soc,
        electronic_converged=electronic_converged,
        ionic_converged=ionic_converged,
        final_free_energy=final_free,
        final_energy_sigma0=final_sigma0,
        fermi_energy=dos.efermi if dos else None,
        total_magnetization=None,
        dos=dos,
        source_sha256=source_sha256,
    )


def _parse_vasprun_dos(dos_elem: ET.Element) -> DosSummary | None:
    efermi_node = dos_elem.find("i[@name='efermi']")
    if efermi_node is None or not efermi_node.text:
        return None
    efermi = _ffloat(efermi_node.text)
    if efermi is None:
        return None

    total_set = dos_elem.find("total/array/set")
    if total_set is None:
        return None

    channels: list[list[tuple[float, float]]] = []
    for spin_set in total_set.findall("set"):
        rows: list[tuple[float, float]] = []
        for r in spin_set.findall("r"):
            parts = (r.text or "").split()
            if len(parts) >= 2:
                energy = _ffloat(parts[0])
                value = _ffloat(parts[1])
                if energy is not None and value is not None:
                    rows.append((energy, value))
        if rows:
            channels.append(rows)
    if not channels:
        return None

    # Set layout: 1 set = nonmagnetic total; 2 sets = collinear spin up/down;
    # 4 sets = noncollinear (total + magnetization x/y/z), where set 1 is the
    # total DOS. Only spin-resolved (collinear) runs report two channels here.
    if len(channels) == 2:
        is_spin_polarized = True
        dos_at_fermi = [_dos_at(channels[0], efermi), _dos_at(channels[1], efermi)]
        total_rows = _sum_channels(channels)
    else:
        is_spin_polarized = False
        total_rows = channels[0]
        dos_at_fermi = [_dos_at(total_rows, efermi)]

    return DosSummary(
        efermi=efermi,
        is_spin_polarized=is_spin_polarized,
        dos_at_fermi=dos_at_fermi,
        estimated_gap=_estimate_gap(total_rows, efermi),
    )


def _dos_at(rows: list[tuple[float, float]], energy: float) -> float:
    """Nearest-grid tDOS value at ``energy``."""

    return min(rows, key=lambda row: abs(row[0] - energy))[1]


def _sum_channels(
    channels: list[list[tuple[float, float]]],
) -> list[tuple[float, float]]:
    """Sum spin channels on a shared energy grid into a total tDOS."""

    base = channels[0]
    summed: list[tuple[float, float]] = []
    for i, (energy, value) in enumerate(base):
        total = value
        for other in channels[1:]:
            if i < len(other):
                total += other[i][1]
        summed.append((energy, total))
    return summed


def _estimate_gap(
    total_rows: list[tuple[float, float]], efermi: float
) -> float | None:
    """Gap only if E_F sits inside a contiguous near-zero tDOS window.

    Returns ``None`` for metals (finite DOS at E_F). This is a coarse estimate
    from the total DOS grid, not a rigorous band-gap calculation.
    """

    rows = sorted(total_rows, key=lambda row: row[0])
    peak = max((value for _, value in rows), default=0.0)
    if peak <= 0.0:
        return None
    tol = peak * 1.0e-3

    fermi_index = min(range(len(rows)), key=lambda i: abs(rows[i][0] - efermi))
    if rows[fermi_index][1] > tol:
        return None  # metallic: finite DOS at the Fermi level

    lo = fermi_index
    while lo - 1 >= 0 and rows[lo - 1][1] <= tol:
        lo -= 1
    hi = fermi_index
    while hi + 1 < len(rows) and rows[hi + 1][1] <= tol:
        hi += 1
    gap = rows[hi][0] - rows[lo][0]
    return gap if gap > 1.0e-3 else None


# --------------------------------------------------------------------------- #
# Combiner                                                                      #
# --------------------------------------------------------------------------- #


def read_oszicar(path: Path) -> OszicarSummary:
    data = path.read_bytes()
    return parse_oszicar(
        data.decode("utf-8", errors="replace"), source_sha256=_sha256_bytes(data)
    )


def read_outcar(path: Path) -> OutcarSummary:
    data = path.read_bytes()
    return parse_outcar(
        data.decode("utf-8", errors="replace"), source_sha256=_sha256_bytes(data)
    )


def read_vasprun(path: Path) -> VasprunSummary:
    return parse_vasprun(path, source_sha256=_sha256_bytes(path.read_bytes()))


def _canonical_json(result: CalculationResult) -> bytes:
    payload = result.model_dump(mode="json", exclude_none=True)
    payload.pop("result_sha256", None)
    return (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _cross_check(
    quantity: str, sources: dict[str, float | None], tolerance: float
) -> CrossCheck | None:
    present = {name: value for name, value in sources.items() if value is not None}
    if len(present) < 2:
        return None
    values = list(present.values())
    max_diff = max(values) - min(values)
    return CrossCheck(
        quantity=quantity,
        sources=present,
        max_abs_diff=abs(max_diff),
        tolerance=tolerance,
        consistent=abs(max_diff) <= tolerance,
    )


def _site_species(directory: Path, count: int) -> list[str] | None:
    from magnetic_vasp_agent.poscar import read_poscar

    for name in ("POSCAR", "CONTCAR"):
        candidate = directory / name
        if candidate.is_file():
            try:
                species = read_poscar(candidate).site_species
            except (ValueError, OSError):
                return None
            if len(species) == count:
                return species
    return None


def analyze_calculation(directory: Path) -> CalculationResult:
    """Parse whichever output files exist in ``directory`` and reconcile them."""

    if not directory.is_dir():
        raise VaspOutputError(f"not a directory: {directory}")

    oszicar = None
    outcar = None
    vasprun = None
    warnings: list[str] = []

    if (directory / "OSZICAR").is_file():
        oszicar = read_oszicar(directory / "OSZICAR")
    if (directory / "OUTCAR").is_file():
        outcar = read_outcar(directory / "OUTCAR")
    if (directory / "vasprun.xml").is_file():
        try:
            vasprun = read_vasprun(directory / "vasprun.xml")
        except VaspOutputError as exc:
            warnings.append(f"vasprun.xml ignored: {exc}")

    if oszicar is None and outcar is None and vasprun is None:
        raise VaspOutputError(
            f"no OSZICAR/OUTCAR/vasprun.xml found in {directory}"
        )

    # Magnetic mode.
    if outcar is not None:
        magnetic_mode = outcar.magnetic_mode
    elif vasprun is not None:
        if vasprun.soc:
            magnetic_mode = DetectedMagneticMode.SOC
        elif vasprun.noncollinear:
            magnetic_mode = DetectedMagneticMode.NONCOLLINEAR
        elif vasprun.ispin == 2:
            magnetic_mode = DetectedMagneticMode.COLLINEAR
        else:
            magnetic_mode = DetectedMagneticMode.NONMAGNETIC
    else:
        magnetic_mode = DetectedMagneticMode.UNKNOWN

    # Energy (OUTCAR TOTEN preferred, then vasprun, then OSZICAR).
    final_energy = None
    for candidate in (
        outcar.free_energy_toten if outcar else None,
        vasprun.final_free_energy if vasprun else None,
        oszicar.final_free_energy if oszicar else None,
    ):
        if candidate is not None:
            final_energy = candidate
            break

    nions = outcar.parameters.nions if outcar else None
    ion_magnetizations = outcar.ion_magnetizations if outcar else []
    species = _site_species(directory, len(ion_magnetizations)) if ion_magnetizations else None
    if species is not None:
        ion_magnetizations = [
            ion.model_copy(update={"species": species[i]})
            for i, ion in enumerate(ion_magnetizations)
        ]

    energy_per_atom = (
        final_energy / nions if (final_energy is not None and nions) else None
    )

    total_magnetization = None
    total_vector = None
    if outcar is not None and outcar.total_magnetization is not None:
        total_magnetization = outcar.total_magnetization
    elif oszicar is not None and oszicar.total_magnetization is not None:
        total_magnetization = oszicar.total_magnetization
    if outcar is not None and outcar.total_magnetization_vector is not None:
        total_vector = outcar.total_magnetization_vector
    elif oszicar is not None and oszicar.magnetization_vector is not None:
        total_vector = oszicar.magnetization_vector

    fermi = None
    if outcar is not None and outcar.fermi_energy is not None:
        fermi = outcar.fermi_energy
    elif vasprun is not None and vasprun.fermi_energy is not None:
        fermi = vasprun.fermi_energy

    converged, conv_warnings = _decide_convergence(oszicar, outcar, vasprun)
    warnings.extend(conv_warnings)

    cross_checks = _build_cross_checks(oszicar, outcar, vasprun)
    for check in cross_checks:
        if not check.consistent:
            warnings.append(
                f"{check.quantity} disagrees across sources "
                f"(max diff {check.max_abs_diff:.4g} > tol {check.tolerance:.4g})"
            )

    result = CalculationResult(
        directory=str(directory),
        magnetic_mode=magnetic_mode,
        converged=converged,
        final_energy=final_energy,
        energy_per_atom=energy_per_atom,
        fermi_energy=fermi,
        total_magnetization=total_magnetization,
        total_magnetization_vector=total_vector,
        ion_magnetizations=ion_magnetizations,
        max_force=outcar.max_force if outcar else None,
        oszicar=oszicar,
        outcar=outcar,
        vasprun=vasprun,
        cross_checks=cross_checks,
        warnings=warnings,
    )
    return result.model_copy(update={"result_sha256": _sha256_bytes(_canonical_json(result))})


def _decide_convergence(
    oszicar: OszicarSummary | None,
    outcar: OutcarSummary | None,
    vasprun: VasprunSummary | None,
) -> tuple[bool, list[str]]:
    warnings: list[str] = []

    if vasprun is not None and vasprun.electronic_converged is not None:
        electronic = vasprun.electronic_converged
    elif outcar is not None and outcar.electronic_converged:
        electronic = True
    elif (
        oszicar is not None
        and outcar is not None
        and oszicar.final_free_energy is not None
    ):
        electronic = oszicar.electronic_steps_last < outcar.parameters.nelm
    else:
        electronic = False
        warnings.append("electronic convergence could not be established from outputs")

    if vasprun is not None and vasprun.ionic_converged is not None:
        ionic = vasprun.ionic_converged
    elif outcar is not None and outcar.parameters.nsw == 0:
        ionic = True
    elif outcar is not None:
        ionic = outcar.electronic_converged
    else:
        ionic = electronic

    return bool(electronic and ionic), warnings


def _build_cross_checks(
    oszicar: OszicarSummary | None,
    outcar: OutcarSummary | None,
    vasprun: VasprunSummary | None,
) -> list[CrossCheck]:
    checks: list[CrossCheck] = []

    energy = _cross_check(
        "final_energy",
        {
            "outcar": outcar.free_energy_toten if outcar else None,
            "oszicar": oszicar.final_free_energy if oszicar else None,
            "vasprun": vasprun.final_free_energy if vasprun else None,
        },
        _ENERGY_TOL,
    )
    if energy is not None:
        checks.append(energy)

    mag = _cross_check(
        "total_magnetization",
        {
            "outcar": outcar.total_magnetization if outcar else None,
            "oszicar": oszicar.total_magnetization if oszicar else None,
        },
        _MAG_TOL,
    )
    if mag is not None:
        checks.append(mag)

    fermi = _cross_check(
        "fermi_energy",
        {
            "outcar": outcar.fermi_energy if outcar else None,
            "vasprun": vasprun.fermi_energy if vasprun else None,
        },
        _FERMI_TOL,
    )
    if fermi is not None:
        checks.append(fermi)

    return checks
