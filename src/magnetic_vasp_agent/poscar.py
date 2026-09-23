"""POSCAR parsing and rendering.

Only the VASP 5 layout with an explicit species line is accepted, because the
species line is the single source of truth for POTCAR ordering.
"""

from __future__ import annotations

from pathlib import Path

from magnetic_vasp_agent.models import CoordinateMode, Structure


class PoscarFormatError(ValueError):
    pass


def _float(token: str, where: str) -> float:
    try:
        return float(token.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise PoscarFormatError(f"{where}: {token!r} is not a number") from exc


def _vector(line: str, where: str) -> list[float]:
    tokens = line.split()
    if len(tokens) < 3:
        raise PoscarFormatError(f"{where}: expected three numbers, got {line!r}")
    return [_float(token, where) for token in tokens[:3]]


def _flag(token: str, where: str) -> bool:
    upper = token.upper()
    if upper in {"T", ".TRUE."}:
        return True
    if upper in {"F", ".FALSE."}:
        return False
    raise PoscarFormatError(f"{where}: selective-dynamics flag {token!r} is not T/F")


def parse_poscar(text: str) -> Structure:
    lines = text.splitlines()
    if len(lines) < 8:
        raise PoscarFormatError("POSCAR needs at least 8 lines")

    comment = lines[0].strip()
    scale_tokens = lines[1].split()
    if not scale_tokens:
        raise PoscarFormatError("line 2 (scale): missing scale factor")
    scale = _float(scale_tokens[0], "line 2 (scale)")
    lattice = [_vector(lines[i], f"line {i + 1} (lattice)") for i in range(2, 5)]

    species_tokens = lines[5].split()
    if not species_tokens:
        raise PoscarFormatError("line 6: missing species line")
    if all(token.isdigit() for token in species_tokens):
        raise PoscarFormatError(
            "line 6 holds atom counts, so this is a VASP 4 POSCAR without a "
            "species line; add the species line so POTCAR order can be derived"
        )
    count_tokens = lines[6].split()
    try:
        counts = [int(token) for token in count_tokens]
    except ValueError as exc:
        raise PoscarFormatError(f"line 7 (counts): {lines[6]!r}") from exc

    cursor = 7
    selective = False
    if lines[cursor].strip()[:1].lower() == "s":
        selective = True
        cursor += 1
    mode_char = lines[cursor].strip()[:1].lower()
    mode = (
        CoordinateMode.CARTESIAN if mode_char in {"c", "k"} else CoordinateMode.DIRECT
    )
    cursor += 1

    num_atoms = sum(counts)
    if len(lines) < cursor + num_atoms:
        raise PoscarFormatError(
            f"expected {num_atoms} coordinate lines, found {len(lines) - cursor}"
        )
    positions: list[list[float]] = []
    flags: list[list[bool]] = []
    for index in range(num_atoms):
        where = f"line {cursor + index + 1} (atom {index + 1})"
        tokens = lines[cursor + index].split()
        positions.append(_vector(lines[cursor + index], where))
        if selective:
            if len(tokens) < 6:
                raise PoscarFormatError(f"{where}: missing selective-dynamics flags")
            flags.append([_flag(token, where) for token in tokens[3:6]])
    cursor += num_atoms

    velocities = _parse_velocities(lines[cursor:], num_atoms)

    return Structure(
        comment=comment,
        scale=scale,
        lattice=lattice,
        species_order=species_tokens,
        counts=counts,
        coordinate_mode=mode,
        positions=positions,
        selective_dynamics=flags if selective else None,
        velocities=velocities,
    )


def _parse_velocities(rest: list[str], num_atoms: int) -> list[list[float]] | None:
    body = [line for line in rest if line.strip()]
    if len(body) < num_atoms:
        return None
    try:
        return [_vector(line, "velocities") for line in body[:num_atoms]]
    except PoscarFormatError:
        return None


def read_poscar(path: Path) -> Structure:
    return parse_poscar(path.read_text(encoding="utf-8", errors="replace"))


def _fmt_vector(vector: list[float]) -> str:
    return "".join(f"{component:22.16f}" for component in vector)


def render_poscar(structure: Structure) -> str:
    lines = [
        structure.comment,
        f"{structure.scale:20.14f}",
        *(_fmt_vector(row) for row in structure.lattice),
        "  " + "  ".join(f"{species:>2}" for species in structure.species_order),
        "  " + "  ".join(f"{count:>2}" for count in structure.counts),
    ]
    if structure.selective_dynamics is not None:
        lines.append("Selective dynamics")
    lines.append(
        "Cartesian"
        if structure.coordinate_mode == CoordinateMode.CARTESIAN
        else "Direct"
    )
    for index, position in enumerate(structure.positions):
        line = _fmt_vector(position)
        if structure.selective_dynamics is not None:
            line += "  " + " ".join(
                "T" if flag else "F" for flag in structure.selective_dynamics[index]
            )
        lines.append(line)
    if structure.velocities is not None:
        lines.append("")
        lines.extend(_fmt_vector(velocity) for velocity in structure.velocities)
    return "\n".join(lines) + "\n"


def write_poscar(structure: Structure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_poscar(structure), encoding="utf-8", newline="\n")
