"""Auditable INCAR import, canonical rendering, and cross-file validation."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

from pymatgen.io.vasp.inputs import Incar

from magnetic_vasp_agent.incar_models import (
    IncarDiagnostic,
    IncarSpec,
    IncarValidation,
    ParsedIncar,
)
from magnetic_vasp_agent.models import (
    CalculationGoal,
    MagneticMode,
    PotentialPlan,
    PotentialRegistry,
    Structure,
)
from magnetic_vasp_agent.potcar.selector import validate_plan

BOOL_TAGS = set(
    "LWAVE LCHARG LSORBIT LNONCOLLINEAR LASPH LORBMOM GGA_COMPAT LWANNIER90 ADDGRID LDAU".split()
)
INT_TAGS = set(
    "ISTART ICHARG ISPIN NELM NELMIN NBANDS NCORE NPAR ISYM ISMEAR NEDOS LORBIT LMAXMIX LDAUTYPE NSW IBRION ISIF".split()
)
FLOAT_TAGS = set(
    "ENCUT EDIFF SIGMA SOCSCALE AMIX BMIX AMIX_MAG BMIX_MAG EDIFFG POTIM".split()
)
LIST_TAGS = {"MAGMOM", "SAXIS", "LDAUL", "LDAUU", "LDAUJ"}
ENUMS = {
    "PREC": ["Low", "Medium", "High", "Normal", "Single", "Accurate"],
    "ALGO": ["Normal", "Fast", "VeryFast", "All", "Damped", "Exact", "None"],
}
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?\Z")


class IncarFormatError(ValueError):
    pass


def _logical_lines(text: str) -> Iterator[tuple[int, str]]:
    parts: list[str] = []
    start = 0
    for line_number, line in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        body = re.split(r"[!#]", line, maxsplit=1)[0].strip()
        if not body:
            continue
        if not parts:
            start = line_number
        continued = body.endswith("\\")
        parts.append(body[:-1].rstrip() if continued else body)
        if not continued:
            yield start, " ".join(parts)
            parts = []
    if parts:
        raise IncarFormatError(f"line {start}: unterminated continuation")


def _number(token: str, integer: bool = False) -> str:
    if not NUMBER.fullmatch(token) or (
        integer and not re.fullmatch(r"[+-]?\d+", token)
    ):
        raise ValueError(f"invalid {'integer' if integer else 'number'}: {token!r}")
    normalized = token.replace("D", "E").replace("d", "e")
    value = float(normalized)
    if not math.isfinite(value):
        raise ValueError("numbers must be finite")
    decimal = Decimal(normalized)
    if not decimal:
        return "0"
    if decimal and value == 0:
        raise ValueError("number underflows a double-precision float")
    # pymatgen's legacy list parser does not reliably handle exponent tokens.
    # Fixed-point tokens also avoid its permissive prefix parsing of scalars.
    return str(int(token)) if integer else format(decimal, "f")


def _value(
    tag: str, value: str, line: int, legacy: bool, diagnostics: list[IncarDiagnostic]
):
    if tag == "SYSTEM":
        return value
    if tag in BOOL_TAGS or tag == "LREAL":
        token = value.upper()
        if token == ".TURE." and legacy:
            diagnostics.append(
                IncarDiagnostic(
                    severity="warning",
                    code="LEGACY_BOOLEAN",
                    tag=tag,
                    line=line,
                    message="Interpreted .Ture. as .TRUE.; original spelling was invalid.",
                )
            )
            token = ".TRUE."
        if token in {"T", "TRUE", ".T.", ".TRUE."}:
            return True
        if token in {"F", "FALSE", ".F.", ".FALSE."}:
            return False
        if tag == "LREAL" and token in {"AUTO", "A", "ON", "O"}:
            return "Auto" if token in {"AUTO", "A"} else "On"
        raise ValueError(f"invalid boolean or LREAL value: {value!r}")
    if tag in ENUMS:
        options = {item.upper(): item for item in ENUMS[tag]}
        if value.upper() not in options:
            raise ValueError(f"unsupported {tag} value: {value!r}")
        return options[value.upper()]
    if tag in LIST_TAGS:
        tokens: list[str] = []
        count = 0
        for token in value.split():
            pieces = token.split("*")
            if len(pieces) > 2:
                raise ValueError("use n*value repetition, not chained repetitions")
            repeats = 1
            if len(pieces) == 2:
                if not pieces[0].isdigit() or not 1 <= int(pieces[0]) <= 1000000:
                    raise ValueError("invalid repetition count")
                repeats = int(pieces[0])
            count += repeats
            if count > 1000000:
                raise ValueError("expanded array exceeds 1000000 values")
            number = _number(pieces[-1], integer=tag == "LDAUL")
            tokens.append(f"{repeats}*{number}" if len(pieces) == 2 else number)
        if not tokens:
            raise ValueError("empty array")
        if tag == "SAXIS":
            # SAXIS is not a list key in pymatgen; use its numeric-list grammar.
            parsed = Incar.proc_val("MAGMOM", " ".join(tokens))
        else:
            parsed = Incar.proc_val(tag, " ".join(tokens))
        if not isinstance(parsed, list) or len(parsed) != count:
            raise ValueError("array parser did not preserve the expanded value count")
        return parsed
    if tag in INT_TAGS | FLOAT_TAGS:
        normalized = _number(value, integer=tag in INT_TAGS)
        return Incar.proc_val(tag, normalized)
    raise ValueError(f"unsupported INCAR tag {tag}")


def parse_incar(text: str, *, legacy_booleans: bool = True) -> ParsedIncar:
    values: dict = {}
    source_lines: dict[str, int] = {}
    diagnostics: list[IncarDiagnostic] = []
    for line, body in _logical_lines(text):
        for statement in body.split(";"):
            statement = statement.strip()
            if not statement:
                continue
            if "=" not in statement:
                first = statement.split()[0].upper()
                if first in IncarSpec.model_fields or re.match(r"^[+\-.\d]", statement):
                    raise IncarFormatError(
                        f"line {line}: missing assignment or continuation"
                    )
                diagnostics.append(
                    IncarDiagnostic(
                        severity="warning",
                        code="IGNORED_TEXT",
                        line=line,
                        message=f"Ignored unassigned heading: {statement}",
                    )
                )
                continue
            tag, raw = (piece.strip() for piece in statement.split("=", 1))
            tag = tag.upper()
            if tag not in IncarSpec.model_fields:
                raise IncarFormatError(f"line {line}: unsupported INCAR tag {tag!r}")
            try:
                parsed = _value(tag, raw, line, legacy_booleans, diagnostics)
                checked = IncarSpec.model_validate({tag: parsed})
                value = getattr(checked, tag)
            except (ValueError, TypeError, OverflowError) as exc:
                raise IncarFormatError(f"line {line} ({tag}): {exc}") from exc
            if tag in values:
                if value != values[tag]:
                    raise IncarFormatError(
                        f"line {line} ({tag}): conflicts with line {source_lines[tag]}"
                    )
                diagnostics.append(
                    IncarDiagnostic(
                        severity="warning",
                        code="DUPLICATE_TAG",
                        tag=tag,
                        line=line,
                        message=f"Repeated value matches line {source_lines[tag]}.",
                    )
                )
            values[tag] = value
            source_lines[tag] = line
    if not values:
        raise IncarFormatError("INCAR contains no supported assignments")
    return ParsedIncar(
        spec=IncarSpec.model_validate(values),
        source_lines=source_lines,
        diagnostics=diagnostics,
    )


def read_incar(path: Path, *, legacy_booleans: bool = True) -> ParsedIncar:
    return parse_incar(
        path.read_text(encoding="utf-8-sig"), legacy_booleans=legacy_booleans
    )


def render_incar(spec: IncarSpec) -> str:
    spec = IncarSpec.model_validate(spec.model_dump())
    data = spec.model_dump(exclude_none=True)
    if not data:
        raise ValueError("cannot render an empty INCAR")
    if spec.spinor and spec.MAGMOM is not None:
        if len(spec.MAGMOM) % 3:
            raise ValueError("spinor MAGMOM must contain complete triples")
        # pymatgen interprets flat spinor MAGMOM as scalar per-site moments.
        data["MAGMOM"] = [spec.MAGMOM[i : i + 3] for i in range(0, len(spec.MAGMOM), 3)]
    rendered = Incar(data).get_str(sort_keys=True)
    lines = []
    for line in rendered.splitlines():
        tag = line.split("=", 1)[0].strip()
        if isinstance(data[tag], bool):
            line = f"{tag} = {'.TRUE.' if data[tag] else '.FALSE.'}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def write_incar(spec: IncarSpec, path: Path) -> None:
    rendered = render_incar(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)


def validate_incar(
    spec: IncarSpec,
    structure: Structure,
    *,
    goal: CalculationGoal | None = None,
    plan: PotentialPlan | None = None,
    registry: PotentialRegistry | None = None,
    executable: str | None = None,
    source_lines: dict[str, int] | None = None,
) -> IncarValidation:
    result = IncarValidation(executable=spec.executable)

    def issue(code: str, tag: str, message: str, severity: str = "error") -> None:
        result.diagnostics.append(
            IncarDiagnostic(
                code=code,
                tag=tag,
                message=message,
                severity=severity,
                line=(source_lines or {}).get(tag),
            )
        )

    expected = structure.num_atoms * (3 if spec.spinor else 1)
    if spec.MAGMOM is not None and len(spec.MAGMOM) != expected:
        issue(
            "MAGMOM_LENGTH",
            "MAGMOM",
            f"Expected {expected} values in POSCAR order; got {len(spec.MAGMOM)}.",
        )
    if spec.MAGMOM is None and (spec.spinor or spec.ISPIN == 2 or goal is not None):
        issue(
            "MAGMOM_MISSING",
            "MAGMOM",
            "Explicit moments are required for a reproducible magnetic workflow.",
        )
    if spec.MAGMOM is not None and not spec.spinor and spec.ISPIN != 2:
        issue(
            "ISPIN_REQUIRED",
            "ISPIN",
            "Collinear MAGMOM requires explicit ISPIN=2 in this workflow.",
        )
    if spec.LSORBIT and spec.LNONCOLLINEAR is False:
        issue(
            "SOC_CONFLICT",
            "LNONCOLLINEAR",
            "LSORBIT implies noncollinear spinors; explicit FALSE conflicts.",
        )
    if spec.spinor and spec.ISPIN == 2:
        issue(
            "SPINOR_ISPIN",
            "ISPIN",
            "ISPIN=2 is redundant for spinors and incompatible with some newer VASP versions; omit it in new inputs.",
            "warning",
        )
    if spec.spinor and (spec.ISYM is None or spec.ISYM > 0):
        issue(
            "SYMMETRY_REVIEW",
            "ISYM",
            "Project recipes use ISYM=-1; review magnetic symmetry when enabling symmetry reduction.",
            "warning",
        )
    if executable is not None and executable != spec.executable:
        issue(
            "EXECUTABLE_MISMATCH",
            "LSORBIT",
            f"Expected {spec.executable}, got {executable}.",
        )
    if goal is not None:
        mode = (
            MagneticMode.SOC
            if spec.LSORBIT
            else (MagneticMode.NONCOLLINEAR if spec.spinor else MagneticMode.COLLINEAR)
        )
        if mode != goal.magnetic_mode:
            issue(
                "MAGNETIC_MODE",
                "LSORBIT",
                f"Goal expects {goal.magnetic_mode.value}; input is {mode.value}.",
            )
        if (
            goal.task.value in {"static", "dos", "band", "magnetic_anisotropy"}
            and (spec.NSW or 0) > 0
        ):
            issue("TASK_IONS", "NSW", "Static electronic tasks must not move ions.")
        if goal.task.value in {"dos", "band"} and spec.ICHARG != 11:
            issue(
                "TASK_CHARGE",
                "ICHARG",
                "This PBE DOS/band recipe requires a preceding SCF and ICHARG=11.",
            )
        if goal.task.value == "relax" and (
            (spec.NSW or 0) <= 0 or spec.IBRION not in {1, 2, 3}
        ):
            issue(
                "RELAX_STEPS",
                "NSW",
                "Relaxation requires NSW>0 and an ionic relaxation algorithm.",
            )
    for tag in ("LDAUL", "LDAUU", "LDAUJ"):
        array = getattr(spec, tag)
        if array is not None and len(array) != len(structure.species_order):
            issue(
                "LDAU_LENGTH",
                tag,
                f"Expected {len(structure.species_order)} values in POSCAR species order.",
            )
        if spec.LDAU and array is None:
            issue(
                "LDAU_MISSING",
                tag,
                "DFT+U requires an explicit array for every species.",
            )
        if not spec.LDAU and array is not None:
            issue(
                "LDAU_DISABLED", tag, "Array is inactive unless LDAU=.TRUE.", "warning"
            )
    if spec.LDAU:
        if spec.LDAUTYPE is None:
            issue(
                "LDAUTYPE_MISSING", "LDAUTYPE", "Specify the U formulation explicitly."
            )
        max_l = max(spec.LDAUL or [-1])
        if max_l >= 2 and (spec.LMAXMIX or 2) < 2 * max_l:
            issue(
                "LMAXMIX_LOW",
                "LMAXMIX",
                f"Use at least {2 * max_l} for the correlated shell, especially when reusing CHGCAR.",
                "warning",
            )
    if spec.NELMIN is not None and spec.NELM is not None and spec.NELMIN > spec.NELM:
        issue("ELECTRONIC_STEPS", "NELMIN", "NELMIN exceeds NELM.")
    if spec.LORBMOM and not spec.LSORBIT:
        issue(
            "ORBITAL_WITHOUT_SOC",
            "LORBMOM",
            "Orbital-moment output should be checked against the SOC setup.",
            "warning",
        )
    if spec.SOCSCALE is not None:
        issue(
            "SOCSCALE_CAPABILITY",
            "SOCSCALE",
            "Historical extension retained; verify support in the target VASP binary before use.",
            "warning",
        )
        if not spec.LSORBIT:
            issue(
                "SOCSCALE_WITHOUT_SOC",
                "SOCSCALE",
                "SOC scaling requires LSORBIT=.TRUE.",
            )
    charge = (
        spec.ICHARG if spec.ICHARG is not None else (2 if spec.ISTART == 0 else None)
    )
    if charge is not None and charge % 10 == 1:
        result.required_artifacts.append("CHGCAR")
    if (spec.ISTART or 0) > 0 or (charge is not None and charge % 10 == 0):
        result.required_artifacts.append("WAVECAR")
    if result.required_artifacts:
        issue(
            "RESTART_INPUTS",
            "ICHARG",
            "Requires compatible "
            + ", ".join(result.required_artifacts)
            + "; artifact compatibility is not checked by this INCAR validator.",
            "warning",
        )
    if (plan is None) != (registry is None):
        issue("POTENTIAL_CONTEXT", "ENCUT", "Provide both potential plan and registry.")
    elif plan is not None and registry is not None:
        try:
            records = validate_plan(plan, registry)
            if plan.species_order != structure.species_order:
                raise ValueError("Potential plan species order differs from POSCAR.")
            if goal is not None and plan.goal.xc_family != goal.xc_family:
                raise ValueError(
                    "Goal and potential plan use different potential families."
                )
            cutoffs = [record.enmax for record in records]
            if not cutoffs or any(
                x is None or not math.isfinite(x) or x <= 0 for x in cutoffs
            ):
                raise ValueError(
                    "Selected potential ENMAX metadata is missing or invalid."
                )
            minimum = max(cutoffs)
            if spec.ENCUT is None or spec.ENCUT < minimum:
                issue(
                    "ENCUT_BELOW_ENMAX",
                    "ENCUT",
                    f"Project cutoff floor is max(selected ENMAX)={minimum:g} eV; convergence testing is still required.",
                )
        except ValueError as exc:
            issue("POTENTIAL_CONTEXT", "ENCUT", str(exc))
    else:
        issue(
            "ENCUT_UNCHECKED",
            "ENCUT",
            "No selected potential registry supplied; ENMAX check was not performed.",
            "warning",
        )
    return result
