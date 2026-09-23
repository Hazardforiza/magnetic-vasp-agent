"""Audit original KPOINTS with adjacent INCAR/POSCAR; write a new JSON report."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from magnetic_vasp_agent.incar import read_incar
from magnetic_vasp_agent.kpoints import (
    parse_kpoints,
    read_kpoints,
    render_kpoints,
    validate_kpoints,
)
from magnetic_vasp_agent.models import CalculationGoal
from magnetic_vasp_agent.poscar import read_poscar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error("reference root is not a directory")
    if args.output.exists():
        parser.error("output exists; choose a new report path")
    rows = []
    modes = Counter()
    for path in sorted(root.rglob("KPOINTS")):
        row = {
            "source": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        try:
            parsed = read_kpoints(path)
            incar = read_incar(path.with_name("INCAR"))
            structure = read_poscar(path.with_name("POSCAR"))
            # Only the reviewed reference directory convention is used here.
            # Production commands receive an explicit shared CalculationGoal.
            name = path.parent.name.lower()
            task = "band" if "band" in name else ("dos" if "dos" in name else "static")
            if name in {"001", "011", "111"}:
                task = "magnetic_anisotropy"
            mode = (
                "soc"
                if incar.spec.LSORBIT
                else ("noncollinear" if incar.spec.spinor else "collinear")
            )
            goal = CalculationGoal(task=task, magnetic_mode=mode)
            report = validate_kpoints(parsed.spec, structure, goal, incar.spec)
            report.incar_validation.diagnostics = (
                incar.diagnostics + report.incar_validation.diagnostics
            )
            row.update(report.model_dump(mode="json"))
            row["roundtrip_equal"] = (
                parse_kpoints(render_kpoints(parsed.spec)).spec == parsed.spec
            )
            row["spec"] = parsed.spec.model_dump(mode="json", exclude_none=True)
            row["goal"] = goal.model_dump(mode="json")
            row["goal_source"] = "reviewed reference directory names; audit only"
            modes[parsed.spec.mode] += 1
        except (ValueError, OSError) as exc:
            row.update(valid=False, roundtrip_equal=False, error=str(exc))
        rows.append(row)
    report = {
        "root": str(root),
        "count": len(rows),
        "mode_counts": dict(modes),
        "passed": sum(row["valid"] and row["roundtrip_equal"] for row in rows),
        "calculations": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(
        f"{report['passed']}/{report['count']} imported, cross-checked, and round-tripped"
    )
    print(json.dumps(report["mode_counts"]))
    print(args.output.resolve())
    return 0 if rows and report["passed"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
