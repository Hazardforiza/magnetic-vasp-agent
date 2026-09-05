"""Read-only audit of reference inputs; writes only the requested JSON report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from magnetic_vasp_agent.incar import (
    parse_incar,
    read_incar,
    render_incar,
    validate_incar,
)
from magnetic_vasp_agent.models import CalculationGoal, CandidateComposition
from magnetic_vasp_agent.poscar import read_poscar
from magnetic_vasp_agent.potcar import load_policy, load_registry, select_by_policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    registry, policy = load_registry(args.registry), load_policy(args.policy)
    rows = []
    tags: dict[str, int] = {}
    for path in sorted(root.rglob("INCAR")):
        row = {
            "source": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        try:
            document = read_incar(path)
            structure = read_poscar(path.with_name("POSCAR"))
            mode = (
                "soc"
                if document.spec.LSORBIT
                else ("noncollinear" if document.spec.spinor else "collinear")
            )
            # This audit checks the imported setup, not a guessed workflow stage.
            goal = CalculationGoal(
                task="static", magnetic_mode=mode, xc_family=registry.family
            )
            plan = select_by_policy(
                CandidateComposition.from_structure(structure), goal, registry, policy
            )
            report = validate_incar(
                document.spec,
                structure,
                plan=plan,
                registry=registry,
                source_lines=document.source_lines,
            )
            report.diagnostics = document.diagnostics + report.diagnostics
            row.update(report.model_dump(mode="json"))
            row["roundtrip_equal"] = (
                parse_incar(render_incar(document.spec), legacy_booleans=False).spec
                == document.spec
            )
            row["species_order"] = structure.species_order
            row["num_atoms"] = structure.num_atoms
            row["potential_names"] = [choice.potential for choice in plan.choices]
            row["typed_values"] = document.spec.model_dump(exclude_none=True)
            for tag in document.spec.model_dump(exclude_none=True):
                tags[tag] = tags.get(tag, 0) + 1
        except (ValueError, OSError) as exc:
            row.update(valid=False, roundtrip_equal=False, error=str(exc))
        rows.append(row)
    output = {
        "root": str(root),
        "count": len(rows),
        "passed": sum(row["valid"] and row["roundtrip_equal"] for row in rows),
        "tag_counts": dict(sorted(tags.items())),
        "calculations": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(output, indent=2, ensure_ascii=True) + "\n")
    print(
        f"{output['passed']}/{output['count']} imported, validated, and round-tripped"
    )
    print(args.output.resolve())
    return 0 if rows and output["passed"] == output["count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
