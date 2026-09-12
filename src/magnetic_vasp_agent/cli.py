from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pydantic
import typer

from magnetic_vasp_agent import __version__
from magnetic_vasp_agent.agent_cli import agent_app
from magnetic_vasp_agent.cli_io import Utf8TyperGroup
from magnetic_vasp_agent.execution_cli import execution_app
from magnetic_vasp_agent.incar import (
    read_incar,
    render_incar,
    validate_incar,
    write_incar,
)
from magnetic_vasp_agent.incar_models import IncarRecipeRequest
from magnetic_vasp_agent.incar_recipes import build_incar
from magnetic_vasp_agent.kpoints_cli import kpoints_app
from magnetic_vasp_agent.knowledge_cli import knowledge_app
from magnetic_vasp_agent.mcif_cli import mcif_app
from magnetic_vasp_agent.models import (
    CalculationGoal,
    CalculationTask,
    CandidateComposition,
    MagneticMode,
    PotentialPlan,
)
from magnetic_vasp_agent.poscar import read_poscar
from magnetic_vasp_agent.postprocess_cli import postprocess_app
from magnetic_vasp_agent.potcar import (
    assemble_potcar,
    load_policy,
    load_registry,
    scan_library,
    select_by_policy,
    write_registry,
)
from magnetic_vasp_agent.workflow_cli import workflow_app

app = typer.Typer(
    cls=Utf8TyperGroup,
    no_args_is_help=True,
    help="Local magnetic VASP control plane.",
)
potcar_app = typer.Typer(no_args_is_help=True, help="Manage licensed POTCAR data.")
structure_app = typer.Typer(no_args_is_help=True, help="Inspect crystal structures.")
incar_app = typer.Typer(
    no_args_is_help=True, help="Parse, validate, and generate typed INCAR."
)
app.add_typer(potcar_app, name="potcar")
app.add_typer(structure_app, name="structure")
app.add_typer(incar_app, name="incar")
app.add_typer(kpoints_app, name="kpoints")
app.add_typer(workflow_app, name="workflow")
app.add_typer(mcif_app, name="mcif")
app.add_typer(execution_app, name="execution")
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(postprocess_app, name="postprocess")
app.add_typer(agent_app, name="agent")


@app.command()
def doctor() -> None:
    """Show the local runtime used by the deterministic core."""
    typer.echo(f"magnetic-vasp-agent {__version__}")
    typer.echo(f"python {platform.python_version()}")
    typer.echo(f"pydantic {pydantic.__version__}")


@potcar_app.command("scan")
def scan_potcar(
    root: Path = typer.Option(..., "--root", exists=True, file_okay=False),
    family: str = typer.Option("PBE_54", "--family"),
    output: Path = typer.Option(Path("var/potcar_registry.json"), "--output"),
) -> None:
    """Inventory POTCAR metadata and hashes without copying POTCAR content."""
    registry = scan_library(root, family)
    write_registry(registry, output)
    typer.echo(f"indexed {len(registry.records)} potentials")
    typer.echo(str(output.resolve()))


@structure_app.command("show")
def show_structure(
    poscar: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Print the species order and atom counts parsed from a POSCAR."""
    structure = read_poscar(poscar)
    typer.echo(f"comment   {structure.comment}")
    typer.echo(f"formula   {structure.formula}")
    typer.echo(f"atoms     {structure.num_atoms}")
    typer.echo(f"mode      {structure.coordinate_mode.value}")
    typer.echo("species   " + " ".join(structure.species_order))
    typer.echo("counts    " + " ".join(str(count) for count in structure.counts))
    if structure.selective_dynamics is not None:
        typer.echo("selective dynamics enabled")


@potcar_app.command("select")
def select_potcar(
    registry_path: Path = typer.Option(..., "--registry", exists=True, dir_okay=False),
    policy_path: Path = typer.Option(..., "--policy", exists=True, dir_okay=False),
    poscar: Path | None = typer.Option(
        None,
        "--poscar",
        exists=True,
        dir_okay=False,
        help="Derive species order from this POSCAR (preferred).",
    ),
    elements: list[str] | None = typer.Option(
        None,
        "--element",
        help="Manual species order; mutually exclusive with --poscar.",
    ),
    candidate_id: str | None = typer.Option(
        None,
        "--candidate-id",
        help="Defaults to the POSCAR formula when --poscar is used.",
    ),
    task: CalculationTask = typer.Option(CalculationTask.STATIC, "--task"),
    magnetic_mode: MagneticMode = typer.Option(
        MagneticMode.COLLINEAR, "--magnetic-mode"
    ),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Create a deterministic baseline potential plan in POSCAR species order."""
    if poscar is not None and elements:
        raise typer.BadParameter("use either --poscar or --element, not both")
    if poscar is not None:
        candidate = CandidateComposition.from_structure(
            read_poscar(poscar), candidate_id
        )
    elif elements:
        if candidate_id is None:
            raise typer.BadParameter("--candidate-id is required with --element")
        candidate = CandidateComposition(
            candidate_id=candidate_id, species_order=elements
        )
    else:
        raise typer.BadParameter("provide --poscar or at least one --element")

    registry = load_registry(registry_path)
    policy = load_policy(policy_path)
    goal = CalculationGoal(
        task=task,
        magnetic_mode=magnetic_mode,
        xc_family=registry.family,
    )
    plan = select_by_policy(candidate, goal, registry, policy)
    rendered = plan.model_dump_json(indent=2) + "\n"
    if output is None:
        typer.echo(rendered, nl=False)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    typer.echo(str(output.resolve()))


@potcar_app.command("build")
def build_potcar(
    registry_path: Path = typer.Option(..., "--registry", exists=True, dir_okay=False),
    plan_path: Path = typer.Option(..., "--plan", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output"),
    manifest_output: Path | None = typer.Option(None, "--manifest-output"),
) -> None:
    """Assemble POTCAR after validating names, order, family, and source hashes."""
    registry = load_registry(registry_path)
    plan = PotentialPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    manifest = assemble_potcar(plan, registry, output)
    manifest_path = manifest_output or output.with_name("potcar_manifest.json")
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(str(output.resolve()))
    typer.echo(str(manifest_path.resolve()))


def _incar_error(exc: Exception) -> None:
    typer.echo(f"INCAR error: {exc}", err=True)
    raise typer.Exit(2)


@incar_app.command("show")
def show_incar(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
    strict: bool = typer.Option(
        False, "--strict", help="Reject legacy .Ture. spelling."
    ),
) -> None:
    """Show typed values and source-line diagnostics as JSON."""
    try:
        document = read_incar(path, legacy_booleans=not strict)
        typer.echo(document.model_dump_json(indent=2, exclude_none=True))
    except (ValueError, OSError) as exc:
        _incar_error(exc)


@incar_app.command("validate")
def check_incar(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
    poscar: Path = typer.Option(..., "--poscar", exists=True, dir_okay=False),
    plan_path: Path | None = typer.Option(None, "--plan", exists=True, dir_okay=False),
    registry_path: Path | None = typer.Option(
        None, "--registry", exists=True, dir_okay=False
    ),
    executable: str | None = typer.Option(None, "--executable"),
    strict: bool = typer.Option(False, "--strict"),
) -> None:
    """Check moments, species arrays, selected ENMAX, and executable routing."""
    try:
        if (plan_path is None) != (registry_path is None):
            raise ValueError("--plan and --registry must be supplied together")
        document = read_incar(path, legacy_booleans=not strict)
        plan = (
            PotentialPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
            if plan_path
            else None
        )
        report = validate_incar(
            document.spec,
            read_poscar(poscar),
            plan=plan,
            registry=load_registry(registry_path) if registry_path else None,
            goal=plan.goal if plan else None,
            executable=executable,
            source_lines=document.source_lines,
        )
        report.diagnostics = document.diagnostics + report.diagnostics
        typer.echo(report.model_dump_json(indent=2, exclude_none=True))
    except (ValueError, OSError) as exc:
        _incar_error(exc)
    if not report.valid:
        raise typer.Exit(1)


@incar_app.command("build")
def generate_incar(
    request_path: Path = typer.Option(..., "--request", exists=True, dir_okay=False),
    poscar: Path = typer.Option(..., "--poscar", exists=True, dir_okay=False),
    plan_path: Path = typer.Option(..., "--plan", exists=True, dir_okay=False),
    registry_path: Path = typer.Option(..., "--registry", exists=True, dir_okay=False),
    output: Path = typer.Option(
        ..., "--output", help="New INCAR file; existing files are never overwritten."
    ),
) -> None:
    """Generate an INCAR and its recipe/provenance report from a JSON request."""
    try:
        manifest_path = output.with_name(output.name + ".manifest.json")
        if output.exists() or manifest_path.exists():
            raise FileExistsError(
                "Output or manifest exists; choose a new calculation directory."
            )
        request = IncarRecipeRequest.model_validate_json(
            request_path.read_text(encoding="utf-8")
        )
        plan = PotentialPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
        result = build_incar(
            read_poscar(poscar), request, plan, load_registry(registry_path)
        )
        content = render_incar(result.spec)
        payload = result.model_dump(mode="json", exclude_none=True)
        payload.update(
            {
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "sources": {
                    name: {
                        "path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for name, path in {
                        "request": request_path,
                        "poscar": poscar,
                        "plan": plan_path,
                        "registry": registry_path,
                    }.items()
                },
            }
        )
        write_incar(result.spec, output)
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
        typer.echo(f"{output.resolve()}\n{manifest_path.resolve()}")
    except (ValueError, OSError) as exc:
        _incar_error(exc)


if __name__ == "__main__":
    app()
