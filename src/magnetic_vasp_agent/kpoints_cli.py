from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer

from magnetic_vasp_agent.incar import read_incar
from magnetic_vasp_agent.incar_models import IncarRecipeRequest
from magnetic_vasp_agent.kpoints import (
    read_kpoints,
    render_kpoints,
    validate_kpoints,
    write_kpoints,
)
from magnetic_vasp_agent.kpoints_models import KpointsRecipeRequest
from magnetic_vasp_agent.kpoints_recipes import build_kpoints
from magnetic_vasp_agent.poscar import read_poscar

kpoints_app = typer.Typer(
    no_args_is_help=True, help="Parse, validate, and generate typed KPOINTS."
)


def _error(exc: Exception) -> None:
    typer.echo(f"KPOINTS error: {exc}", err=True)
    raise typer.Exit(2)


@kpoints_app.command("show")
def show_kpoints(path: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Show the typed sampling and original line numbers as JSON."""
    try:
        typer.echo(read_kpoints(path).model_dump_json(indent=2, exclude_none=True))
    except (ValueError, OSError) as exc:
        _error(exc)


@kpoints_app.command("validate")
def check_kpoints(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
    poscar: Path = typer.Option(..., "--poscar", exists=True, dir_okay=False),
    incar_path: Path = typer.Option(..., "--incar", exists=True, dir_okay=False),
    incar_request_path: Path = typer.Option(
        ...,
        "--incar-request",
        exists=True,
        dir_okay=False,
        help="The shared task/goal comes from this INCAR request.",
    ),
) -> None:
    """Check task consistency, path labels, weights and tetrahedron prerequisites."""
    try:
        incar_request = IncarRecipeRequest.model_validate_json(
            incar_request_path.read_text(encoding="utf-8")
        )
        document = read_incar(incar_path)
        report = validate_kpoints(
            read_kpoints(path).spec,
            read_poscar(poscar),
            incar_request.goal,
            document.spec,
        )
        report.incar_validation.diagnostics = (
            document.diagnostics + report.incar_validation.diagnostics
        )
        typer.echo(report.model_dump_json(indent=2, exclude_none=True))
    except (ValueError, OSError) as exc:
        _error(exc)
    if not report.valid:
        raise typer.Exit(1)


@kpoints_app.command("build")
def generate_kpoints(
    request_path: Path = typer.Option(..., "--request", exists=True, dir_okay=False),
    incar_request_path: Path = typer.Option(
        ..., "--incar-request", exists=True, dir_okay=False
    ),
    incar_path: Path = typer.Option(..., "--incar", exists=True, dir_okay=False),
    poscar: Path = typer.Option(..., "--poscar", exists=True, dir_okay=False),
    output: Path = typer.Option(
        ..., "--output", help="New KPOINTS file; existing outputs are refused."
    ),
) -> None:
    """Generate task-matched KPOINTS and a report with input/output hashes."""
    try:
        manifest_path = output.with_name(output.name + ".manifest.json")
        if output.exists() or manifest_path.exists():
            raise FileExistsError("Output or manifest exists; choose a new directory.")
        request = KpointsRecipeRequest.model_validate_json(
            request_path.read_text(encoding="utf-8")
        )
        incar_request = IncarRecipeRequest.model_validate_json(
            incar_request_path.read_text(encoding="utf-8")
        )
        document = read_incar(incar_path)
        result = build_kpoints(
            read_poscar(poscar), request, document.spec, incar_goal=incar_request.goal
        )
        result.validation.incar_validation.diagnostics = (
            document.diagnostics + result.validation.incar_validation.diagnostics
        )
        payload = result.model_dump(mode="json", exclude_none=True)
        payload.update(
            {
                "sha256": hashlib.sha256(
                    render_kpoints(result.spec).encode("utf-8")
                ).hexdigest(),
                "sources": {
                    name: {
                        "path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for name, path in {
                        "request": request_path,
                        "incar_request": incar_request_path,
                        "incar": incar_path,
                        "poscar": poscar,
                    }.items()
                },
            }
        )
        write_kpoints(result.spec, output)
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
        typer.echo(f"{output.resolve()}\n{manifest_path.resolve()}")
    except (ValueError, OSError) as exc:
        _error(exc)
