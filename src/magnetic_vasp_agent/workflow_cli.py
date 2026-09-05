from __future__ import annotations

from pathlib import Path

import typer

from magnetic_vasp_agent.models import PotentialPolicy, PotentialRegistry
from magnetic_vasp_agent.workflow import (
    digest,
    inspect_bundle,
    json_bytes,
    materialize_workflow,
    prepare_workflow,
)
from magnetic_vasp_agent.workflow_models import Workflow

workflow_app = typer.Typer(
    no_args_is_help=True,
    help="Validate and assemble local workflows without running VASP.",
)


def _prepare(request: Path, registry: Path, policy: Path):
    paths = {"workflow": request, "registry": registry, "policy": policy}
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    return prepare_workflow(
        Workflow.model_validate_json(payloads["workflow"]),
        PotentialRegistry.model_validate_json(payloads["registry"]),
        PotentialPolicy.model_validate_json(payloads["policy"]),
        request.resolve().parent,
        sources={
            name: {"path": str(path.resolve()), "sha256": digest(payloads[name])}
            for name, path in paths.items()
        },
    )


def _error(exc: Exception) -> None:
    typer.echo(f"Workflow error: {exc}", err=True)
    raise typer.Exit(2)


@workflow_app.command("validate")
def validate(
    request: Path = typer.Option(..., "--request", exists=True, dir_okay=False),
    registry: Path = typer.Option(..., "--registry", exists=True, dir_okay=False),
    policy: Path = typer.Option(..., "--policy", exists=True, dir_okay=False),
) -> None:
    """Compile and validate sources/dependencies without writing any output files."""
    try:
        typer.echo(
            json_bytes(_prepare(request, registry, policy).manifest).decode(), nl=False
        )
    except (ValueError, OSError) as exc:
        _error(exc)


@workflow_app.command("build")
def build(
    request: Path = typer.Option(..., "--request", exists=True, dir_okay=False),
    registry: Path = typer.Option(..., "--registry", exists=True, dir_okay=False),
    policy: Path = typer.Option(..., "--policy", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Materialize a new local input bundle; never run VASP or submit jobs."""
    try:
        if output.exists() or output.is_symlink():
            raise FileExistsError("Output exists; choose a new workflow directory")
        path = materialize_workflow(_prepare(request, registry, policy), output)
        typer.echo(str(path.resolve()))
        typer.echo(json_bytes(inspect_bundle(path.parent)).decode(), nl=False)
    except (ValueError, OSError) as exc:
        _error(exc)


@workflow_app.command("inspect")
def inspect(path: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    """Check all input and step-manifest hashes without claiming runtime readiness."""
    try:
        typer.echo(json_bytes(inspect_bundle(path)).decode(), nl=False)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        _error(exc)
