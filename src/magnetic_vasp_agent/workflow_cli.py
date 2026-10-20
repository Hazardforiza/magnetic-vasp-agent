from __future__ import annotations

import json
from pathlib import Path

import typer

from magnetic_vasp_agent.input_review import (
    build_review_card,
    load_review_history,
    record_input_review,
)
from magnetic_vasp_agent.mcif_workflow import plan_workflow_from_mcif
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


@workflow_app.command("from-mcif")
def from_mcif(
    mcif: list[Path] = typer.Option(..., "--mcif", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output"),
    workflow_id: str | None = typer.Option(None, "--workflow-id"),
    task: str = typer.Option("static", "--task"),
    magnetic_mode: str = typer.Option("auto", "--magnetic-mode"),
    encut: float = typer.Option(600.0, "--encut"),
    spacing: float = typer.Option(0.25, "--spacing"),
    include_dos: bool = typer.Option(False, "--include-dos"),
    dos_spacing: float = typer.Option(0.15, "--dos-spacing"),
    notes: str | None = typer.Option(None, "--notes"),
) -> None:
    """Write a typed workflow request from mCIF. MAGMOM is mapped, never invented."""
    try:
        report = plan_workflow_from_mcif(
            mcif,
            output,
            workflow_id=workflow_id,
            task=task,
            magnetic_mode=magnetic_mode,
            encut=encut,
            spacing=spacing,
            include_dos=include_dos,
            dos_spacing=dos_spacing,
            notes=notes,
        )
        typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
    except (ValueError, OSError) as exc:
        _error(exc)


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
        card = build_review_card(path.parent)
        typer.echo(str(path.resolve()))
        typer.echo(json_bytes(inspect_bundle(path.parent)).decode(), nl=False)
        typer.echo(card.model_dump_json(indent=2))
    except (ValueError, OSError) as exc:
        _error(exc)


@workflow_app.command("inspect")
def inspect(path: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    """Check all input and step-manifest hashes without claiming runtime readiness."""
    try:
        typer.echo(json_bytes(inspect_bundle(path)).decode(), nl=False)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        _error(exc)


@workflow_app.command("review-card")
def review_card(
    bundle: Path = typer.Option(..., "--bundle", exists=True, file_okay=False),
) -> None:
    """Show or create the human review card for a generated input bundle."""
    try:
        typer.echo(build_review_card(bundle).model_dump_json(indent=2))
    except (ValueError, OSError) as exc:
        _error(exc)


@workflow_app.command("review")
def review(
    bundle: Path = typer.Option(..., "--bundle", exists=True, file_okay=False),
    reviewer: str = typer.Option(..., "--reviewer"),
    decision: str = typer.Option(..., "--decision", help="accept, reject, or revise"),
    notes: str | None = typer.Option(None, "--notes"),
    lesson: str | None = typer.Option(None, "--lesson"),
    tag: list[str] = typer.Option([], "--tag"),
) -> None:
    """Record a human review. The agent cannot approve itself."""
    try:
        if decision not in {"accept", "reject", "revise"}:
            raise ValueError("decision must be accept, reject, or revise")
        card = record_input_review(
            bundle,
            reviewer=reviewer,
            decision=decision,
            notes=notes,
            lesson=lesson,
            tags=tag,
        )
        typer.echo(card.model_dump_json(indent=2))
    except (ValueError, OSError) as exc:
        _error(exc)


@workflow_app.command("reviews")
def reviews(
    species: list[str] = typer.Option([], "--species"),
    task: str | None = typer.Option(None, "--task"),
    magnetic_mode: str | None = typer.Option(None, "--magnetic-mode"),
    reviewer: str | None = typer.Option(None, "--reviewer"),
    limit: int = typer.Option(10, "--limit"),
) -> None:
    """Read past human input reviews the agent may learn from."""
    try:
        history = load_review_history(
            species=species or None,
            task=task,
            magnetic_mode=magnetic_mode,
            reviewer=reviewer,
            limit=limit,
        )
        typer.echo(
            json.dumps(
                [item.model_dump(mode="json") for item in history],
                indent=2,
                ensure_ascii=False,
            )
        )
    except (ValueError, OSError) as exc:
        _error(exc)
