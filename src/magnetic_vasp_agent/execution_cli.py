# ruff: noqa: B008 - Typer requires option declarations in function signatures.
from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer

from magnetic_vasp_agent.execution import prepare_execution, write_report
from magnetic_vasp_agent.execution_models import (
    ClusterProfile,
    ExecutionRequest,
    ProbeSnapshot,
)
from magnetic_vasp_agent.execution_probe import collect_probe, probe_preview
from magnetic_vasp_agent.execution_simulator import simulate
from magnetic_vasp_agent.workflow import json_bytes

execution_app = typer.Typer(no_args_is_help=True, help="Prepare SSH/direct-MPI plans; remote execution is disabled.")


def _emit(report: dict, output: Path | None) -> None:
    if output:
        write_report(report, output)
    typer.echo(json_bytes(report).decode("utf-8"), nl=False)


def _error(exc: Exception) -> None:
    typer.echo(f"Execution preparation error: {exc}", err=True)
    raise typer.Exit(2)


@execution_app.command("plan")
def plan(
    bundle: Path = typer.Option(..., "--bundle", exists=True, file_okay=False),
    profile: Path = typer.Option(..., "--profile", exists=True, dir_okay=False),
    request: Path = typer.Option(..., "--request", exists=True, dir_okay=False),
    snapshot: Path | None = typer.Option(None, "--snapshot", exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Offline input-integrity check and blocked execution preview. Never connects to SSH."""
    try:
        _emit(prepare_execution(
            bundle, ClusterProfile.model_validate_json(profile.read_bytes()),
            ExecutionRequest.model_validate_json(request.read_bytes()),
            ProbeSnapshot.model_validate_json(snapshot.read_bytes()) if snapshot else None,
        ), output)
    except (ValueError, OSError) as exc:
        _error(exc)


@execution_app.command("probe")
def probe(
    profile: Path = typer.Option(..., "--profile", exists=True, dir_okay=False),
    node: str = typer.Option(..., "--node"),
    connect: bool = typer.Option(False, "--connect", help="Explicitly run the fixed read-only SSH probe."),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Preview by default; --connect reads one node without running MPI/VASP."""
    try:
        if output and (output.exists() or output.is_symlink()):
            raise FileExistsError("Report exists; choose a new path before probing")
        settings = ClusterProfile.model_validate_json(profile.read_bytes())
        report = (
            collect_probe(settings, node).model_dump(mode="json") if connect
            else probe_preview(settings, node)
        )
        _emit(report, output)
    except (ValueError, OSError) as exc:
        _error(exc)


class SimulationScenario(str, Enum):
    complete = "complete"
    fail = "fail"
    timeout = "timeout"
    cancel = "cancel"


@execution_app.command("simulate")
def simulation(
    output: Path = typer.Option(..., "--output"),
    scenario: SimulationScenario = typer.Option(SimulationScenario.complete, "--scenario"),
) -> None:
    """Run a fixed tiny local Python process, not VASP, SSH or an arbitrary command."""
    try:
        _emit(simulate(output, scenario.value), None)
    except (ValueError, OSError) as exc:
        _error(exc)
