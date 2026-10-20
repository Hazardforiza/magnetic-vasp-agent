"""Command-line summaries for magnetic result sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from magnetic_vasp_agent.magnetic_analysis import (
    compare_candidate_energies,
    fit_heisenberg,
    summarize_mae,
    summarize_parameter_sensitivity,
)

analysis_app = typer.Typer(no_args_is_help=True, help="Compare magnetic VASP result sets.")


def _load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("input JSON must contain a list of result records")
    return payload


def _emit(value: Any) -> None:
    typer.echo(json.dumps(value.model_dump(mode="json"), indent=2, ensure_ascii=False))


@analysis_app.command("compare")
def compare(path: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Rank compatible candidate records using sigma->0 energies where available."""
    try:
        _emit(compare_candidate_energies(_load(path)))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        typer.echo(f"Analysis error: {exc}", err=True)
        raise typer.Exit(2) from exc


@analysis_app.command("mae")
def mae(path: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Summarize directional energies; compatibility is the caller's responsibility."""
    try:
        _emit(summarize_mae(_load(path)))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        typer.echo(f"Analysis error: {exc}", err=True)
        raise typer.Exit(2) from exc


@analysis_app.command("exchange")
def exchange(path: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Fit a one-parameter energy/correlation model and report residuals."""
    try:
        _emit(fit_heisenberg(_load(path)))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        typer.echo(f"Analysis error: {exc}", err=True)
        raise typer.Exit(2) from exc


@analysis_app.command("sensitivity")
def sensitivity(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
    parameter: str = typer.Option("U", "--parameter"),
) -> None:
    """Group result records by a declared model parameter such as U or strain."""
    try:
        _emit(summarize_parameter_sensitivity(_load(path), parameter=parameter))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        typer.echo(f"Analysis error: {exc}", err=True)
        raise typer.Exit(2) from exc

