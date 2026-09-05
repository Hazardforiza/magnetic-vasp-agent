from __future__ import annotations

from pathlib import Path

import typer

from magnetic_vasp_agent.mcif import read_mcif

mcif_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect magnetic candidates with traceable site and moment expansion.",
)


@mcif_app.command("show")
def show(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
    candidate_id: str | None = typer.Option(None, "--candidate-id"),
) -> None:
    """Parse one explicit magnetic cell and print its candidate model as JSON."""
    try:
        typer.echo(read_mcif(path, candidate_id).model_dump_json(indent=2))
    except (ValueError, OSError) as exc:
        typer.echo(f"mCIF error: {exc}", err=True)
        raise typer.Exit(2)
