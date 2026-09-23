# ruff: noqa: B008 - Typer requires option declarations in function signatures.
from __future__ import annotations

import json
from pathlib import Path

import typer

from magnetic_vasp_agent.skills import (
    DEFAULT_SKILL_CATALOG,
    SkillError,
    list_skills,
    read_skill_file,
    show_skill,
)

skill_app = typer.Typer(
    no_args_is_help=True,
    help="List and read project skills. Scripts are never executed.",
)


def _error(exc: Exception) -> None:
    typer.echo(f"Skill error: {exc}", err=True)
    raise typer.Exit(2)


@skill_app.command("list")
def list_command(
    catalog: Path = typer.Option(DEFAULT_SKILL_CATALOG, "--catalog", exists=True, dir_okay=False),
) -> None:
    """Show enabled skills and their reference files."""
    try:
        payload = {
            "skills": [item.model_dump(mode="json") for item in list_skills(catalog)],
            "scripts_executable": False,
        }
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    except (SkillError, ValueError, OSError) as exc:
        _error(exc)


@skill_app.command("show")
def show_command(
    skill_id: str = typer.Option(..., "--id"),
    catalog: Path = typer.Option(DEFAULT_SKILL_CATALOG, "--catalog", exists=True, dir_okay=False),
) -> None:
    """Print one SKILL.md body."""
    try:
        typer.echo(show_skill(catalog, skill_id).model_dump_json(indent=2))
    except (SkillError, ValueError, OSError) as exc:
        _error(exc)


@skill_app.command("read")
def read_command(
    skill_id: str = typer.Option(..., "--id"),
    path: str = typer.Option(..., "--path"),
    catalog: Path = typer.Option(DEFAULT_SKILL_CATALOG, "--catalog", exists=True, dir_okay=False),
) -> None:
    """Read a Markdown or text file inside a skill directory."""
    try:
        typer.echo(read_skill_file(catalog, skill_id, path).model_dump_json(indent=2))
    except (SkillError, ValueError, OSError) as exc:
        _error(exc)
