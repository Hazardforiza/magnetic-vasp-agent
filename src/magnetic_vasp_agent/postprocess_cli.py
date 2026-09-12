# ruff: noqa: B008 - Typer requires option declarations in function signatures.
from __future__ import annotations

from pathlib import Path

import typer

from magnetic_vasp_agent.postprocess import analyze_calculation
from magnetic_vasp_agent.postprocess_models import CalculationResult

postprocess_app = typer.Typer(
    no_args_is_help=True,
    help="Parse VASP outputs (OSZICAR/OUTCAR/vasprun.xml) into typed, hashed results.",
)


def _error(exc: Exception) -> None:
    typer.echo(f"Post-processing error: {exc}", err=True)
    raise typer.Exit(2)


@postprocess_app.command("parse")
def parse_calculation(
    directory: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path | None = typer.Option(
        None, "--output", help="New JSON file; existing outputs are refused."
    ),
) -> None:
    """Parse a calculation directory and emit the typed result as JSON."""
    try:
        if output is not None and (output.exists() or output.is_symlink()):
            raise FileExistsError("output exists; choose a new path")
        result = analyze_calculation(directory)
        payload = result.model_dump_json(indent=2, exclude_none=True)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payload + "\n")
        typer.echo(payload)
    except (ValueError, OSError) as exc:
        _error(exc)
    if not result.converged:
        raise typer.Exit(1)


@postprocess_app.command("show")
def show_calculation(
    directory: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Print a compact human-readable summary of a calculation."""
    try:
        result = analyze_calculation(directory)
        for line in _render_summary(result):
            typer.echo(line)
    except (ValueError, OSError) as exc:
        _error(exc)
    if not result.converged:
        raise typer.Exit(1)


def _render_summary(result: CalculationResult) -> list[str]:
    lines = [
        f"directory        : {result.directory}",
        f"magnetic mode    : {result.magnetic_mode.value}",
        f"converged        : {'yes' if result.converged else 'NO'}",
        f"final energy     : {_fmt(result.final_energy)} eV",
        f"energy per atom  : {_fmt(result.energy_per_atom)} eV",
        f"Fermi energy     : {_fmt(result.fermi_energy)} eV",
    ]
    if result.total_magnetization_vector is not None:
        vec = ", ".join(_fmt(v) for v in result.total_magnetization_vector)
        lines.append(f"total moment (v) : [{vec}] muB")
    else:
        lines.append(f"total moment     : {_fmt(result.total_magnetization)} muB")
    if result.max_force is not None:
        lines.append(f"max force        : {_fmt(result.max_force)} eV/A")

    if result.ion_magnetizations:
        lines.append("local moments (muB):")
        for ion in result.ion_magnetizations:
            label = ion.species or "?"
            if ion.vector is not None:
                vec = ", ".join(_fmt(v) for v in ion.vector)
                lines.append(
                    f"  ion {ion.index:>3} {label:<2} |m|={_fmt(ion.moment)} [{vec}]"
                )
            else:
                lines.append(f"  ion {ion.index:>3} {label:<2} m={_fmt(ion.moment)}")

    if result.vasprun is not None and result.vasprun.dos is not None:
        dos = result.vasprun.dos
        at_ef = ", ".join(_fmt(v) for v in dos.dos_at_fermi)
        gap = "metallic" if dos.estimated_gap is None else f"{_fmt(dos.estimated_gap)} eV"
        lines.append(f"DOS at E_F       : [{at_ef}] (gap: {gap})")

    for check in result.cross_checks:
        status = "ok" if check.consistent else "MISMATCH"
        lines.append(
            f"cross-check {check.quantity}: {status} "
            f"(max diff {check.max_abs_diff:.4g})"
        )
    for warning in result.warnings:
        lines.append(f"warning          : {warning}")
    lines.append(f"result sha256    : {result.result_sha256}")
    return lines


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"
