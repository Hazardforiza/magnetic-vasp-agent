"""Create auditable electronic-structure artifacts for a completed E2E run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.io.vasp.outputs import Vasprun

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


PLOT_MIN_EV = -6.0
PLOT_MAX_EV = 6.0
UP_COLOR = "#006d77"
DOWN_COLOR = "#c44536"
FE_D_COLOR = "#7a5195"
O_P_COLOR = "#2a9d8f"
GRID_COLOR = "#c9ced6"
FERMI_COLOR = "#3f4752"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(text)


def _save_new(figure: plt.Figure, path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"plot output exists: {path}")
    figure.savefig(
        path,
        format="png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "magnetic-vasp-agent"},
    )
    plt.close(figure)
    if path.stat().st_size == 0:
        raise RuntimeError(f"empty plot output: {path}")


def _verified_fetch(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    destination = Path(report["destination"])
    checked: dict[str, dict[str, Any]] = {}
    for name, expected in report["files"].items():
        path = destination / name
        actual = _file_record(path)
        if actual["size_bytes"] != expected["size"]:
            raise ValueError(f"size mismatch for {path}")
        if actual["sha256"] != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}")
        checked[name] = actual
    return {
        "report": str(report_path.resolve()),
        "run_id": report["run_id"],
        "node": report["node"],
        "remote_hashes_verified": report["remote_hashes_verified"],
        "local_hashes_verified": True,
        "files": checked,
    }


def _format_tick(label: str) -> str:
    parts = label.split("|")
    rendered = [r"\Gamma" if part.upper() == "GAMMA" else part for part in parts]
    return "$" + "|".join(rendered) + "$"


def _branch_ticks(band_structure: Any) -> list[tuple[float, str]]:
    ticks: list[tuple[float, list[str]]] = []
    for branch in band_structure.branches:
        for index in (branch["start_index"], branch["end_index"]):
            coordinate = float(band_structure.distance[index])
            label = band_structure.kpoints[index].label or "?"
            if ticks and abs(ticks[-1][0] - coordinate) < 1.0e-10:
                if label not in ticks[-1][1]:
                    ticks[-1][1].append(label)
            else:
                ticks.append((coordinate, [label]))
    return [(coordinate, "|".join(labels)) for coordinate, labels in ticks]


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3.5, width=0.7)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.65)


def _plot_dos(
    energies: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    output: Path,
) -> None:
    mask = (energies >= PLOT_MIN_EV) & (energies <= PLOT_MAX_EV)
    figure, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    axis.plot(energies[mask], up[mask], color=UP_COLOR, linewidth=1.2, label="Spin up")
    axis.plot(
        energies[mask],
        -down[mask],
        color=DOWN_COLOR,
        linewidth=1.2,
        label="Spin down",
    )
    axis.fill_between(energies[mask], 0.0, up[mask], color=UP_COLOR, alpha=0.13)
    axis.fill_between(energies[mask], 0.0, -down[mask], color=DOWN_COLOR, alpha=0.13)
    axis.axvline(0.0, color=FERMI_COLOR, linewidth=0.9, linestyle="--")
    axis.axhline(0.0, color=FERMI_COLOR, linewidth=0.65)
    axis.set_xlim(PLOT_MIN_EV, PLOT_MAX_EV)
    bound = max(float(up[mask].max()), float(down[mask].max())) * 1.08
    axis.set_ylim(-bound, bound)
    axis.set_xlabel(r"Energy - $E_F^{DOS}$ (eV)")
    axis.set_ylabel("DOS (states/eV/cell)")
    axis.set_title(r"FeO spin-resolved total DOS: AFM, $U_{eff}=5$ eV")
    axis.legend(frameon=False, ncols=2, loc="upper right")
    _style_axis(axis)
    _save_new(figure, output)


def _plot_pdos(
    energies: np.ndarray,
    fe_d: np.ndarray,
    o_p: np.ndarray,
    output: Path,
) -> None:
    mask = (energies >= PLOT_MIN_EV) & (energies <= PLOT_MAX_EV)
    figure, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    axis.plot(energies[mask], fe_d[mask], color=FE_D_COLOR, linewidth=1.25, label="Fe 3d")
    axis.plot(energies[mask], o_p[mask], color=O_P_COLOR, linewidth=1.25, label="O 2p")
    axis.fill_between(energies[mask], 0.0, fe_d[mask], color=FE_D_COLOR, alpha=0.10)
    axis.fill_between(energies[mask], 0.0, o_p[mask], color=O_P_COLOR, alpha=0.10)
    axis.axvline(0.0, color=FERMI_COLOR, linewidth=0.9, linestyle="--")
    axis.set_xlim(PLOT_MIN_EV, PLOT_MAX_EV)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel(r"Energy - $E_F^{DOS}$ (eV)")
    axis.set_ylabel("DOS (states/eV/cell)")
    axis.set_title(r"FeO spin-summed orbital-projected DOS: AFM, $U_{eff}=5$ eV")
    axis.legend(frameon=False, ncols=2, loc="upper right")
    _style_axis(axis)
    _save_new(figure, output)


def _draw_bands(axis: plt.Axes, band_structure: Any) -> None:
    for branch in band_structure.branches:
        section = slice(branch["start_index"], branch["end_index"] + 1)
        x = np.asarray(band_structure.distance[section])
        for values in band_structure.bands[Spin.up]:
            axis.plot(x, values[section] - band_structure.efermi, color=UP_COLOR, linewidth=0.65)
        for values in band_structure.bands[Spin.down]:
            axis.plot(
                x,
                values[section] - band_structure.efermi,
                color=DOWN_COLOR,
                linewidth=0.55,
                linestyle=(0, (2.2, 1.8)),
                alpha=0.72,
            )
    ticks = _branch_ticks(band_structure)
    for coordinate, _ in ticks[1:-1]:
        axis.axvline(coordinate, color=GRID_COLOR, linewidth=0.65)
    axis.axhline(0.0, color=FERMI_COLOR, linewidth=0.9, linestyle="--")
    axis.set_xlim(float(band_structure.distance[0]), float(band_structure.distance[-1]))
    axis.set_ylim(PLOT_MIN_EV, PLOT_MAX_EV)
    axis.set_xticks([coordinate for coordinate, _ in ticks])
    axis.set_xticklabels([_format_tick(label) for _, label in ticks])
    axis.set_ylabel(r"Energy - $E_F^{SCF}$ (eV)")
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3.5, width=0.7)


def _plot_bands(band_structure: Any, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    _draw_bands(axis, band_structure)
    axis.set_xlabel("Magnetic-cell wave-vector path")
    axis.set_title(r"FeO electronic bands: AFM, $U_{eff}=5$ eV")
    axis.legend(
        handles=[
            Line2D([0], [0], color=UP_COLOR, linewidth=1.2, label="Spin up"),
            Line2D(
                [0],
                [0],
                color=DOWN_COLOR,
                linewidth=1.0,
                linestyle=(0, (2.2, 1.8)),
                label="Spin down",
            ),
        ],
        frameon=False,
        ncols=2,
        loc="upper right",
    )
    _save_new(figure, output)


def _plot_combined(
    band_structure: Any,
    energies: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    output: Path,
    *,
    dos_fermi_energy: float,
) -> None:
    figure, (band_axis, dos_axis) = plt.subplots(
        1,
        2,
        figsize=(10.2, 5.5),
        sharey=True,
        gridspec_kw={"width_ratios": (3.1, 1.2), "wspace": 0.08},
    )
    _draw_bands(band_axis, band_structure)
    band_axis.set_xlabel("Magnetic-cell wave-vector path")
    band_axis.set_title("Bands")
    band_axis.legend(
        handles=[
            Line2D([0], [0], color=UP_COLOR, linewidth=1.1, label="Spin up"),
            Line2D(
                [0],
                [0],
                color=DOWN_COLOR,
                linewidth=1.0,
                linestyle=(0, (2.2, 1.8)),
                label="Spin down",
            ),
        ],
        frameon=False,
        fontsize=9,
        ncols=2,
        loc="upper right",
    )

    # The band panel is referenced to the SCF Fermi level. Shift the DOS
    # energies to the same zero before sharing the y-axis.
    dos_energies_scf = energies + dos_fermi_energy - band_structure.efermi
    dos_mask = (dos_energies_scf >= PLOT_MIN_EV) & (dos_energies_scf <= PLOT_MAX_EV)
    dos_axis.plot(up[dos_mask], dos_energies_scf[dos_mask], color=UP_COLOR, linewidth=1.05)
    dos_axis.plot(
        -down[dos_mask], dos_energies_scf[dos_mask], color=DOWN_COLOR, linewidth=1.05
    )
    dos_axis.fill_betweenx(
        dos_energies_scf[dos_mask], 0.0, up[dos_mask], color=UP_COLOR, alpha=0.12
    )
    dos_axis.fill_betweenx(
        dos_energies_scf[dos_mask], 0.0, -down[dos_mask], color=DOWN_COLOR, alpha=0.12
    )
    dos_axis.axhline(0.0, color=FERMI_COLOR, linewidth=0.9, linestyle="--")
    dos_axis.axvline(0.0, color=FERMI_COLOR, linewidth=0.65)
    dos_axis.set_xlabel("DOS (states/eV/cell)")
    dos_axis.set_title("Spin-resolved DOS")
    dos_axis.legend(
        handles=[
            Line2D([0], [0], color=UP_COLOR, linewidth=1.1, label="Spin up"),
            Line2D(
                [0],
                [0],
                color=DOWN_COLOR,
                linewidth=1.0,
                label="Spin down",
            ),
        ],
        frameon=False,
        fontsize=9,
        loc="upper right",
    )
    dos_axis.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.65)
    dos_axis.spines["top"].set_visible(False)
    dos_axis.spines["right"].set_visible(False)
    dos_axis.tick_params(direction="out", length=3.5, width=0.7, labelleft=False)
    figure.suptitle(r"FeO electronic structure: AFM, $U_{eff}=5$ eV", y=0.98)
    _save_new(figure, output)


def _plot_dos_near_fermi(
    energies: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    output: Path,
) -> None:
    """Zoom around E_F so a small but finite DOS is not hidden by the full window."""

    window = 0.5
    mask = (energies >= -window) & (energies <= window)
    figure, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    axis.plot(energies[mask], up[mask], color=UP_COLOR, linewidth=1.35, label="Spin up")
    axis.plot(
        energies[mask],
        -down[mask],
        color=DOWN_COLOR,
        linewidth=1.35,
        label="Spin down",
    )
    axis.axvline(0.0, color=FERMI_COLOR, linewidth=0.9, linestyle="--")
    axis.axhline(0.0, color=FERMI_COLOR, linewidth=0.65)
    ef_index = int(np.argmin(np.abs(energies)))
    axis.scatter(
        [energies[ef_index]],
        [up[ef_index]],
        color=UP_COLOR,
        s=24,
        zorder=3,
        label=f"DOS(E_F) = {up[ef_index] + down[ef_index]:.4f} total",
    )
    axis.set_xlim(-window, window)
    bound = max(float(up[mask].max()), float(down[mask].max())) * 1.15
    axis.set_ylim(-bound, bound)
    axis.set_xlabel(r"Energy - $E_F^{DOS}$ (eV)")
    axis.set_ylabel("DOS (states/eV/cell)")
    axis.set_title(r"FeO spin-resolved DOS near $E_F$: finite low-energy weight")
    axis.legend(frameon=False, loc="upper left")
    _style_axis(axis)
    _save_new(figure, output)


def _band_edges(dense_run: Vasprun) -> dict[str, Any]:
    spin_channels = tuple(dense_run.eigenvalues)
    nelect = float(dense_run.parameters["NELECT"])
    if len(spin_channels) != 2:
        raise ValueError("this AFM analysis requires two collinear spin channels")
    nominal_filled = int(round(nelect / 2.0))
    if nominal_filled < 1:
        raise ValueError("invalid nominal filled-band count")

    per_spin: dict[str, Any] = {}
    for spin in spin_channels:
        energies = dense_run.eigenvalues[spin][:, :, 0]
        if energies.shape[1] <= nominal_filled:
            raise ValueError("NBANDS does not include a conduction band")
        valence = energies[:, nominal_filled - 1]
        conduction = energies[:, nominal_filled]
        vbm_index = int(np.argmax(valence))
        cbm_index = int(np.argmin(conduction))
        direct = conduction - valence
        direct_index = int(np.argmin(direct))
        name = "up" if spin == Spin.up else "down"
        per_spin[name] = {
            "vbm_eV": float(valence[vbm_index]),
            "vbm_kpoint_fractional": [float(v) for v in dense_run.actual_kpoints[vbm_index]],
            "cbm_eV": float(conduction[cbm_index]),
            "cbm_kpoint_fractional": [float(v) for v in dense_run.actual_kpoints[cbm_index]],
            "indirect_gap_eV": float(conduction[cbm_index] - valence[vbm_index]),
            "minimum_direct_gap_eV": float(direct[direct_index]),
            "direct_gap_kpoint_fractional": [
                float(v) for v in dense_run.actual_kpoints[direct_index]
            ],
        }
    indirect = min(channel["indirect_gap_eV"] for channel in per_spin.values())
    direct = min(channel["minimum_direct_gap_eV"] for channel in per_spin.values())
    return {
        "method": "nominal band-14/band-15 edges on the 2016-point DOS mesh",
        "nominal_filled_bands_per_spin": nominal_filled,
        "indirect_gap_eV": indirect,
        "minimum_direct_gap_eV": direct,
        "classification": "metallic_or_semimetal" if indirect <= 0.0 else "insulator",
        "per_spin": per_spin,
    }


def _write_tables(
    output: Path,
    dos_energies: np.ndarray,
    dos_up: np.ndarray,
    dos_down: np.ndarray,
    fe_d_up: np.ndarray,
    fe_d_down: np.ndarray,
    o_p_up: np.ndarray,
    o_p_down: np.ndarray,
    band_structure: Any,
) -> dict[str, Path]:
    tdos = output / "TDOS.dat"
    tdos_lines = ["#Energy TDOS-UP TDOS-DOWN"]
    tdos_lines.extend(
        f"{energy:.8f} {up:.8f} {-down:.8f}"
        for energy, up, down in zip(dos_energies, dos_up, dos_down, strict=True)
    )
    _write_new(tdos, "\n".join(tdos_lines) + "\n")

    pdos = output / "projected-dos.tsv"
    pdos_lines = ["# Energy-eV\tFe-3d-up\tFe-3d-down\tO-2p-up\tO-2p-down"]
    pdos_lines.extend(
        f"{energy:.8f}\t{fdu:.8f}\t{-fdd:.8f}\t{opu:.8f}\t{-opd:.8f}"
        for energy, fdu, fdd, opu, opd in zip(
            dos_energies,
            fe_d_up,
            fe_d_down,
            o_p_up,
            o_p_down,
            strict=True,
        )
    )
    _write_new(pdos, "\n".join(pdos_lines) + "\n")

    band = output / "REFORMATTED_BAND.dat"
    band_columns = []
    for spin in (Spin.up, Spin.down):
        prefix = "UP" if spin == Spin.up else "DOWN"
        band_columns.extend(f"{prefix}-{index + 1}" for index in range(band_structure.nb_bands))
    band_lines = ["#K-Path " + " ".join(band_columns)]
    for point_index, coordinate in enumerate(band_structure.distance):
        values: list[float] = []
        for spin in (Spin.up, Spin.down):
            values.extend(
                float(band_structure.bands[spin][band_index, point_index] - band_structure.efermi)
                for band_index in range(band_structure.nb_bands)
            )
        band_lines.append(
            f"{coordinate:.10f} " + " ".join(f"{value:.8f}" for value in values)
        )
    _write_new(band, "\n".join(band_lines) + "\n")

    labels = output / "KLABELS"
    label_lines = ["K-Label K-Coordinate"]
    label_lines.extend(f"{label} {coordinate:.10f}" for coordinate, label in _branch_ticks(band_structure))
    _write_new(labels, "\n".join(label_lines) + "\n")
    return {"tdos": tdos, "pdos": pdos, "band": band, "labels": labels}


def analyze(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory exists: {output}")
    output.mkdir(parents=True)

    scf_result_path = root / "scf-result-final.json"
    if not scf_result_path.is_file():
        scf_result_path = root / "scf-result.json"
    scf_result = json.loads(scf_result_path.read_text(encoding="utf-8"))
    scf_fermi = float(scf_result["vasprun"]["fermi_energy"])

    verified_fetches = {
        "dos": _verified_fetch(root / "dos-r2-fetch-report.json"),
        "band": _verified_fetch(root / "band-r1-fetch-report.json"),
    }
    dos_dir = root / "dos-results-r2"
    band_dir = root / "band-results-r1"
    dos_run = Vasprun(
        dos_dir / "vasprun.xml",
        parse_projected_eigen=False,
        parse_potcar_file=False,
        exception_on_bad_xml=True,
    )
    band_run = Vasprun(
        band_dir / "vasprun.xml",
        parse_projected_eigen=False,
        parse_potcar_file=False,
        exception_on_bad_xml=True,
    )
    if not dos_run.converged or not band_run.converged:
        raise ValueError("DOS and band calculations must both be converged")

    complete_dos = dos_run.complete_dos
    dos_energies = np.asarray(complete_dos.energies) - float(complete_dos.efermi)
    dos_up = np.asarray(complete_dos.densities[Spin.up])
    dos_down = np.asarray(complete_dos.densities[Spin.down])
    fe_d = complete_dos.get_element_spd_dos("Fe")[OrbitalType.d]
    o_p = complete_dos.get_element_spd_dos("O")[OrbitalType.p]
    fe_d_up = np.asarray(fe_d.densities[Spin.up])
    fe_d_down = np.asarray(fe_d.densities[Spin.down])
    o_p_up = np.asarray(o_p.densities[Spin.up])
    o_p_down = np.asarray(o_p.densities[Spin.down])

    band_structure = band_run.get_band_structure(
        band_dir / "KPOINTS",
        efermi=scf_fermi,
        line_mode=True,
    )
    tables = _write_tables(
        output,
        dos_energies,
        dos_up,
        dos_down,
        fe_d_up,
        fe_d_down,
        o_p_up,
        o_p_down,
        band_structure,
    )

    plots = {
        "total_dos": output / "total-dos.png",
        "dos_near_fermi": output / "total-dos-near-fermi.png",
        "projected_dos": output / "projected-dos.png",
        "band_structure": output / "band-structure.png",
        "combined": output / "band-dos-combined.png",
    }
    _plot_dos(dos_energies, dos_up, dos_down, plots["total_dos"])
    _plot_dos_near_fermi(dos_energies, dos_up, dos_down, plots["dos_near_fermi"])
    _plot_pdos(
        dos_energies,
        fe_d_up + fe_d_down,
        o_p_up + o_p_down,
        plots["projected_dos"],
    )
    _plot_bands(band_structure, plots["band_structure"])
    _plot_combined(
        band_structure,
        dos_energies,
        dos_up,
        dos_down,
        plots["combined"],
        dos_fermi_energy=float(complete_dos.efermi),
    )

    dos_fermi_index = int(np.argmin(np.abs(dos_energies)))
    band_up = band_structure.bands[Spin.up]
    band_down = band_structure.bands[Spin.down]
    relative_up = band_up - scf_fermi
    plot_mask = (relative_up >= PLOT_MIN_EV) & (relative_up <= PLOT_MAX_EV)
    max_spin_split = float(np.max(np.abs(band_up - band_down)[plot_mask]))
    highest_relative = max(
        float(np.max(band_structure.bands[spin] - scf_fermi))
        for spin in (Spin.up, Spin.down)
    )
    fermi_crossing_bands: list[dict[str, Any]] = []
    for spin in (Spin.up, Spin.down):
        for band_index, values in enumerate(band_structure.bands[spin]):
            low = float(np.min(values - scf_fermi))
            high = float(np.max(values - scf_fermi))
            if low <= 0.0 <= high:
                fermi_crossing_bands.append(
                    {
                        "spin": "up" if spin == Spin.up else "down",
                        "band_index_1_based": band_index + 1,
                        "minimum_relative_eV": low,
                        "maximum_relative_eV": high,
                    }
                )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "system": "Fe2O2 AFM",
        "method": {
            "functional": "DFT+U (Dudarev)",
            "u_eff_Fe_3d_eV": 5.0,
            "soc": False,
            "band_path_scope": "low-symmetry magnetic primitive cell; no unfolding",
        },
        "source_integrity": {
            "scf_result": _file_record(scf_result_path),
            "fetches": verified_fetches,
        },
        "dos": {
            "converged": bool(dos_run.converged),
            "fermi_energy_eV": float(complete_dos.efermi),
            "plot_energy_reference": "DOS vasprun.xml E_F",
            "energy_points": int(len(dos_energies)),
            "dos_at_fermi_states_per_eV_per_cell": {
                "up": float(dos_up[dos_fermi_index]),
                "down": float(dos_down[dos_fermi_index]),
                "total": float(dos_up[dos_fermi_index] + dos_down[dos_fermi_index]),
            },
            "max_spin_channel_difference_states_per_eV": float(
                np.max(np.abs(dos_up - dos_down))
            ),
            "spin_plot_convention": "spin-up positive; spin-down mirrored negative",
        },
        "bands": {
            "converged": bool(band_run.converged),
            "fermi_reference": "SCF vasprun.xml",
            "fermi_energy_eV": scf_fermi,
            "band_run_fermi_energy_not_used_eV": float(band_run.efermi),
            "kpoints": len(band_structure.kpoints),
            "bands_per_spin": int(band_structure.nb_bands),
            "branches": [branch["name"] for branch in band_structure.branches],
            "fermi_crossing_bands": fermi_crossing_bands,
            "dense_mesh_edges": _band_edges(dos_run),
            "highest_calculated_energy_above_scf_fermi_eV": highest_relative,
            "max_spin_splitting_in_plot_window_eV": max_spin_split,
            "conduction_coverage_assessment": (
                "sufficient for the displayed +6 eV window"
                if highest_relative >= PLOT_MAX_EV
                else "insufficient for the displayed energy window"
            ),
            "combined_plot_dos_reference_conversion_eV": float(
                complete_dos.efermi - scf_fermi
            ),
        },
        "tables": {name: _file_record(path) for name, path in tables.items()},
        "plots": {name: _file_record(path) for name, path in plots.items()},
        "limitations": [
            "The k-point and Hubbard-U convergence of the scientific result were not established by this E2E test.",
            "The line path belongs to the low-symmetry AFM calculation cell and was not unfolded to the conventional cubic FeO Brillouin zone.",
            "VASPKIT was not executed because no approved Linux/WSL backend was available; tables were derived directly from hash-verified VASP outputs with pymatgen.",
            "The projected-DOS figure sums the two spin channels for Fe 3d and O 2p; it is not a site-resolved spin-PDOS or fat-band result.",
        ],
    }
    summary_path = output / "electronic-structure-summary.json"
    _write_new(summary_path, json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    summary["summary"] = _file_record(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(analyze(arguments.root, arguments.output), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
