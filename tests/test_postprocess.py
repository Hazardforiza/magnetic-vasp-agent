from __future__ import annotations

from pathlib import Path

import pytest

from magnetic_vasp_agent.postprocess import (
    VaspOutputError,
    analyze_calculation,
    parse_oszicar,
    parse_outcar,
    read_vasprun,
)
from magnetic_vasp_agent.postprocess_models import DetectedMagneticMode

FIXTURES = Path(__file__).parent / "fixtures" / "postprocess"
COLLINEAR = FIXTURES / "collinear"
SOC = FIXTURES / "soc"

# Optional cross-validation against the real (uncommitted) historical archive.
REAL_ARCHIVE = Path("C:/Users/18355/vasp实例/CuMnAs_vasp/CuMnAs_vasp")


# --------------------------------------------------------------------------- #
# OSZICAR                                                                       #
# --------------------------------------------------------------------------- #


def test_oszicar_collinear_final_energy_and_moment() -> None:
    summary = parse_oszicar(
        (COLLINEAR / "OSZICAR").read_text(encoding="utf-8"),
        source_sha256="0" * 64,
    )
    assert summary.ionic_steps == 1
    assert summary.electronic_steps_last == 2
    assert summary.final_free_energy == pytest.approx(-35.136196, abs=1e-4)
    assert summary.total_magnetization == pytest.approx(0.0, abs=1e-6)
    assert summary.magnetization_vector is None


def test_oszicar_soc_reports_three_component_moment() -> None:
    summary = parse_oszicar(
        (SOC / "OSZICAR").read_text(encoding="utf-8"),
        source_sha256="0" * 64,
    )
    assert summary.magnetization_vector == [pytest.approx(-0.0), 0.0001, 0.0]
    assert summary.total_magnetization is None


def test_oszicar_incomplete_run_has_no_ionic_step() -> None:
    text = (
        "       N       E             dE\n"
        "DAV:   1     0.9E+03    0.9E+03   0.2E+03\n"
        "DAV:   2     0.5E+02   -0.9E+03   0.5E+02\n"
    )
    summary = parse_oszicar(text, source_sha256="0" * 64)
    assert summary.ionic_steps == 0
    assert summary.final_free_energy is None
    assert summary.electronic_steps_last == 2


# --------------------------------------------------------------------------- #
# OUTCAR                                                                        #
# --------------------------------------------------------------------------- #


def test_outcar_collinear_moments_and_mode() -> None:
    summary = parse_outcar(
        (COLLINEAR / "OUTCAR").read_text(encoding="utf-8"),
        source_sha256="0" * 64,
    )
    assert summary.magnetic_mode == DetectedMagneticMode.COLLINEAR
    assert summary.electronic_converged is True
    assert summary.free_energy_toten == pytest.approx(-35.13619563, abs=1e-6)
    assert summary.fermi_energy == pytest.approx(7.5281, abs=1e-4)
    moments = {ion.index: ion.moment for ion in summary.ion_magnetizations}
    assert moments[5] == pytest.approx(3.675, abs=1e-3)
    assert moments[6] == pytest.approx(-3.675, abs=1e-3)
    assert summary.total_magnetization == pytest.approx(0.0, abs=1e-6)
    # max |F| on ions 5/6: sqrt(0.3^2 + 0.4^2) = 0.5
    assert summary.max_force == pytest.approx(0.5, abs=1e-6)


def test_outcar_soc_builds_vector_moments() -> None:
    summary = parse_outcar(
        (SOC / "OUTCAR").read_text(encoding="utf-8"),
        source_sha256="0" * 64,
    )
    assert summary.magnetic_mode == DetectedMagneticMode.SOC
    by_index = {ion.index: ion for ion in summary.ion_magnetizations}
    mn = by_index[5]
    assert mn.vector == [pytest.approx(3.6), pytest.approx(0.5), pytest.approx(0.1)]
    assert mn.moment == pytest.approx((3.6**2 + 0.5**2 + 0.1**2) ** 0.5, abs=1e-4)
    assert summary.total_magnetization_vector is not None


def test_outcar_orbital_moments_present() -> None:
    summary = parse_outcar(
        (COLLINEAR / "OUTCAR").read_text(encoding="utf-8"),
        source_sha256="0" * 64,
    )
    mn = next(ion for ion in summary.ion_magnetizations if ion.index == 5)
    assert mn.orbital_moments is not None
    assert mn.orbital_moments["d"] == pytest.approx(3.598, abs=1e-3)


def test_outcar_static_ediff_stop_is_electronic_convergence() -> None:
    text = """
 ISPIN = 2
 LNONCOLLINEAR = F
 LSORBIT = F
 NELM = 300
 NSW = 0
 NIONS = 1
 NELECT = 2.0
 aborting loop because EDIFF is reached
"""
    summary = parse_outcar(text, source_sha256="0" * 64)
    assert summary.electronic_converged is True


# --------------------------------------------------------------------------- #
# vasprun.xml                                                                   #
# --------------------------------------------------------------------------- #


def test_vasprun_dos_summary_metallic() -> None:
    summary = read_vasprun(COLLINEAR / "vasprun.xml")
    assert summary.ispin == 2
    assert summary.electronic_converged is True
    assert summary.final_free_energy == pytest.approx(-35.13619563, abs=1e-6)
    assert summary.dos is not None
    assert summary.dos.is_spin_polarized is True
    assert summary.dos.dos_at_fermi == [pytest.approx(0.6), pytest.approx(0.6)]
    assert summary.dos.estimated_gap is None  # finite DOS at E_F -> metallic


# --------------------------------------------------------------------------- #
# Combined result                                                               #
# --------------------------------------------------------------------------- #


def test_analyze_collinear_directory() -> None:
    result = analyze_calculation(COLLINEAR)
    assert result.magnetic_mode == DetectedMagneticMode.COLLINEAR
    assert result.converged is True
    assert result.final_energy == pytest.approx(-35.13619563, abs=1e-6)
    assert result.energy_per_atom == pytest.approx(-35.13619563 / 6, abs=1e-6)
    # Species come from the POSCAR in the directory.
    species = {ion.index: ion.species for ion in result.ion_magnetizations}
    assert species[1] == "As"
    assert species[5] == "Mn"
    # All three sources agree.
    assert result.cross_checks
    assert all(check.consistent for check in result.cross_checks)
    assert not result.warnings


def test_analyze_result_hash_is_deterministic() -> None:
    first = analyze_calculation(COLLINEAR)
    second = analyze_calculation(COLLINEAR)
    assert first.result_sha256 == second.result_sha256
    assert len(first.result_sha256) == 64


def test_analyze_soc_directory_vector_moments() -> None:
    result = analyze_calculation(SOC)
    assert result.magnetic_mode == DetectedMagneticMode.SOC
    assert result.total_magnetization_vector is not None
    mn = next(ion for ion in result.ion_magnetizations if ion.index == 5)
    assert mn.vector is not None


def test_analyze_empty_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(VaspOutputError):
        analyze_calculation(tmp_path)


def test_analyze_missing_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(VaspOutputError):
        analyze_calculation(tmp_path / "does_not_exist")


# --------------------------------------------------------------------------- #
# Real-data cross-validation (skipped if the local archive is absent)          #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not (REAL_ARCHIVE / "Mn_pv_nosoc" / "scf" / "OUTCAR").is_file(),
    reason="historical VASP archive not present on this machine",
)
def test_real_collinear_sources_agree() -> None:
    result = analyze_calculation(REAL_ARCHIVE / "Mn_pv_nosoc" / "scf")
    assert result.magnetic_mode == DetectedMagneticMode.COLLINEAR
    assert result.converged is True
    # OUTCAR TOTEN, OSZICAR F=, and vasprun e_fr_energy must all agree.
    energy_check = next(c for c in result.cross_checks if c.quantity == "final_energy")
    assert energy_check.consistent
    assert len(energy_check.sources) == 3
    # CuMnAs is an antiferromagnet: large opposite Mn moments, ~zero net moment.
    mn_moments = sorted(
        ion.moment for ion in result.ion_magnetizations if ion.species == "Mn"
    )
    assert mn_moments[0] == pytest.approx(-mn_moments[-1], abs=1e-2)
    assert abs(mn_moments[-1]) > 3.0


@pytest.mark.skipif(
    not (REAL_ARCHIVE / "Mn_pv_soc" / "scf" / "OUTCAR").is_file(),
    reason="historical VASP archive not present on this machine",
)
def test_real_soc_is_noncollinear() -> None:
    result = analyze_calculation(REAL_ARCHIVE / "Mn_pv_soc" / "scf")
    assert result.magnetic_mode == DetectedMagneticMode.SOC
    assert result.total_magnetization_vector is not None
    assert result.converged is True
