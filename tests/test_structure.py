from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from magnetic_vasp_agent.models import CandidateComposition, CoordinateMode, Structure
from magnetic_vasp_agent.poscar import (
    PoscarFormatError,
    parse_poscar,
    read_poscar,
    render_poscar,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_mn3al_direct_with_velocities() -> None:
    structure = read_poscar(FIXTURES / "Mn3Al" / "POSCAR")
    assert structure.comment == "Mn3Al"
    assert structure.scale == pytest.approx(5.8)
    assert structure.lattice[0] == pytest.approx([0.0, 0.5, 0.5])
    assert structure.species_order == ["Mn", "Al"]
    assert structure.counts == [3, 1]
    assert structure.coordinate_mode == CoordinateMode.DIRECT
    assert structure.num_atoms == 4
    assert structure.positions[0] == pytest.approx([0.25, 0.25, 0.25])
    assert structure.selective_dynamics is None
    assert structure.velocities is not None and len(structure.velocities) == 4
    assert structure.site_species == ["Mn", "Mn", "Mn", "Al"]
    assert structure.formula == "Mn3Al"


def test_cumnas_cartesian_alphabetical_species() -> None:
    structure = read_poscar(FIXTURES / "CuMnAs" / "POSCAR")
    assert structure.species_order == ["As", "Cu", "Mn"]
    assert structure.counts == [2, 2, 2]
    assert structure.coordinate_mode == CoordinateMode.CARTESIAN
    assert structure.lattice[1][1] == pytest.approx(6.3218)
    assert structure.positions[5] == pytest.approx([2.8503, 2.12602134, 2.8503])
    assert structure.velocities is None
    assert structure.formula == "As2Cu2Mn2"


def test_srruo3_monoclinic_twenty_atoms() -> None:
    structure = read_poscar(FIXTURES / "SrRuO3" / "POSCAR")
    assert structure.species_order == ["Sr", "Ru", "O"]
    assert structure.counts == [4, 4, 12]
    assert structure.num_atoms == 20
    assert structure.lattice[2][0] == pytest.approx(0.0581882646634546)
    assert structure.site_species[4:8] == ["Ru"] * 4
    assert structure.site_species[8:] == ["O"] * 12


def test_selective_dynamics_flags() -> None:
    text = (
        "slab\n1.0\n3 0 0\n0 3 0\n0 0 10\nFe O\n1 1\nSelective dynamics\nDirect\n"
        "0 0 0 T T F\n0.5 0.5 0.5 F F F\n"
    )
    structure = parse_poscar(text)
    assert structure.selective_dynamics == [[True, True, False], [False, False, False]]
    assert "Selective dynamics" in render_poscar(structure)


def test_vasp4_without_species_line_is_rejected() -> None:
    text = "old\n1.0\n3 0 0\n0 3 0\n0 0 3\n3 1\nDirect\n0 0 0\n0.5 0.5 0.5\n0 0.5 0.5\n0.5 0 0.5\n"
    with pytest.raises(PoscarFormatError, match="VASP 4"):
        parse_poscar(text)


def test_count_and_position_mismatch_is_rejected() -> None:
    text = "x\n1.0\n3 0 0\n0 3 0\n0 0 3\nMn Al\n3 1\nDirect\n0 0 0\n0.5 0.5 0.5\n"
    with pytest.raises(PoscarFormatError, match="expected 4 coordinate lines"):
        parse_poscar(text)


def test_non_element_species_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not an element symbol"):
        Structure(
            scale=1.0,
            lattice=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            species_order=["Mn_pv"],
            counts=[1],
            coordinate_mode=CoordinateMode.DIRECT,
            positions=[[0, 0, 0]],
        )


@pytest.mark.parametrize("system", ["Mn3Al", "CuMnAs", "SrRuO3"])
def test_render_round_trip(system: str) -> None:
    original = read_poscar(FIXTURES / system / "POSCAR")
    again = parse_poscar(render_poscar(original))
    assert again == original


def test_candidate_from_structure_uses_poscar_order() -> None:
    structure = read_poscar(FIXTURES / "CuMnAs" / "POSCAR")
    candidate = CandidateComposition.from_structure(structure)
    assert candidate.candidate_id == "As2Cu2Mn2"
    assert candidate.species_order == ["As", "Cu", "Mn"]
    named = CandidateComposition.from_structure(structure, "CuMnAs")
    assert named.candidate_id == "CuMnAs"
