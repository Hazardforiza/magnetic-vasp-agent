from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from magnetic_vasp_agent.cli import app
from magnetic_vasp_agent.agent.tools import default_registry
from magnetic_vasp_agent.vaspkit_plot import (
    parse_band,
    parse_klabels,
    parse_tdos,
    plot_band,
    plot_tdos,
)


TDOS_SPIN = """#Energy        TDOS-UP        TDOS-DOWN
-2.0 0.10 -0.20
-1.0 0.30 -0.40
0.0 0.50 -0.60
1.0 0.20 -0.10
"""

REFORMATTED_BAND = """#K-Path  Band-1  Band-2
0.0 -2.0 1.0
0.5 -1.5 1.5
0.5 -1.4 1.6
1.0 -1.0 2.0
"""

BAND_BLOCKS = """#K-Path     Energy-Level
#Band-index    1
0.0 -2.0
0.5 -1.5
1.0 -1.0

#Band-index    2
0.0 1.0
0.5 1.5
1.0 2.0
"""

KLABELS = """K-Label    K-Coordinate in band-structure plots
GAMMA 0.0
X 0.5
M 1.0

* explanatory footer
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="ascii")
    return path


def test_parse_spin_tdos_exact_schema(tmp_path: Path) -> None:
    table = parse_tdos(_write(tmp_path / "TDOS.dat", TDOS_SPIN))
    assert table.data_format == "vaspkit_tdos_spin_v1"
    assert table.labels == ("TDOS-UP", "TDOS-DOWN")
    assert table.energies == (-2.0, -1.0, 0.0, 1.0)
    assert table.values[1][-1] == pytest.approx(-0.1)


def test_parse_tdos_rejects_unknown_columns_and_non_monotonic_energy(
    tmp_path: Path,
) -> None:
    unknown = _write(tmp_path / "unknown.dat", "#Energy TDOS IDOS\n0 1 2\n1 2 3\n")
    with pytest.raises(ValueError, match="unsupported TDOS header"):
        parse_tdos(unknown)
    descending = _write(tmp_path / "descending.dat", "#Energy TDOS\n1 2\n0 3\n")
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_tdos(descending)


@pytest.mark.parametrize(
    ("filename", "payload", "expected_format"),
    [
        ("REFORMATTED_BAND.dat", REFORMATTED_BAND, "vaspkit_reformatted_band_v1"),
        ("BAND.dat", BAND_BLOCKS, "vaspkit_band_blocks_v1"),
    ],
)
def test_parse_supported_band_layouts(
    tmp_path: Path, filename: str, payload: str, expected_format: str
) -> None:
    table = parse_band(_write(tmp_path / filename, payload))
    assert table.data_format == expected_format
    assert len(table.bands) == 2
    assert table.kpath[0] == 0.0
    assert table.kpath[-1] == 1.0


def test_klabels_must_cover_band_domain(tmp_path: Path) -> None:
    labels = _write(
        tmp_path / "KLABELS",
        "K-Label K-Coordinate\nGAMMA 0.0\nX 0.5\n",
    )
    with pytest.raises(ValueError, match="last KLABELS"):
        parse_klabels(labels, (0.0, 0.5, 1.0))


def test_plot_tdos_writes_png_and_hashes_source_and_plot(tmp_path: Path) -> None:
    source = _write(tmp_path / "TDOS.dat", TDOS_SPIN)
    output = tmp_path / "tdos.png"
    result = plot_tdos(source, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result.plot_kind == "dos"
    assert result.points == 4
    assert result.series == 2
    assert len(result.data.sha256) == 64
    assert len(result.plot.sha256) == 64
    with pytest.raises(FileExistsError):
        plot_tdos(source, output)


def test_plot_band_writes_png_with_label_hash(tmp_path: Path) -> None:
    source = _write(tmp_path / "REFORMATTED_BAND.dat", REFORMATTED_BAND)
    labels = _write(tmp_path / "KLABELS", KLABELS)
    output = tmp_path / "band.png"
    result = plot_band(source, labels, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result.plot_kind == "band"
    assert result.series == 2
    assert result.labels is not None
    assert len(result.labels.sha256) == 64


def test_band_parser_rejects_inconsistent_block_coordinates(tmp_path: Path) -> None:
    malformed = BAND_BLOCKS.replace("0.5 1.5", "0.6 1.5")
    path = _write(tmp_path / "BAND.dat", malformed)
    with pytest.raises(ValueError, match="same k-path"):
        parse_band(path)


def test_agent_plot_tools_are_typed_local_writes(tmp_path: Path) -> None:
    registry = default_registry()
    dos_tool = registry.get("postprocess.plot_dos")
    dos_source = _write(tmp_path / "TDOS.dat", TDOS_SPIN)
    dos_result = dos_tool.run(
        dos_tool.input_model(data=str(dos_source), output=str(tmp_path / "agent-dos.png"))
    )
    assert dos_result["plot_kind"] == "dos"

    band_tool = registry.get("postprocess.plot_band")
    band_source = _write(tmp_path / "REFORMATTED_BAND.dat", REFORMATTED_BAND)
    labels = _write(tmp_path / "KLABELS", KLABELS)
    band_result = band_tool.run(
        band_tool.input_model(
            data=str(band_source),
            labels=str(labels),
            output=str(tmp_path / "agent-band.png"),
        )
    )
    assert band_result["plot_kind"] == "band"


def test_plot_cli_returns_typed_hash_manifest(tmp_path: Path) -> None:
    source = _write(tmp_path / "TDOS.dat", TDOS_SPIN)
    output = tmp_path / "cli-dos.png"
    result = CliRunner().invoke(
        app,
        ["postprocess", "plot-dos", str(source), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["plot_kind"] == "dos"
    assert payload["plot"]["path"] == str(output.resolve())
    assert len(payload["plot"]["sha256"]) == 64
