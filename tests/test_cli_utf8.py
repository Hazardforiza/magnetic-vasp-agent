from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from magnetic_vasp_agent.cli import app
from magnetic_vasp_agent.cli_io import configure_utf8_stdio

ROOT = Path(__file__).parents[1]
UNICODE_TEXT = "中文磁矩 ✓"


def run_cli(args: list[str], encoding: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-m", "magnetic_vasp_agent.cli", *args],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONIOENCODING": encoding,
            "PYTHONUTF8": "0",
            "PYTHONPATH": str(ROOT / "src"),
            "NO_COLOR": "1",
        },
        capture_output=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("encoding", ["gbk", "cp1252"])
def test_redirected_json_is_utf8_and_preserves_unicode(tmp_path, encoding):
    source = tmp_path / f"{UNICODE_TEXT}.INCAR"
    source.write_text(f"SYSTEM = {UNICODE_TEXT}\n", encoding="utf-8")
    result = run_cli(["incar", "show", str(source)], encoding)
    assert result.returncode == 0, result.stderr.decode("utf-8")
    assert result.stderr == b""
    assert json.loads(result.stdout.decode("utf-8"))["spec"]["SYSTEM"] == UNICODE_TEXT
    assert UNICODE_TEXT.encode("utf-8") in result.stdout
    assert b"\r\n" not in result.stdout


@pytest.mark.parametrize("encoding", ["gbk", "cp1252"])
def test_redirected_command_error_is_utf8_with_no_traceback(tmp_path, encoding):
    result = run_cli(
        [
            "knowledge", "search", "--index", str(tmp_path), "--query", "MAGMOM",
            "--source-types", UNICODE_TEXT,
        ],
        encoding,
    )
    assert result.returncode == 2
    assert result.stdout == b""
    assert f"invalid filter value: {UNICODE_TEXT}" in result.stderr.decode("utf-8")
    assert b"Traceback" not in result.stderr


def test_framework_error_is_utf8_before_any_command_runs():
    result = run_cli([UNICODE_TEXT], "gbk")
    assert result.returncode == 2
    assert result.stdout == b""
    assert UNICODE_TEXT in result.stderr.decode("utf-8")
    assert b"Traceback" not in result.stderr


def test_help_and_typer_capture_still_work():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Local magnetic VASP control plane" in result.output


def test_text_only_capture_stream_is_supported():
    stream = io.StringIO()
    configure_utf8_stdio(stream)
    stream.write(UNICODE_TEXT)
    assert stream.getvalue() == UNICODE_TEXT
