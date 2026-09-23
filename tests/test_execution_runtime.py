from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from magnetic_vasp_agent.cli import app
from magnetic_vasp_agent.execution_runtime import (
    ExecutionRuntimeError,
    cancel,
    launch,
    read_run,
    status,
)
from magnetic_vasp_agent.execution_runtime_models import (
    LOCAL_CONFIRM,
    REMOTE_CONFIRM,
    JobState,
    LaunchBackend,
    LaunchRequest,
)


def _local(output: Path, scenario: str = "complete") -> LaunchRequest:
    return LaunchRequest(
        backend=LaunchBackend.LOCAL_SIMULATION,
        confirm=LOCAL_CONFIRM,
        output=str(output),
        scenario=scenario,
        run_id="sim1",
    )


def test_local_launch_requires_exact_phrase(tmp_path: Path) -> None:
    with pytest.raises(ExecutionRuntimeError, match="I_CONFIRM_LOCAL_LAUNCH"):
        launch(
            LaunchRequest(
                backend=LaunchBackend.LOCAL_SIMULATION,
                confirm="yes",
                output=str(tmp_path / "denied"),
            )
        )


def test_local_launch_complete_writes_hashed_record(tmp_path: Path) -> None:
    record = launch(_local(tmp_path / "ok"))
    assert record.backend == LaunchBackend.LOCAL_SIMULATION
    assert record.state == JobState.COMPLETED_UNVERIFIED
    assert record.cluster_execution == "disabled"
    assert record.scientific_converged is None
    assert record.owned_locally is True
    assert len(record.record_sha256) == 64
    assert status(tmp_path / "ok").record_sha256 == record.record_sha256


def test_local_launch_fail_and_cancel_scenarios(tmp_path: Path) -> None:
    failed = launch(_local(tmp_path / "fail", "fail"))
    assert failed.state == JobState.FAILED
    cancelled = launch(_local(tmp_path / "cancel", "cancel"))
    assert cancelled.state == JobState.CANCELLED


def test_cancel_refuses_remote_or_unowned(tmp_path: Path) -> None:
    record = launch(
        LaunchRequest(
            backend=LaunchBackend.REMOTE_MPI,
            confirm=REMOTE_CONFIRM,
            output=str(tmp_path / "remote"),
            run_id="remote1",
        )
    )
    assert record.state == JobState.BLOCKED
    assert "REMOTE_EXECUTION_DISABLED" in record.blockers
    with pytest.raises(ExecutionRuntimeError, match="refused"):
        cancel(tmp_path / "remote")


def test_cancel_finished_local_run_is_idempotent(tmp_path: Path) -> None:
    launch(_local(tmp_path / "done"))
    again = cancel(tmp_path / "done")
    assert again.state == JobState.COMPLETED_UNVERIFIED


def test_cli_remote_launch_exits_blocked(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "execution",
            "launch",
            "--backend",
            "remote_mpi",
            "--confirm",
            REMOTE_CONFIRM,
            "--output",
            str(tmp_path / "cli-remote"),
        ],
    )
    assert result.exit_code == 1
    assert "REMOTE_EXECUTION_DISABLED" in result.output
    assert '"cluster_execution": "disabled"' in result.output


def test_cli_local_launch_and_status(tmp_path: Path) -> None:
    out = tmp_path / "cli-local"
    launched = CliRunner().invoke(
        app,
        [
            "execution",
            "launch",
            "--backend",
            "local_simulation",
            "--confirm",
            LOCAL_CONFIRM,
            "--output",
            str(out),
        ],
    )
    assert launched.exit_code == 0
    shown = CliRunner().invoke(app, ["execution", "status", "--run", str(out)])
    assert shown.exit_code == 0
    assert read_run(out).state == JobState.COMPLETED_UNVERIFIED
