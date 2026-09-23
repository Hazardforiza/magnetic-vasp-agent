from __future__ import annotations

from types import SimpleNamespace

import pytest

from magnetic_vasp_agent.execution_models import ClusterProfile, ExecutionRequest
from magnetic_vasp_agent.execution_remote import (
    RemoteExecutionError,
    launch_preview,
    run_mpi_smoke,
    smoke_preview,
    start_remote_vasp,
)
from magnetic_vasp_agent.execution_runtime_models import SMOKE_CONFIRM
from magnetic_vasp_agent.transfer_models import TransferFile, TransferReport


def _profile(**kwargs) -> ClusterProfile:
    values = {
        "execution_enabled": True,
        "remote_root": "/home/cluster_user/magvasp/runs",
    }
    values.update(kwargs)
    return ClusterProfile(**values)


def _runner(stdout: bytes, code: int = 0):
    def run(argv, stdin, timeout=0):
        assert argv[0] == "ssh"
        assert b"cu22" in stdin
        assert b"mpirun" in stdin or b"vasp_" in stdin
        return SimpleNamespace(returncode=code, stdout=stdout, stderr=b"")

    return run


def test_smoke_preview_is_single_host_and_not_vasp() -> None:
    preview = smoke_preview(_profile(), "cu22")
    assert preview["executes_vasp"] is False
    assert "-hosts" in preview["stdin_script"]
    assert "cu22" in preview["stdin_script"]
    assert "vasp_" not in preview["stdin_script"]
    assert "/bin/hostname" in preview["stdin_script"]
    assert "</dev/null" in preview["stdin_script"]
    assert "setvars.sh" in preview["stdin_script"]


def test_smoke_rejects_gateway_and_wrong_phrase() -> None:
    with pytest.raises(RemoteExecutionError, match="gateway"):
        smoke_preview(_profile(gateway_hostname="login01"), "login01")
    with pytest.raises(RemoteExecutionError, match="I_CONFIRM_MPI_SMOKE"):
        run_mpi_smoke(_profile(), "cu22", confirm="yes")
    with pytest.raises(RemoteExecutionError, match="REMOTE_EXECUTION_DISABLED"):
        run_mpi_smoke(ClusterProfile(), "cu22", confirm=SMOKE_CONFIRM)


def test_smoke_passes_with_fixed_hostname_and_cpu_output() -> None:
    report = run_mpi_smoke(
        _profile(),
        "cu22",
        confirm=SMOKE_CONFIRM,
        run_id="run1",
        runner=_runner(b"MAGVASP_SMOKE_V2\ncu22|0\ncu22|1\nMAGVASP_SMOKE_END\n"),
    )
    assert report["passed"] is True
    assert report["run_id"] == "run1"
    assert report["mpi_ranks"] == 2
    assert report["rank_cpu_ids"] == [0, 1]


def test_smoke_fails_on_cross_node_rank() -> None:
    with pytest.raises(RemoteExecutionError, match="SMOKE_CROSS_NODE"):
        run_mpi_smoke(
            _profile(),
            "cu22",
            confirm=SMOKE_CONFIRM,
            runner=_runner(b"MAGVASP_SMOKE_V2\ncu22|0\ncu41|1\nMAGVASP_SMOKE_END\n"),
        )


def test_smoke_fails_when_rank_escapes_approved_cpu_set() -> None:
    with pytest.raises(RemoteExecutionError, match="SMOKE_CPU_BINDING"):
        run_mpi_smoke(
            _profile(),
            "cu22",
            confirm=SMOKE_CONFIRM,
            cpu_ids=[4, 5],
            runner=_runner(
                b"MAGVASP_SMOKE_V2\ncu22|4\ncu22|6\nMAGVASP_SMOKE_END\n"
            ),
        )


def test_remote_launch_uses_transfer_leaf_and_returns_pid() -> None:
    transfer = TransferReport(
        mode="transfer",
        workflow_id="Mn3Al-scf-dos-band",
        run_id="live01",
        node="cu22",
        local_bundle="C:/bundle",
        remote_directory="/home/cluster_user/magvasp/runs/Mn3Al-scf-dos-band--live01",
        files=[
            TransferFile(
                relative_path="scf/INCAR",
                size=10,
                sha256="0" * 64,
            )
        ],
        ready_to_transfer=False,
        transferred=True,
        remote_hashes_verified=True,
        remote_directory_created=True,
        execution_enabled=True,
    )
    request = ExecutionRequest(node="cu22", step_id="scf", run_id="live01", mpi_ranks=2)
    started = start_remote_vasp(
        _profile(),
        request,
        transfer,
        executable="vasp_std",
        cpu_ids=[0, 1],
        runner=_runner(
            b"MAGVASP_LAUNCH_V1\npid=4321\ncwd=/home/cluster_user/magvasp/runs/Mn3Al-scf-dos-band--live01/scf\nhost=cu22\nMAGVASP_LAUNCH_END\n"
        ),
    )
    assert started["remote_pid"] == 4321
    assert started["workdir"].endswith("/scf")
    assert any(item.endswith("vasp_std") for item in started["mpi_argv"])


def test_dependent_launch_rechecks_promoted_chgcar_before_mpi() -> None:
    transfer = TransferReport(
        mode="transfer",
        workflow_id="demo",
        run_id="dos01",
        node="cu22",
        local_bundle="C:/bundle",
        remote_directory="/home/cluster_user/magvasp/runs/demo--dos01",
        files=[TransferFile(relative_path="dos/INCAR", size=1, sha256="0" * 64)],
        ready_to_transfer=False,
        transferred=True,
        remote_hashes_verified=True,
        remote_directory_created=True,
        execution_enabled=True,
    )
    preview = launch_preview(
        _profile(),
        ExecutionRequest(node="cu22", step_id="dos", run_id="dos01", mpi_ranks=2),
        transfer,
        executable="vasp_std",
        cpu_ids=[0, 1],
        artifact_sha256="a" * 64,
        artifact_size=1234,
    )
    script = preview["stdin_script"]
    artifact_check = script.index("sha256sum --")
    mpi_launch = script.index("nohup ")
    assert "/CHGCAR" in script
    assert ' = 1234' in script
    assert "a" * 64 in script
    assert artifact_check < mpi_launch


@pytest.mark.parametrize(
    ("artifact_sha256", "artifact_size"),
    [("a" * 64, None), (None, 1), ("invalid", 1), ("a" * 64, 0)],
)
def test_launch_refuses_incomplete_artifact_evidence(
    artifact_sha256: str | None, artifact_size: int | None
) -> None:
    transfer = TransferReport(
        mode="transfer",
        workflow_id="demo",
        run_id="dos01",
        node="cu22",
        local_bundle="C:/bundle",
        remote_directory="/home/cluster_user/magvasp/runs/demo--dos01",
        files=[TransferFile(relative_path="dos/INCAR", size=1, sha256="0" * 64)],
        ready_to_transfer=False,
        transferred=True,
        remote_hashes_verified=True,
        remote_directory_created=True,
        execution_enabled=True,
    )
    with pytest.raises(RemoteExecutionError, match="ARTIFACT_EVIDENCE"):
        launch_preview(
            _profile(),
            ExecutionRequest(node="cu22", step_id="dos", run_id="dos01", mpi_ranks=2),
            transfer,
            executable="vasp_std",
            cpu_ids=[0, 1],
            artifact_sha256=artifact_sha256,
            artifact_size=artifact_size,
        )
