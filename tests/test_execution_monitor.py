from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import magnetic_vasp_agent.execution_monitor as monitor
from magnetic_vasp_agent.execution_models import ClusterProfile
from magnetic_vasp_agent.execution_runtime_models import (
    JobState,
    LaunchBackend,
    RunRecord,
)


def _profile(**updates: object) -> ClusterProfile:
    values: dict[str, object] = {
        "execution_enabled": True,
        "remote_root": "/home/cluster_user/magvasp/runs",
    }
    values.update(updates)
    return ClusterProfile(**values)


def _stamp(record: RunRecord) -> RunRecord:
    payload = record.model_dump(mode="json")
    payload.pop("record_sha256")
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    return record.model_copy(
        update={"record_sha256": hashlib.sha256(encoded).hexdigest()}
    )


def _record(**updates: object) -> RunRecord:
    values: dict[str, object] = {
        "backend": LaunchBackend.REMOTE_MPI,
        "state": JobState.RUNNING,
        "run_id": "run01",
        "output_dir": "C:/audit/run01",
        "events": ["prepared", "running"],
        "owned_locally": False,
        "cluster_execution": "remote_mpi",
        "node": "node03",
        "remote_directory": "/home/cluster_user/magvasp/runs/demo--run01/scf",
        "remote_pid": 4321,
        "executable": "vasp_std",
    }
    values.update(updates)
    return _stamp(RunRecord(**values))


def _status_payload(
    *,
    alive: bool = True,
    cwd: str | None = None,
    kinds: dict[str, tuple[str, int]] | None = None,
    tails: dict[str, bytes] | None = None,
) -> bytes:
    cwd = cwd or ("yes" if alive else "unknown")
    kinds = kinds or {
        name: (("regular", 12) if name in {"OUTCAR", "OSZICAR"} else ("missing", 0))
        for name in monitor.STATUS_FILES
    }
    tails = tails or {"OUTCAR": b"", "OSZICAR": b""}
    lines = [
        "MAGVASP_STATUS_V1",
        f"pid_alive={'yes' if alive else 'no'}",
        f"pid_cwd_match={cwd}",
    ]
    lines.extend(
        f"file={name}|{kinds[name][0]}|{kinds[name][1]}"
        for name in monitor.STATUS_FILES
    )
    lines.extend(
        f"tail={name}|{base64.b64encode(tails[name]).decode('ascii')}"
        for name in ("OUTCAR", "OSZICAR")
    )
    lines.append("MAGVASP_STATUS_END")
    return ("\n".join(lines) + "\n").encode("ascii")


def _inventory_payload(contents: dict[str, bytes]) -> bytes:
    lines = ["MAGVASP_FETCH_V1"]
    lines.extend(
        f"file={name}|{len(value)}|{hashlib.sha256(value).hexdigest()}"
        for name, value in contents.items()
    )
    lines.append("MAGVASP_FETCH_END")
    return ("\n".join(lines) + "\n").encode("ascii")


def test_previews_use_fixed_hardened_ssh_and_sftp(tmp_path: Path) -> None:
    profile = _profile()
    record = _record()

    status = monitor.status_preview(profile, record)
    assert status["ssh_argv"][0] == "ssh"
    assert "BatchMode=yes" in status["ssh_argv"]
    assert "StrictHostKeyChecking=yes" in status["ssh_argv"]
    assert status["remote_directory"] not in " ".join(status["ssh_argv"])
    assert "/bin/bash -s" in status["ssh_argv"][-1]
    assert 'test "$(hostname -s)" = node03' in status["stdin_script"]
    assert "readlink -f" in status["stdin_script"]
    assert "tail -n 64" in status["stdin_script"]

    fetch = monitor.fetch_preview(profile, record, tmp_path / "fetched", ["OUTCAR"])
    assert fetch["sftp_argv"][0] == "sftp"
    assert "BatchMode=yes" in fetch["sftp_argv"]
    assert "StrictHostKeyChecking=yes" in fetch["sftp_argv"]
    assert "ProxyJump=cluster_user@gateway.example.org:22" in fetch["sftp_argv"]
    assert fetch["sftp_batch"].count("get ") == 1
    assert fetch["sftp_batch"].endswith("bye\n")
    assert "test ! -L" in fetch["ssh_inventory_stdin"]
    assert "stat -c '%d:%i:%h:%s'" in fetch["ssh_inventory_stdin"]
    assert "sha256sum --" in fetch["ssh_inventory_stdin"]
    assert "POTCAR" not in fetch["sftp_batch"]
    assert "WAVECAR" not in fetch["sftp_batch"]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"record_sha256": "0" * 64}, "RUN_RECORD_INTEGRITY"),
        ({"record_sha256": ""}, "RUN_RECORD_INTEGRITY"),
    ],
)
def test_record_hash_is_required_and_verified(
    updates: dict[str, object], message: str
) -> None:
    record = _record().model_copy(update=updates)
    with pytest.raises(monitor.ExecutionMonitorError, match=message):
        monitor.status_preview(_profile(), record)


@pytest.mark.parametrize(
    "remote_directory",
    [
        "/home/cluster_user/other/demo--run01/scf",
        "/home/cluster_user/magvasp/runs/demo--other/scf",
        "/home/cluster_user/magvasp/runs/demo--run01/../POTCAR",
        "/home/cluster_user/magvasp/runs//demo--run01/scf",
    ],
)
def test_remote_directory_is_scoped_and_bound(remote_directory: str) -> None:
    with pytest.raises(monitor.ExecutionMonitorError, match="REMOTE_DIRECTORY"):
        monitor.status_preview(
            _profile(), _record(remote_directory=remote_directory)
        )


@pytest.mark.parametrize(
    "profile",
    [
        _profile(gateway="jump,second"),
        _profile(user="cluster_user@other"),
        _profile(gateway_hostname="login01%h"),
    ],
)
def test_ssh_identity_tokens_cannot_change_route(profile: ClusterProfile) -> None:
    with pytest.raises(monitor.ExecutionMonitorError, match="SSH_IDENTITY_INVALID"):
        monitor.status_preview(profile, _record())


def test_node_pid_and_remote_backend_are_checked() -> None:
    for record in (
        _record(node="node13"),
        _record(node="login01"),
        _record(remote_pid=1),
        _record(backend=LaunchBackend.LOCAL_SIMULATION, cluster_execution="disabled"),
    ):
        with pytest.raises(monitor.ExecutionMonitorError):
            monitor.status_preview(_profile(), record)


def test_remote_status_distinguishes_running_completed_and_failed() -> None:
    def report(payload: bytes) -> dict:
        return monitor.remote_status(
            _profile(),
            _record(),
            runner=lambda argv, stdin, timeout: SimpleNamespace(
                returncode=0, stdout=payload, stderr=b""
            ),
        )

    running = report(_status_payload())
    assert running["state"] == "running"
    assert running["scientific_converged"] is None

    footer = b"General timing and accounting informations for this job:"
    completed = report(_status_payload(alive=False, tails={"OUTCAR": footer, "OSZICAR": b""}))
    assert completed["state"] == "completed"
    assert not completed["blockers"]

    failed = report(_status_payload(alive=False))
    assert failed["state"] == "failed"
    assert failed["blockers"] == ["CLEAN_COMPLETION_NOT_OBSERVED"]


def test_remote_status_flags_pid_mismatch_and_output_symlink() -> None:
    kinds = {
        name: (("regular", 12) if name == "OSZICAR" else ("missing", 0))
        for name in monitor.STATUS_FILES
    }
    kinds["OUTCAR"] = ("symlink", 0)
    output = _status_payload(alive=True, cwd="no", kinds=kinds)
    report = monitor.remote_status(
        _profile(),
        _record(),
        runner=lambda argv, stdin, timeout: SimpleNamespace(
            returncode=0, stdout=output, stderr=b""
        ),
    )
    assert report["state"] == "failed"
    assert set(report["blockers"]) == {
        "REMOTE_PID_IDENTITY_MISMATCH",
        "REMOTE_OUTPUT_SYMLINK",
    }


@pytest.mark.parametrize(
    "payload",
    [
        b"not-a-protocol\n",
        _status_payload(alive=True, cwd="unknown"),
        _status_payload(alive=False, cwd="yes"),
        _status_payload(
            alive=False,
            kinds={name: ("missing", 0) for name in monitor.STATUS_FILES},
            tails={"OUTCAR": b"forged tail", "OSZICAR": b""},
        ),
    ],
)
def test_status_protocol_rejects_malformed_or_inconsistent_payloads(
    payload: bytes,
) -> None:
    with pytest.raises(monitor.ExecutionMonitorError, match="REMOTE_STATUS_PROTOCOL"):
        monitor._parse_status(payload)


def test_status_protocol_rejects_unbounded_numeric_field() -> None:
    payload = _status_payload().replace(
        b"file=OUTCAR|regular|12",
        b"file=OUTCAR|regular|" + b"9" * 10000,
    )
    with pytest.raises(monitor.ExecutionMonitorError, match="REMOTE_STATUS_PROTOCOL"):
        monitor._parse_status(payload)


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(returncode=0, stdout="text", stderr=b""),
        SimpleNamespace(returncode=True, stdout=b"", stderr=b""),
        SimpleNamespace(returncode=None, stdout=b"", stderr=b""),
    ],
)
def test_runner_protocol_is_strict(result: object) -> None:
    with pytest.raises(monitor.ExecutionMonitorError, match="RUNNER_PROTOCOL"):
        monitor.remote_status(
            _profile(),
            _record(),
            runner=lambda argv, stdin, timeout: result,
        )


def test_runner_timeout_and_output_limit_are_normalized() -> None:
    def timeout_runner(argv: list[str], stdin: bytes, timeout: int) -> object:
        raise subprocess.TimeoutExpired(argv, timeout)

    with pytest.raises(monitor.ExecutionMonitorError, match="REMOTE_STATUS_TIMEOUT"):
        monitor.remote_status(_profile(), _record(), runner=timeout_runner)

    huge = b"x" * (monitor.MAX_STATUS_OUTPUT_BYTES + 1)
    with pytest.raises(monitor.ExecutionMonitorError, match="OUTPUT_TOO_LARGE"):
        monitor.remote_status(
            _profile(),
            _record(),
            runner=lambda argv, stdin, timeout: SimpleNamespace(
                returncode=0, stdout=huge, stderr=b""
            ),
        )


@pytest.mark.parametrize(
    ("files", "message"),
    [
        (["POTCAR"], "FETCH_SENSITIVE_FILE"),
        (["potcar"], "FETCH_SENSITIVE_FILE"),
        (["WAVECAR"], "FETCH_SENSITIVE_FILE"),
        (["../OUTCAR"], "FETCH_SCOPE"),
        (["OUTCAR/other"], "FETCH_SCOPE"),
        (["OUTCAR", "OUTCAR"], "FETCH_SCOPE"),
        ([], "FETCH_SCOPE"),
        ("OUTCAR", "FETCH_SCOPE"),
    ],
)
def test_fetch_accepts_only_unique_allowlisted_basenames(
    tmp_path: Path, files: object, message: str
) -> None:
    assert "POTCAR" not in monitor.FETCH_ALLOWED_FILES
    assert "WAVECAR" not in monitor.FETCH_ALLOWED_FILES
    with pytest.raises(monitor.ExecutionMonitorError, match=message):
        monitor.fetch_preview(
            _profile(),
            _record(),
            tmp_path / "dest",
            files,  # type: ignore[arg-type]
        )


def test_fetch_destination_must_be_new_absolute_and_without_traversal(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    cases = [Path("relative"), existing]
    nested = tmp_path / "nested"
    nested.mkdir()
    cases.append(nested / ".." / "escaped")
    for destination in cases:
        with pytest.raises(monitor.ExecutionMonitorError, match="FETCH_DESTINATION"):
            monitor.fetch_preview(_profile(), _record(), destination, ["OUTCAR"])


def test_fetch_destination_rejects_symlinked_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this host")
    with pytest.raises(monitor.ExecutionMonitorError, match="symlinked parent"):
        monitor.fetch_preview(_profile(), _record(), linked / "dest", ["OUTCAR"])


def test_fetch_results_uses_injected_runner_and_verifies_both_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "results"
    contents = {"OUTCAR": b"finished\n", "vasprun.xml": b"<model/>\n"}
    inventory = _inventory_payload(contents)
    calls: list[tuple[list[str], bytes, int]] = []

    def forbidden_subprocess(*args: object, **kwargs: object) -> object:
        raise AssertionError("injected runner must bypass subprocess.run")

    monkeypatch.setattr(monitor.subprocess, "run", forbidden_subprocess)

    def runner(argv: list[str], stdin: bytes, timeout: int) -> object:
        calls.append((argv, stdin, timeout))
        if argv[0] == "sftp":
            for name, value in contents.items():
                (destination / name).write_bytes(value)
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=inventory, stderr=b"")

    report = monitor.fetch_results(
        _profile(),
        _record(),
        destination,
        list(contents),
        runner=runner,
    )

    assert [call[0][0] for call in calls] == ["ssh", "sftp", "ssh"]
    assert [call[2] for call in calls] == [
        monitor.HASH_TIMEOUT_SECONDS,
        monitor.SFTP_TIMEOUT_SECONDS,
        monitor.HASH_TIMEOUT_SECONDS,
    ]
    assert report["remote_hashes_verified"] is True
    assert report["local_hashes_verified"] is True
    assert report["total_bytes"] == sum(map(len, contents.values()))
    assert report["potcar_fetched"] is False
    assert report["wavecar_fetched"] is False
    assert set(report["files"]) == set(contents)


def test_fetch_rejects_download_hash_mismatch(tmp_path: Path) -> None:
    destination = tmp_path / "mismatch"
    remote = {"OUTCAR": b"approved bytes"}

    def runner(argv: list[str], stdin: bytes, timeout: int) -> object:
        if argv[0] == "sftp":
            (destination / "OUTCAR").write_bytes(b"different bytes")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(
            returncode=0, stdout=_inventory_payload(remote), stderr=b""
        )

    with pytest.raises(monitor.ExecutionMonitorError, match="FETCH_HASH_MISMATCH"):
        monitor.fetch_results(
            _profile(), _record(), destination, ["OUTCAR"], runner=runner
        )


def test_fetch_rejects_remote_change_during_transfer(tmp_path: Path) -> None:
    destination = tmp_path / "changed"
    before = {"OUTCAR": b"first"}
    after = {"OUTCAR": b"second"}
    inventories = iter([_inventory_payload(before), _inventory_payload(after)])

    def runner(argv: list[str], stdin: bytes, timeout: int) -> object:
        if argv[0] == "sftp":
            (destination / "OUTCAR").write_bytes(before["OUTCAR"])
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=next(inventories), stderr=b"")

    with pytest.raises(monitor.ExecutionMonitorError, match="REMOTE_RESULT_CHANGED"):
        monitor.fetch_results(
            _profile(), _record(), destination, ["OUTCAR"], runner=runner
        )


def test_fetch_rejects_unexpected_local_file(tmp_path: Path) -> None:
    destination = tmp_path / "extra"
    contents = {"OUTCAR": b"expected"}

    def runner(argv: list[str], stdin: bytes, timeout: int) -> object:
        if argv[0] == "sftp":
            (destination / "OUTCAR").write_bytes(contents["OUTCAR"])
            (destination / "POTCAR").write_bytes(b"must not remain unnoticed")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(
            returncode=0, stdout=_inventory_payload(contents), stderr=b""
        )

    with pytest.raises(monitor.ExecutionMonitorError, match="unexpected or missing"):
        monitor.fetch_results(
            _profile(), _record(), destination, ["OUTCAR"], runner=runner
        )


def test_fetch_detects_destination_replacement_by_runner(tmp_path: Path) -> None:
    destination = tmp_path / "race"
    displaced = tmp_path / "displaced"
    contents = {"OUTCAR": b"expected"}

    def runner(argv: list[str], stdin: bytes, timeout: int) -> object:
        if argv[0] == "sftp":
            destination.rename(displaced)
            destination.mkdir()
            (destination / "OUTCAR").write_bytes(contents["OUTCAR"])
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(
            returncode=0, stdout=_inventory_payload(contents), stderr=b""
        )

    with pytest.raises(monitor.ExecutionMonitorError, match="destination was replaced"):
        monitor.fetch_results(
            _profile(), _record(), destination, ["OUTCAR"], runner=runner
        )


def test_local_symlink_is_not_hashed(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"content")
    link = tmp_path / "link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("file symlinks are unavailable on this host")
    with pytest.raises(monitor.ExecutionMonitorError, match="non-regular"):
        monitor._regular_file_digest(link)


def test_inventory_protocol_enforces_file_and_total_size_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    too_large = (
        "MAGVASP_FETCH_V1\n"
        f"file=OUTCAR|{monitor.MAX_FETCH_FILE_BYTES + 1}|{'0' * 64}\n"
        "MAGVASP_FETCH_END\n"
    ).encode()
    with pytest.raises(monitor.ExecutionMonitorError, match="REMOTE_HASH_PROTOCOL"):
        monitor._parse_inventory(too_large, ("OUTCAR",))

    monkeypatch.setattr(monitor, "MAX_FETCH_FILE_BYTES", 10)
    monkeypatch.setattr(monitor, "MAX_FETCH_TOTAL_BYTES", 5)
    total_too_large = (
        "MAGVASP_FETCH_V1\n"
        f"file=OUTCAR|3|{'0' * 64}\n"
        f"file=OSZICAR|3|{'1' * 64}\n"
        "MAGVASP_FETCH_END\n"
    ).encode()
    with pytest.raises(monitor.ExecutionMonitorError, match="FETCH_SIZE"):
        monitor._parse_inventory(total_too_large, ("OUTCAR", "OSZICAR"))


@pytest.mark.parametrize(
    "payload",
    [
        b"MAGVASP_FETCH_V1\nMAGVASP_FETCH_END\n",
        (
            "MAGVASP_FETCH_V1\n"
            f"file=OUTCAR|1|{'A' * 64}\n"
            "MAGVASP_FETCH_END\n"
        ).encode(),
        (
            "MAGVASP_FETCH_V1\n"
            f"file=OUTCAR|1|{'0' * 64}\n"
            f"file=OUTCAR|1|{'0' * 64}\n"
            "MAGVASP_FETCH_END\n"
        ).encode(),
    ],
)
def test_inventory_protocol_requires_exact_bound_inventory(payload: bytes) -> None:
    with pytest.raises(monitor.ExecutionMonitorError, match="REMOTE_HASH_PROTOCOL"):
        monitor._parse_inventory(payload, ("OUTCAR",))


def test_sftp_failure_is_fail_closed_and_keeps_audit_directory(tmp_path: Path) -> None:
    destination = tmp_path / "failed"
    contents = {"OUTCAR": b"value"}

    def runner(argv: list[str], stdin: bytes, timeout: int) -> object:
        if argv[0] == "sftp":
            return SimpleNamespace(returncode=7, stdout=b"", stderr=b"ignored")
        return SimpleNamespace(
            returncode=0, stdout=_inventory_payload(contents), stderr=b""
        )

    with pytest.raises(monitor.ExecutionMonitorError, match="SFTP_FETCH_FAILED"):
        monitor.fetch_results(
            _profile(), _record(), destination, ["OUTCAR"], runner=runner
        )
    assert destination.is_dir()
    assert not any(destination.iterdir())
