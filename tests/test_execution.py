from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from magnetic_vasp_agent.cli import app
from magnetic_vasp_agent.execution import prepare_execution, write_report
from magnetic_vasp_agent.execution_models import (
    ALLOWED_NODES,
    ClusterProfile,
    Cpu,
    ExecutionRequest,
    ProbeSnapshot,
    ProcessObservation,
)
from magnetic_vasp_agent.execution_probe import (
    assess_resources,
    collect_probe,
    parse_probe,
    probe_preview,
)
from magnetic_vasp_agent.execution_simulator import LocalSimulation, simulate
from magnetic_vasp_agent.models import (
    PotentialPolicy,
    PotentialRecord,
    PotentialRegistry,
)
from magnetic_vasp_agent.transfer import build_transfer_plan, transfer_bundle
from magnetic_vasp_agent.transfer_models import TransferRequest
from magnetic_vasp_agent.workflow import (
    digest,
    json_bytes,
    materialize_workflow,
    prepare_workflow,
)
from magnetic_vasp_agent.workflow_models import Workflow

ROOT = Path(__file__).parents[1]


def topology_snapshot(*, busy: bool, node: str = "cu21") -> ProbeSnapshot:
    cpus = [Cpu(cpu=i, core=i % 20, socket=(i % 20) // 10, online=True) for i in range(40)]
    processes = []
    if busy:
        processes = [
            ProcessObservation(pid=1000 + i, user="songyz", last_cpu=i, cpu_percent=99.7, command="kkrscf")
            for i in range(20)
        ]
    return ProbeSnapshot(
        source="offline_fixture", observed_at=datetime.now(UTC),
        requested_node=node, hostname=f"{node}.cmphc", username="zyy", cpus=cpus,
        allowed_cpu_ids=list(range(40)), processes=processes,
        load1=20.0 if busy else 0.0, mem_available_kib=8 * 1024 * 1024,
        module_loaded=True,
        mpi_path="/public/intel/oneapi/mpi/2021.1.1/bin/mpirun",
        executables={"vasp_std": True, "vasp_ncl": True, "vasp_gam": True},
    )


def test_node_policy_is_exact():
    assert set(ALLOWED_NODES) == {"cu21", "cu22", "cu23", "cu41", "cu42", "cu43", "cu48", "cu49", "cu50"}
    with pytest.raises(ValueError):
        probe_preview(ClusterProfile(), "mu01")


def test_busy_cu21_cannot_be_declared_safe():
    request = ExecutionRequest(node="cu21", step_id="step", run_id="run", mpi_ranks=1)
    assessment = assess_resources(topology_snapshot(busy=True), request)
    codes = {item.code for item in assessment.blockers}
    assert "NODE_BUSY" in codes
    assert "ALLOCATION_UNCONFIRMED" in codes
    assert not assessment.ready_to_run
    assert assessment.physical_cores == 20
    assert assessment.logical_cpus == 40
    assert assessment.observed_active_physical_cores == 20


def test_smt_siblings_are_not_two_physical_cores():
    snapshot = topology_snapshot(busy=False)
    request = ExecutionRequest(node="cu21", step_id="step", run_id="run", mpi_ranks=2, cpu_ids=[0, 20])
    codes = {item.code for item in assess_resources(snapshot, request).blockers}
    assert "SMT_SIBLING_CONFLICT" in codes


def test_probe_preview_is_non_executing():
    preview = probe_preview(ClusterProfile(), "cu21")
    assert preview["mode"] == "readonly_probe_preview"
    assert preview["executes_mpi"] is False
    assert preview["writes_remote_files"] is False
    assert "vasp_std" in preview["stdin_script"]


def test_probe_parser_rejects_extra_sections():
    with pytest.raises(ValueError):
        parse_probe("MAGVASP_PROBE_V1\n@@identity\ncu21\nzyy\n@@end\n", "cu21")


@pytest.mark.parametrize("scenario,expected", [("complete", "completed_unverified"), ("fail", "failed"), ("timeout", "timed_out"), ("cancel", "cancelled")])
def test_local_simulator_lifecycle(tmp_path, scenario, expected):
    report = simulate(tmp_path / scenario, scenario)
    assert report["state"] == expected
    assert report["scientific_converged"] is None
    assert report["artifacts_released"] is False


@pytest.fixture
def packages(tmp_path):
    records = []
    for element in ("Fe", "O", "Mn", "Al"):
        path = tmp_path / "potentials" / element / "POTCAR"
        path.parent.mkdir(parents=True)
        payload = f"SYNTHETIC ONLY: {element}\n".encode()
        path.write_bytes(payload)
        records.append(PotentialRecord(
            element=element, potential=element, family="PBE_54", titel=f"Synthetic {element}",
            enmax=400, relative_path=f"{element}/POTCAR", sha256=digest(payload),
        ))
    registry = PotentialRegistry(root=str(tmp_path / "potentials"), family="PBE_54", records=records)
    result = {}
    for name in ("FeO.generated.mcif", "Mn3Al.scf-dos-band"):
        request_path = ROOT / "config" / "workflows" / f"{name}.json"
        prepared = prepare_workflow(
            Workflow.model_validate_json(request_path.read_bytes()), registry,
            PotentialPolicy(family="PBE_54", preferred={}), request_path.parent,
        )
        folder = tmp_path / name
        materialize_workflow(prepared, folder)
        result[name] = folder
    return result


def test_execution_plan_is_blocked_without_remote_side_effects(packages, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline plan must not spawn processes")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    bundle = packages["FeO.generated.mcif"]
    request = ExecutionRequest(node="cu21", step_id="candidate-01", run_id="review", mpi_ranks=1)
    report = prepare_execution(bundle, ClusterProfile(), request)
    codes = {item["code"] for item in report["blockers"]}
    assert report["ready_to_run"] is False
    assert "REMOTE_EXECUTION_DISABLED" in codes
    assert report["guarded_script_preview"] is None
    assert report["executable"] == "vasp_ncl"
    assert "SYNTHETIC ONLY" not in json.dumps(report)


@pytest.mark.parametrize("node", ["mu01", "mu01.cmphc", "cu20", "cu24", "cu51", "10.59.143.97", "cu21; id", "cu21\n", "cu21.cmphc", "-oProxyCommand=id"])
def test_unknown_nodes_never_reach_subprocess(node, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("invalid node reached SSH"))
    with pytest.raises(ValueError):
        collect_probe(ClusterProfile(), node)
    with pytest.raises(ValueError):
        ExecutionRequest(node=node, step_id="scf", run_id="run")


@pytest.mark.parametrize("value", [True, 1, 0, "false", "true"])
def test_no_execution_enable_switch(value):
    with pytest.raises(ValueError):
        ClusterProfile(execution_enabled=value)


@pytest.mark.parametrize("value", ["/", "/home/zyy", "~/runs", "/home/zyy/run/../old", "/home/zyy/run/a;id", "/home/zyy/run/a\n", "/home//zyy/runs", "/home/zyy/run/", "C:/runs"])
def test_remote_root_rejects_broad_and_unsafe_paths(value):
    with pytest.raises(ValueError):
        ClusterProfile(remote_root=value)


@pytest.mark.parametrize("kwargs", [{"mpi_ranks": 0}, {"mpi_ranks": True}, {"mpi_ranks": "2"}, {"mpi_ranks": 2, "cpu_ids": [0]}, {"mpi_ranks": 2, "cpu_ids": [0, 0]}, {"cpu_ids": [-1]}, {"run_id": "../old"}, {"step_id": "CON"}, {"command": "id"}])
def test_execution_request_strict(kwargs):
    with pytest.raises(ValueError):
        ExecutionRequest(**{"node": "cu21", "step_id": "scf", "run_id": "run", **kwargs})


@pytest.mark.parametrize("offset", [-61, 10])
def test_stale_or_future_snapshot_blocks(offset):
    snapshot = topology_snapshot(busy=False)
    now = snapshot.observed_at
    snapshot = snapshot.model_copy(update={"observed_at": now + timedelta(seconds=offset)})
    assessment = assess_resources(snapshot, ExecutionRequest(node="cu21", step_id="scf", run_id="run"), now=now)
    assert "SNAPSHOT_STALE" in {b.code for b in assessment.blockers}


def test_idle_node_does_not_establish_allocation():
    snapshot = topology_snapshot(busy=False).model_copy(update={"source": "ssh_probe"})
    assessment = assess_resources(snapshot, ExecutionRequest(node="cu21", step_id="scf", run_id="run", mpi_ranks=20))
    assert assessment.suggested_cpu_ids == list(range(20))
    assert not assessment.ready_to_run and not assessment.allocation_verified
    assert "ALLOCATION_UNCONFIRMED" in {b.code for b in assessment.blockers}


def test_affinity_and_ranks_limits():
    snapshot = topology_snapshot(busy=False).model_copy(update={"allowed_cpu_ids": [0, 20]})
    report = assess_resources(snapshot, ExecutionRequest(node="cu21", step_id="scf", run_id="run", mpi_ranks=2, cpu_ids=[0, 1]))
    assert report.accessible_physical_cores == 1
    assert {"CPU_NOT_ACCESSIBLE", "INSUFFICIENT_PHYSICAL_CORES"} <= {b.code for b in report.blockers}


def test_wrong_host_and_runtime_block():
    snapshot = topology_snapshot(busy=False).model_copy(update={"hostname": "mu01", "mpi_path": "/unexpected/mpirun", "mem_available_kib": 0})
    report = assess_resources(snapshot, ExecutionRequest(node="cu21", step_id="scf", run_id="run"))
    assert {"IDENTITY_MISMATCH", "RUNTIME_MISMATCH", "LOW_MEMORY"} <= {b.code for b in report.blockers}


def probe_text():
    snapshot = topology_snapshot(busy=True)
    topology = {"cpus": [{"cpu": str(c.cpu), "core": str(c.core), "socket": str(c.socket), "online": "yes"} for c in snapshot.cpus]}
    return "\n".join([
        "SSH banner", "MAGVASP_PROBE_V1", "@@identity", "cu21.cmphc", "zyy",
        "@@time", "2026-09-05T12:00:00Z", "@@cpus", json.dumps(topology),
        "@@affinity", "0-39", "@@processes",
        *[f"{p.pid} {p.user} {p.last_cpu} {p.cpu_percent} {p.command}" for p in snapshot.processes],
        "@@load", "20.0 20.1 19.9 20/200 1234", "@@memory", "8388608",
        "@@runtime", "module_loaded=yes", "mpi_path=/public/intel/oneapi/mpi/2021.1.1/bin/mpirun",
        "vasp_std=yes", "vasp_ncl=yes", "vasp_gam=yes", "@@end", "",
    ])


def test_probe_parse_complete_protocol():
    snapshot = parse_probe(probe_text(), "cu21")
    assert len(snapshot.cpus) == 40
    assert len(snapshot.processes) == 20
    assert snapshot.module_loaded
    assert snapshot.source == "ssh_probe"


@pytest.mark.parametrize("before,after", [
    ("@@end", "@@unexpected"), ("vasp_std=yes", "vasp_std=perhaps"),
    ("0-39", "0-99999999"), ("99.7", "nan"), ("@@load", "@@cpus"),
    ("2026-09-05T12:00:00Z", "2026-09-05T12:00:00"),
])
def test_bad_probe_protocol(before, after):
    with pytest.raises(ValueError):
        parse_probe(probe_text().replace(before, after), "cu21")


def test_collect_only_runs_fixed_probe(monkeypatch):
    captured = {}
    monkeypatch.setattr("magnetic_vasp_agent.execution_probe.shutil.which", lambda _: "trusted-ssh")
    def run(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        return SimpleNamespace(returncode=0, stdout=probe_text().encode(), stderr=b"")
    monkeypatch.setattr(subprocess, "run", run)
    assert collect_probe(ClusterProfile(), "cu21").hostname == "cu21.cmphc"
    assert captured["argv"][0] == "trusted-ssh"
    assert "StrictHostKeyChecking=yes" in captured["argv"]
    assert "StrictHostKeyChecking=yes" in captured["argv"][-1]
    assert captured["timeout"] == 30
    assert "shell" not in captured
    script = captured["input"].decode()
    assert "-np" not in script and "kill" not in script and "sbatch" not in script
    assert script.index('hostname -s') < script.index("module load")


@pytest.mark.parametrize("failure", ["timeout", "exit", "large"])
def test_probe_failures_are_explicit_and_do_not_echo_remote_data(monkeypatch, failure):
    monkeypatch.setattr("magnetic_vasp_agent.execution_probe.shutil.which", lambda _: "trusted-ssh")
    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("ssh", 30)
        return SimpleNamespace(returncode=255 if failure == "exit" else 0,
                               stdout=b"x" * 2_000_001 if failure == "large" else b"",
                               stderr=b"UNTRUSTED_REMOTE_TEXT")
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError) as exc:
        collect_probe(ClusterProfile(), "cu21")
    assert "UNTRUSTED_REMOTE_TEXT" not in str(exc.value)


def test_guarded_preview_has_physical_binding(packages):
    profile = ClusterProfile(remote_root="/home/zyy/magvasp/runs")
    request = ExecutionRequest(node="cu21", step_id="candidate-01", run_id="review", mpi_ranks=2)
    report = prepare_execution(packages["FeO.generated.mcif"], profile, request, topology_snapshot(busy=True))
    assert report["guarded_script_preview"].splitlines()[2] == "exit 78"
    assert report["environment_candidate"]["I_MPI_PIN_PROCESSOR_LIST"] == "0,1"
    assert report["mpi_argv_candidate"][1:5] == ["-bootstrap", "fork", "-hosts", "cu21"]
    assert "NODE_BUSY" in {b["code"] for b in report["blockers"]}


def test_waiting_band_cannot_be_released_by_local_charge(packages):
    bundle = packages["Mn3Al.scf-dos-band"]
    (bundle / "band" / "CHGCAR").write_text("not a converged charge density")
    report = prepare_execution(bundle, ClusterProfile(), ExecutionRequest(node="cu21", step_id="band", run_id="review"))
    assert "UPSTREAM_NOT_VERIFIED" in {b["code"] for b in report["blockers"]}
    assert report["executable"] == "vasp_std"


@pytest.mark.parametrize("filename", ["INCAR", "POSCAR", "POTCAR", "KPOINTS", "manifest.json"])
def test_plan_rejects_tampering(packages, filename):
    bundle = packages["FeO.generated.mcif"]
    with (bundle / "candidate-01" / filename).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        prepare_execution(bundle, ClusterProfile(), ExecutionRequest(node="cu21", step_id="candidate-01", run_id="review"))


def test_manifest_contract_checked_even_with_updated_hashes(packages):
    bundle = packages["FeO.generated.mcif"]
    path = bundle / "candidate-01" / "manifest.json"
    step = json.loads(path.read_bytes())
    step["goal"]["task"] = "band"
    path.write_bytes(json_bytes(step))
    path = bundle / "workflow.manifest.json"
    top = json.loads(path.read_bytes())
    top["steps"]["candidate-01"]["manifest_sha256"] = digest(json_bytes(step))
    path.write_bytes(json_bytes(top))
    with pytest.raises(ValueError, match="MANIFEST_CONTRACT"):
        prepare_execution(bundle, ClusterProfile(), ExecutionRequest(node="cu21", step_id="candidate-01", run_id="review"))


def test_reports_and_simulations_refuse_overwrite(tmp_path):
    output = tmp_path / "report.json"
    write_report({"original": True}, output)
    with pytest.raises(FileExistsError):
        write_report({"original": False}, output)
    assert json.loads(output.read_bytes()) == {"original": True}
    with pytest.raises(FileExistsError):
        simulate(tmp_path, "complete")


def test_local_cancel_only_affects_owned_process(tmp_path):
    first = LocalSimulation(tmp_path / "first", "cancel", timeout_seconds=10)
    other = LocalSimulation(tmp_path / "other", "cancel", timeout_seconds=10)
    try:
        first.start()
        other.start()
        first.cancel()
        first.cancel()
        assert first.state == "cancelled"
        assert first.process.poll() is not None
        assert other.poll() == "running"
        with pytest.raises(ValueError):
            first.start()
    finally:
        first.close()
        other.close()


def test_cli_previews_never_connect_and_have_no_launch_option(packages, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("preview tried a subprocess"))
    runner = CliRunner()
    profile = str(ROOT / "config/execution/cluster.json")
    result = runner.invoke(app, ["execution", "probe", "--profile", profile, "--node", "cu21"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["mode"] == "readonly_probe_preview"
    result = runner.invoke(app, ["execution", "plan", "--profile", profile,
        "--request", str(ROOT / "config/execution/FeO.cu21.preview.json"),
        "--bundle", str(packages["FeO.generated.mcif"])])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["ready_to_run"] is False
    for command in ("launch", "upload", "cancel", "retry"):
        assert runner.invoke(app, ["execution", command]).exit_code == 2


def test_transfer_plan_is_scoped_and_non_overwriting(packages):
    profile = ClusterProfile(remote_root="/home/zyy/magvasp/runs")
    request = TransferRequest(node="cu21", run_id="review")
    report = build_transfer_plan(packages["FeO.generated.mcif"], profile, request)
    assert report["ready_to_transfer"] is True
    assert report["transferred"] is False
    assert report["remote_directory"].endswith("--review")
    assert len(report["files"]) == 16
    assert "POTCAR" in report["sftp_batch"]
    assert "put" in report["sftp_batch"]
    assert "rm " not in report["sftp_batch"]
    assert "vasp_" not in report["sftp_batch"]
    assert "test ! -e" in report["ssh_guard_stdin"]
    assert "mkdir --" in report["ssh_guard_stdin"]
    assert report["execution_enabled"] is False


def test_transfer_plan_rejects_unapproved_files(packages):
    bundle = packages["FeO.generated.mcif"]
    (bundle / "candidate-01" / "OUTCAR").write_text("not uploaded")
    with pytest.raises(ValueError, match="TRANSFER_SCOPE"):
        build_transfer_plan(bundle, ClusterProfile(remote_root="/home/zyy/magvasp/runs"), TransferRequest(node="cu21", run_id="review"))


def test_transfer_plan_rejects_bundle_and_manifest_symlinks(packages, tmp_path):
    bundle = packages["FeO.generated.mcif"]
    link = tmp_path / "bundle-link"
    try:
        link.symlink_to(bundle, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="TRANSFER_SCOPE"):
        build_transfer_plan(link, ClusterProfile(remote_root="/home/zyy/magvasp/runs"), TransferRequest(node="cu21", run_id="review"))
    manifest_link = bundle / "workflow.manifest.json.link"
    manifest_link.symlink_to(bundle / "workflow.manifest.json")
    with pytest.raises(ValueError, match="TRANSFER_SCOPE"):
        build_transfer_plan(bundle, ClusterProfile(remote_root="/home/zyy/magvasp/runs"), TransferRequest(node="cu21", run_id="review"))


def test_transfer_plan_rejects_unsafe_request_and_root():
    with pytest.raises(ValueError):
        TransferRequest(node="cu21", run_id="../review")
    with pytest.raises(ValueError):
        ClusterProfile(remote_root="/home/zyy/magvasp/runs/../old")


def test_transfer_upload_verifies_hashes_and_never_runs_vasp(packages, monkeypatch):
    profile = ClusterProfile(remote_root="/home/zyy/magvasp/runs")
    request = TransferRequest(node="cu21", run_id="verified")
    plan = build_transfer_plan(packages["FeO.generated.mcif"], profile, request)
    calls = []

    monkeypatch.setattr("magnetic_vasp_agent.transfer.shutil.which", lambda name: name)

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0] == "ssh" and "sha256sum" in kwargs["input"].decode():
            lines = ["MAGVASP_TRANSFER_V1"]
            for item in plan["files"]:
                lines.append(f"{item['sha256']}  {plan['remote_directory']}/{item['relative_path']}")
            return SimpleNamespace(returncode=0, stdout=("\n".join(lines) + "\n").encode(), stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("magnetic_vasp_agent.transfer.subprocess.run", run)
    report = transfer_bundle(packages["FeO.generated.mcif"], profile, request)
    assert report["transferred"] is True
    assert report["remote_hashes_verified"] is True
    assert len(calls) == 3
    assert calls[0][0][0] == "ssh"
    assert calls[1][0][0] == "sftp"
    assert calls[2][0][0] == "ssh"
    assert all(call[1]["shell"] is False for call in calls)
    assert all("vasp" not in " ".join(call[0]).lower() for call in calls)


def test_transfer_stops_after_directory_guard_failure(packages, monkeypatch):
    profile = ClusterProfile(remote_root="/home/zyy/magvasp/runs")
    request = TransferRequest(node="cu21", run_id="guard-fails")
    monkeypatch.setattr("magnetic_vasp_agent.transfer.shutil.which", lambda name: name)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv[0])
        return SimpleNamespace(returncode=17, stdout=b"", stderr=b"remote directory exists")

    monkeypatch.setattr("magnetic_vasp_agent.transfer.subprocess.run", run)
    with pytest.raises(ValueError, match="REMOTE_DIRECTORY_GUARD_FAILED"):
        transfer_bundle(packages["FeO.generated.mcif"], profile, request)
    assert calls == ["ssh"]
