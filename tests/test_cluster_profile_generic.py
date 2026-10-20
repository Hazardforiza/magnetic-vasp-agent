from magnetic_vasp_agent.execution_models import ClusterProfile, ExecutionRequest
from magnetic_vasp_agent.execution_probe import probe_preview


def test_profile_accepts_another_cluster_without_changing_code() -> None:
    profile = ClusterProfile(
        gateway="login.example.org",
        gateway_hostname="login01",
        user="alice",
        port=2200,
        allowed_nodes=("gpu01", "gpu02"),
        vasp_module="vasp/6.4.2",
        mpi_path="/opt/intel/mpi/bin/mpirun",
        vasp_paths={
            "vasp_std": "/opt/vasp/bin/vasp_std",
            "vasp_ncl": "/opt/vasp/bin/vasp_ncl",
            "vasp_gam": "/opt/vasp/bin/vasp_gam",
        },
    )
    request = ExecutionRequest(node="gpu01", step_id="scf", run_id="r1")
    preview = probe_preview(profile, request.node)
    assert "alice@login.example.org" in preview["ssh_argv"]
    assert "login01" in preview["ssh_argv"][-1]
    assert "mu01" not in preview["ssh_argv"][-1]
    assert "alice" in preview["stdin_script"]


def test_profile_rejects_unsafe_cluster_tokens() -> None:
    for kwargs in (
        {"gateway": "login;id"},
        {"gateway_hostname": "login one"},
        {"user": "alice$HOME"},
        {"allowed_nodes": ("gpu01;id",)},
        {"mpi_path": "mpirun;id"},
    ):
        try:
            ClusterProfile(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe profile accepted: {kwargs}")


def test_gateway_cannot_be_compute_node() -> None:
    try:
        ClusterProfile(
            gateway_hostname="login01",
            allowed_nodes=("login01", "gpu01"),
        )
    except ValueError as exc:
        assert "gateway hostname" in str(exc)
    else:
        raise AssertionError("jump host was accepted as a compute node")
