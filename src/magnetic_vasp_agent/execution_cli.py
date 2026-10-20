# ruff: noqa: B008 - Typer requires option declarations in function signatures.
from __future__ import annotations

from datetime import timedelta
from enum import Enum
from pathlib import Path

import typer

from magnetic_vasp_agent.execution import prepare_execution, write_report
from magnetic_vasp_agent.execution_approval import (
    DEFAULT_GRANT_MINUTES,
    MAX_GRANT_MINUTES,
    RESOURCE_GRANT_CONFIRM,
    issue_resource_grant,
)
from magnetic_vasp_agent.execution_artifacts import release_chgcar
from magnetic_vasp_agent.execution_models import (
    ClusterProfile,
    ExecutionRequest,
    ProbeSnapshot,
    ResourceGrant,
)
from magnetic_vasp_agent.execution_monitor import fetch_results, remote_status
from magnetic_vasp_agent.execution_probe import (
    collect_login,
    collect_probe,
    collect_wannier_probe,
    login_preview,
    probe_preview,
    wannier_probe_preview,
)
from magnetic_vasp_agent.execution_remote import (
    RemoteExecutionError,
    run_mpi_smoke,
    smoke_preview,
    write_json_report,
)
from magnetic_vasp_agent.execution_runtime import cancel as cancel_run
from magnetic_vasp_agent.execution_runtime import launch as launch_run
from magnetic_vasp_agent.execution_runtime import status as status_run
from magnetic_vasp_agent.execution_runtime_models import LaunchBackend, LaunchRequest
from magnetic_vasp_agent.execution_simulator import simulate
from magnetic_vasp_agent.execution_site import (
    UserIdentity,
    compose_cluster_profile,
    load_site_template,
    load_user_identity,
    write_cluster_profile,
)
from magnetic_vasp_agent.transfer import (
    build_transfer_plan,
    transfer_bundle,
    write_transfer_report,
)
from magnetic_vasp_agent.transfer_models import TransferRequest
from magnetic_vasp_agent.workflow import json_bytes

execution_app = typer.Typer(
    no_args_is_help=True,
    help="Prepare and, when enabled, start gated single-node MPI/VASP runs.",
)


def _emit(report: dict, output: Path | None) -> None:
    if output:
        write_report(report, output)
    typer.echo(json_bytes(report).decode("utf-8"), nl=False)


def _error(exc: Exception) -> None:
    typer.echo(f"Execution preparation error: {exc}", err=True)
    raise typer.Exit(2)


@execution_app.command("compose")
def compose(
    site: Path = typer.Option(..., "--site", exists=True, dir_okay=False),
    identity: Path | None = typer.Option(None, "--identity", exists=True, dir_okay=False),
    user: str | None = typer.Option(None, "--user", help="Cluster username; required without --identity"),
    connection: str | None = typer.Option(None, "--connection", help="Site connection id, e.g. office or tunnel"),
    execution_enabled: bool = typer.Option(False, "--execution-enabled/--execution-disabled"),
    output: Path = typer.Option(..., "--output"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Build a private ClusterProfile from the shared site template plus one identity."""
    try:
        template = load_site_template(site)
        if identity is not None:
            person = load_user_identity(identity)
            if user or connection:
                person = person.model_copy(
                    update={
                        key: value
                        for key, value in {"user": user, "connection": connection}.items()
                        if value
                    }
                )
            if execution_enabled:
                person = person.model_copy(update={"execution_enabled": True})
        else:
            if not user or not connection:
                raise ValueError("compose without --identity requires --user and --connection")
            person = UserIdentity(
                user=user,
                connection=connection,
                execution_enabled=execution_enabled,
            )
        profile = compose_cluster_profile(template, person)
        write_cluster_profile(profile, output, overwrite=overwrite)
        typer.echo(json_bytes(profile.model_dump(mode="json")).decode("utf-8"), nl=False)
    except (ValueError, OSError) as exc:
        _error(exc)


@execution_app.command("plan")
def plan(
    bundle: Path = typer.Option(..., "--bundle", exists=True, file_okay=False),
    profile: Path = typer.Option(..., "--profile", exists=True, dir_okay=False),
    request: Path = typer.Option(..., "--request", exists=True, dir_okay=False),
    snapshot: Path | None = typer.Option(None, "--snapshot", exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Offline input-integrity check and blocked execution preview. Never connects to SSH."""
    try:
        _emit(prepare_execution(
            bundle, ClusterProfile.model_validate_json(profile.read_bytes()),
            ExecutionRequest.model_validate_json(request.read_bytes()),
            ProbeSnapshot.model_validate_json(snapshot.read_bytes()) if snapshot else None,
        ), output)
    except (ValueError, OSError) as exc:
        _error(exc)


@execution_app.command("grant")
def grant_command(
    bundle: Path = typer.Option(..., "--bundle", exists=True, file_okay=False),
    profile: Path = typer.Option(..., "--profile", exists=True, dir_okay=False),
    request: Path = typer.Option(..., "--request", exists=True, dir_okay=False),
    snapshot: Path = typer.Option(..., "--snapshot", exists=True, dir_okay=False),
    approver: str = typer.Option(..., "--approver"),
    confirm: str = typer.Option(
        "", "--confirm", help=f"Exact phrase {RESOURCE_GRANT_CONFIRM}"
    ),
    output: Path = typer.Option(..., "--output"),
    valid_minutes: int = typer.Option(
        DEFAULT_GRANT_MINUTES,
        "--valid-minutes",
        "--valid-for-minutes",
        min=1,
        max=MAX_GRANT_MINUTES,
    ),
    grant_id: str | None = typer.Option(None, "--grant-id"),
) -> None:
    """Issue a reviewed, time-bounded ResourceGrant without contacting SSH."""

    try:
        result = issue_resource_grant(
            bundle,
            ClusterProfile.model_validate_json(profile.read_bytes()),
            ExecutionRequest.model_validate_json(request.read_bytes()),
            ProbeSnapshot.model_validate_json(snapshot.read_bytes()),
            output,
            approver=approver,
            confirm=confirm,
            valid_for=timedelta(minutes=valid_minutes),
            grant_id=grant_id,
        )
        typer.echo(result.model_dump_json(indent=2))
    except (ValueError, OSError) as exc:
        _error(exc)


@execution_app.command("login")
def login(
    profile: Path = typer.Option(..., "--profile", exists=True, dir_okay=False),
    connect: bool = typer.Option(False, "--connect", help="Open the one-shot gateway identity check."),
    interactive: bool = typer.Option(
        True,
        "--interactive/--batch",
        help="Type the password in this terminal (never stored). Default: interactive.",
    ),
) -> None:
    """Check the configured gateway like MobaXterm; password stays on your keyboard."""
    try:
        settings = ClusterProfile.model_validate_json(profile.read_bytes())
        report = (
            collect_login(settings, interactive=interactive)
            if connect
            else login_preview(settings, interactive=interactive)
        )
        _emit(report, None)
    except (ValueError, OSError) as exc:
        _error(exc)


@execution_app.command("probe")
def probe(
    profile: Path = typer.Option(..., "--profile", exists=True, dir_okay=False),
    node: str = typer.Option(..., "--node"),
    connect: bool = typer.Option(False, "--connect", help="Explicitly run the fixed read-only SSH probe."),
    interactive: bool = typer.Option(
        False,
        "--interactive/--batch",
        help="Prompt for the gateway password in this terminal; never stored.",
    ),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Preview by default; --connect reads one node without running MPI/VASP."""
    try:
        if output and (output.exists() or output.is_symlink()):
            raise FileExistsError("Report exists; choose a new path before probing")
        settings = ClusterProfile.model_validate_json(profile.read_bytes())
        report = (
            collect_probe(settings, node, interactive=interactive).model_dump(mode="json")
            if connect
            else probe_preview(settings, node, interactive=interactive)
        )
        _emit(report, output)
    except (ValueError, OSError) as exc:
        _error(exc)


@execution_app.command("wannier-probe")
def wannier_probe(
    profile: Path = typer.Option(..., "--profile", exists=True, dir_okay=False),
    node: str = typer.Option(..., "--node"),
    connect: bool = typer.Option(
        False,
        "--connect",
        help="Explicitly run the fixed read-only Wannier capability probe.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive/--batch",
        help="Prompt for the gateway password in this terminal; never stored.",
    ),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Check module, Wannier90, and vasp_ncl markers without computation."""
    try:
        if output and (output.exists() or output.is_symlink()):
            raise FileExistsError("Report exists; choose a new path before probing")
        settings = ClusterProfile.model_validate_json(profile.read_bytes())
        report = (
            collect_wannier_probe(settings, node, interactive=interactive).model_dump(mode="json")
            if connect
            else wannier_probe_preview(settings, node, interactive=interactive)
        )
        _emit(report, output)
    except (ValueError, OSError) as exc:
        _error(exc)


class SimulationScenario(str, Enum):
    complete = "complete"
    fail = "fail"
    timeout = "timeout"
    cancel = "cancel"


@execution_app.command("simulate")
def simulation(
    output: Path = typer.Option(..., "--output"),
    scenario: SimulationScenario = typer.Option(SimulationScenario.complete, "--scenario"),
) -> None:
    """Run a fixed tiny local Python process, not VASP, SSH or an arbitrary command."""
    try:
        _emit(simulate(output, scenario.value), None)
    except (ValueError, OSError) as exc:
        _error(exc)


@execution_app.command("smoke")
def smoke_command(
    profile: Path = typer.Option(..., "--profile", exists=True, dir_okay=False),
    node: str | None = typer.Option(None, "--node"),
    request: Path | None = typer.Option(None, "--request", exists=True, dir_okay=False),
    resource_grant: Path | None = typer.Option(
        None, "--resource-grant", exists=True, dir_okay=False
    ),
    confirm: str = typer.Option("", "--confirm", help="Exact phrase I_CONFIRM_MPI_SMOKE"),
    connect: bool = typer.Option(False, "--connect", help="Run the fixed single-node MPI smoke test."),
    ranks: int = typer.Option(2, "--ranks", min=1, max=8),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Preview, or run a resource-grant-bound Intel MPI smoke. Never starts VASP."""
    try:
        settings = ClusterProfile.model_validate_json(profile.read_bytes())
        if not connect:
            if not node:
                raise ValueError("smoke preview requires --node")
            report = smoke_preview(settings, node, ranks=ranks)
        else:
            if request is None or resource_grant is None:
                raise ValueError(
                    "connected smoke requires --request and --resource-grant"
                )
            intent = ExecutionRequest.model_validate_json(request.read_bytes())
            grant = ResourceGrant.model_validate_json(resource_grant.read_bytes())
            if not grant.matches_request(intent):
                raise ValueError("RESOURCE_GRANT_MISMATCH")
            report = run_mpi_smoke(
                settings,
                intent.node,
                confirm=confirm,
                run_id=intent.run_id,
                ranks=grant.mpi_ranks,
                cpu_ids=grant.cpu_ids,
            )
        if output:
            write_json_report(report, output)
        typer.echo(json_bytes(report).decode("utf-8"), nl=False)
    except (ValueError, OSError, RemoteExecutionError) as exc:
        _error(exc)


@execution_app.command("launch")
def launch_command(
    backend: str = typer.Option(..., "--backend", help="local_simulation or remote_mpi"),
    confirm: str = typer.Option(..., "--confirm", help="Exact confirmation phrase"),
    output: Path = typer.Option(..., "--output"),
    scenario: SimulationScenario = typer.Option(SimulationScenario.complete, "--scenario"),
    run_id: str = typer.Option("local-sim", "--run-id"),
    profile: Path | None = typer.Option(None, "--profile", exists=True, dir_okay=False),
    bundle: Path | None = typer.Option(None, "--bundle", exists=True, file_okay=False),
    request: Path | None = typer.Option(None, "--request", exists=True, dir_okay=False),
    snapshot: Path | None = typer.Option(None, "--snapshot", exists=True, dir_okay=False),
    transfer_report: Path | None = typer.Option(None, "--transfer-report", exists=True, dir_okay=False),
    smoke_report: Path | None = typer.Option(None, "--smoke-report", exists=True, dir_okay=False),
    resource_grant: Path | None = typer.Option(None, "--resource-grant", exists=True, dir_okay=False),
    artifact_release: Path | None = typer.Option(None, "--artifact-release", exists=True, dir_okay=False),
) -> None:
    """Start a local simulation, or a gated remote VASP run after smoke + transfer."""
    try:
        record = launch_run(
            LaunchRequest(
                backend=LaunchBackend(backend),
                confirm=confirm,
                output=str(output),
                scenario=scenario.value,
                run_id=run_id,
                profile=str(profile) if profile else None,
                bundle=str(bundle) if bundle else None,
                request=str(request) if request else None,
                snapshot=str(snapshot) if snapshot else None,
                transfer_report=str(transfer_report) if transfer_report else None,
                smoke_report=str(smoke_report) if smoke_report else None,
                resource_grant=str(resource_grant) if resource_grant else None,
                artifact_release=str(artifact_release) if artifact_release else None,
            )
        )
        typer.echo(record.model_dump_json(indent=2), nl=False)
        typer.echo("")
    except (ValueError, OSError) as exc:
        _error(exc)
    if record.state.value == "blocked":
        raise typer.Exit(1)


@execution_app.command("status")
def status_command(
    run: Path = typer.Option(..., "--run", exists=True, file_okay=False),
) -> None:
    """Read a local run record. Does not contact the cluster."""
    try:
        record = status_run(run)
        typer.echo(record.model_dump_json(indent=2), nl=False)
        typer.echo("")
    except (ValueError, OSError) as exc:
        _error(exc)


@execution_app.command("remote-status")
def remote_status_command(
    profile: Path = typer.Option(..., "--profile", exists=True, dir_okay=False),
    run: Path = typer.Option(..., "--run", exists=True, file_okay=False),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Read the recorded remote PID and bounded VASP status files; never signals."""

    try:
        if output and (output.exists() or output.is_symlink()):
            raise FileExistsError("Report exists; choose a new output path")
        settings = ClusterProfile.model_validate_json(profile.read_bytes())
        report = remote_status(settings, status_run(run))
        if output:
            write_json_report(report, output)
        typer.echo(json_bytes(report).decode("utf-8"), nl=False)
    except (ValueError, OSError) as exc:
        _error(exc)


@execution_app.command("fetch")
def fetch_command(
    profile: Path = typer.Option(..., "--profile", exists=True, dir_okay=False),
    run: Path = typer.Option(..., "--run", exists=True, file_okay=False),
    destination: Path = typer.Option(..., "--destination"),
    files: list[str] = typer.Option(..., "--file"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Fetch exact allowlisted result files and verify remote/local hashes."""

    try:
        if output and (output.exists() or output.is_symlink()):
            raise FileExistsError("Report exists; choose a new output path")
        settings = ClusterProfile.model_validate_json(profile.read_bytes())
        report = fetch_results(settings, status_run(run), destination.resolve(), files)
        if output:
            write_json_report(report, output)
        typer.echo(json_bytes(report).decode("utf-8"), nl=False)
    except (ValueError, OSError) as exc:
        _error(exc)


@execution_app.command("artifact-release")
def artifact_release_command(
    bundle: Path = typer.Option(..., "--bundle", exists=True, file_okay=False),
    profile: Path = typer.Option(..., "--profile", exists=True, dir_okay=False),
    source_run: Path = typer.Option(..., "--source-run", exists=True, file_okay=False),
    source_status: Path = typer.Option(..., "--source-status", exists=True, dir_okay=False),
    fetch_report: Path = typer.Option(..., "--fetch-report", exists=True, dir_okay=False),
    target_transfer_report: Path = typer.Option(
        ..., "--target-transfer-report", exists=True, dir_okay=False
    ),
    target_step: str = typer.Option(..., "--target-step"),
    confirm: str = typer.Option(..., "--confirm", help="Exact confirmation phrase"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Promote one verified, converged SCF CHGCAR to a declared dependent."""

    try:
        if output.exists() or output.is_symlink():
            raise FileExistsError("Report exists; choose a new output path")
        evidence = release_chgcar(
            bundle=bundle,
            profile=ClusterProfile.model_validate_json(profile.read_bytes()),
            source_run=status_run(source_run),
            source_status=source_status,
            fetch_report=fetch_report,
            target_transfer_report=target_transfer_report,
            target_step=target_step,
            confirm=confirm,
        )
        write_json_report(evidence.model_dump(mode="json"), output)
        typer.echo(evidence.model_dump_json(indent=2), nl=False)
        typer.echo("")
    except (ValueError, OSError) as exc:
        _error(exc)


@execution_app.command("cancel")
def cancel_command(
    run: Path = typer.Option(..., "--run", exists=True, file_okay=False),
) -> None:
    """Cancel only a locally owned simulation. Refuses remote or unmatched PIDs."""
    try:
        record = cancel_run(run)
        typer.echo(record.model_dump_json(indent=2), nl=False)
        typer.echo("")
    except (ValueError, OSError) as exc:
        _error(exc)


@execution_app.command("transfer")
def transfer(
    bundle: Path = typer.Option(..., "--bundle", exists=True, file_okay=False),
    profile: Path = typer.Option(..., "--profile", exists=True, dir_okay=False),
    request: Path = typer.Option(..., "--request", exists=True, dir_okay=False),
    connect: bool = typer.Option(False, "--connect", help="Explicitly upload through fixed SFTP and verify hashes."),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Plan by default; --connect performs only the controlled package transfer."""
    try:
        settings = ClusterProfile.model_validate_json(profile.read_bytes())
        transfer_request = TransferRequest.model_validate_json(request.read_bytes())
        report = transfer_bundle(bundle, settings, transfer_request) if connect else build_transfer_plan(bundle, settings, transfer_request)
        if output:
            write_transfer_report(report, output)
        typer.echo(json_bytes(report).decode("utf-8"), nl=False)
    except (ValueError, OSError) as exc:
        _error(exc)
