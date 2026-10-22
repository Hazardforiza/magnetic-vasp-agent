"""Restricted, project-bound tools for resource planning in the web UI.

The selected resources are a constraint, not a resource grant.  This registry
can inspect a bundle and prepare a resource assessment, but cannot generate a
different request, approve resources, transfer files, or launch a calculation.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from magnetic_vasp_agent.agent.tools import Tool, ToolRegistry, default_registry
from magnetic_vasp_agent.execution_models import (
    ClusterProfile,
    ExecutionRequest,
    ProbeSnapshot,
)
from magnetic_vasp_agent.workflow import digest


def project_registry(
    profile_path: Path,
    bundle: Path,
    request: ExecutionRequest,
    *,
    request_path: Path,
    work_dir: Path | None = None,
    validate_context: Callable[[], None] | None = None,
) -> ToolRegistry:
    """Bind planning tools to one saved project resource selection.

    ``request_path`` must already contain the server-created ``request``.
    Live observations and assessments may only be written to ``node.probe.json``
    and ``node.resource-plan.json`` in ``work_dir`` (the request's directory by
    default).  Existing approval tiers are retained.  ``validate_context`` may
    reject a session whose project selection has since been changed.
    """

    profile_file = profile_path.resolve()
    bundle_dir = bundle.resolve()
    request_file = request_path.resolve()
    output_dir = (work_dir or request_file.parent).resolve()
    probe_file = (output_dir / "node.probe.json").resolve()
    plan_file = (output_dir / "node.resource-plan.json").resolve()
    expected_request = request.model_copy(deep=True)
    profile_bytes = profile_file.read_bytes()
    profile = ClusterProfile.model_validate_json(profile_bytes)

    if (
        request.node == profile.gateway_hostname
        or request.node not in profile.allowed_nodes
    ):
        raise ValueError("PROJECT_NODE_FORBIDDEN: select an allowlisted compute node")
    if output_dir == bundle_dir or output_dir.is_relative_to(bundle_dir):
        raise ValueError("PROJECT_OUTPUT_FORBIDDEN: planning outputs must be outside the bundle")
    if len({profile_file, request_file, probe_file, plan_file}) != 4:
        raise ValueError("PROJECT_OUTPUT_FORBIDDEN: planning outputs must not replace inputs")

    def guard() -> None:
        if validate_context is not None:
            validate_context()
        if profile_file.read_bytes() != profile_bytes:
            raise ValueError("PROJECT_PROFILE_CHANGED: start a new project planning session")
        saved = ExecutionRequest.model_validate_json(request_file.read_bytes())
        if saved != expected_request:
            raise ValueError("PROJECT_REQUEST_CHANGED: saved resources differ from this session")

    def require_path(value: str, expected: Path, label: str) -> None:
        if Path(value).resolve() != expected:
            raise ValueError(f"PROJECT_{label}_MISMATCH: use this project's bound {label.lower()}")

    def check_snapshot(value: str) -> None:
        require_path(value, probe_file, "SNAPSHOT")
        snapshot = ProbeSnapshot.model_validate_json(probe_file.read_bytes())
        if snapshot.requested_node != expected_request.node:
            raise ValueError("PROJECT_NODE_MISMATCH: snapshot belongs to another node")

    def bind(tool: Tool) -> Callable[[BaseModel], dict[str, Any]]:
        def run(data: BaseModel) -> dict[str, Any]:
            guard()
            values = data.model_dump()
            if "profile" in values:
                require_path(values["profile"], profile_file, "PROFILE")
            if "bundle" in values:
                require_path(values["bundle"], bundle_dir, "BUNDLE")
            if "request" in values:
                require_path(values["request"], request_file, "REQUEST")
            if "node" in values and values["node"] != expected_request.node:
                raise ValueError("PROJECT_NODE_MISMATCH: use the project's selected node")
            if values.get("snapshot") is not None:
                check_snapshot(values["snapshot"])
            if "output" in values:
                expected = probe_file if tool.spec.name == "execution.probe_live" else plan_file
                require_path(values["output"], expected, "OUTPUT")
            result = tool.run(data)
            if tool.spec.name == "execution.probe_live":
                # Keep complete evidence locally for deterministic assessment;
                # process identities and commands must never enter LLM context
                # or the browser's session transcript.
                snapshot = ProbeSnapshot.model_validate_json(probe_file.read_bytes())
                online = [cpu for cpu in snapshot.cpus if cpu.online]
                accessible = [cpu for cpu in online if cpu.cpu in snapshot.allowed_cpu_ids]
                return {
                    "mode": "project_node_observation",
                    "node": snapshot.requested_node,
                    "observed_at": snapshot.observed_at.isoformat(),
                    "snapshot_path": str(probe_file),
                    "snapshot_sha256": digest(probe_file.read_bytes()),
                    "summary": {
                        "logical_cpus": len(online),
                        "physical_cores": len({(cpu.socket, cpu.core) for cpu in online}),
                        "accessible_physical_cores": len({(cpu.socket, cpu.core) for cpu in accessible}),
                        "load1": snapshot.load1,
                        "mem_available_kib": snapshot.mem_available_kib,
                        "process_count": len(snapshot.processes),
                        "active_process_count": sum(item.cpu_percent >= 5 for item in snapshot.processes),
                        "total_cpu_percent": round(sum(item.cpu_percent for item in snapshot.processes), 2),
                        "module_loaded": snapshot.module_loaded,
                        "mpi_path_observed": snapshot.mpi_path == profile.mpi_path,
                        "executables": snapshot.executables,
                    },
                    "observation_note": "只读瞬时观察，不代表资源分配或启动授权。",
                }
            return result

        return run

    guard()
    source = default_registry()
    registry = ToolRegistry()
    for name in (
        "workflow.inspect",
        "workflow.review_card",
        "execution.probe",
        "execution.probe_live",
        "execution.plan",
        "execution.resource_plan",
        "skill.list",
        "skill.show",
        "skill.read",
    ):
        tool = source.get(name)
        registry.register(
            name=name,
            description=tool.spec.description,
            risk_tier=tool.spec.risk_tier,
            produces_files=tool.spec.produces_files,
            input_model=tool.input_model,
            run=bind(tool),
        )
    return registry
