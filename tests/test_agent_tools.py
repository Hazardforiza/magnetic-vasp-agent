from __future__ import annotations

import pytest

from magnetic_vasp_agent.agent.tool_models import RiskTier, ToolSpec
from magnetic_vasp_agent.agent.tools import (
    StructureShowInput,
    ToolRegistry,
    default_registry,
)

EXPECTED_TOOLS = {
    "structure.show": RiskTier.READ_ONLY,
    "incar.show": RiskTier.READ_ONLY,
    "incar.validate": RiskTier.READ_ONLY,
    "kpoints.show": RiskTier.READ_ONLY,
    "kpoints.validate": RiskTier.READ_ONLY,
    "mcif.show": RiskTier.READ_ONLY,
    "potcar.select": RiskTier.READ_ONLY,
    "workflow.validate": RiskTier.READ_ONLY,
    "workflow.inspect": RiskTier.READ_ONLY,
    "knowledge.search": RiskTier.READ_ONLY,
    "knowledge.answer": RiskTier.READ_ONLY,
    "execution.probe": RiskTier.READ_ONLY,
    "execution.plan": RiskTier.READ_ONLY,
    "postprocess.parse": RiskTier.READ_ONLY,
    "potcar.build": RiskTier.LOCAL_WRITE,
    "incar.build": RiskTier.LOCAL_WRITE,
    "kpoints.build": RiskTier.LOCAL_WRITE,
    "workflow.build": RiskTier.LOCAL_WRITE,
    "execution.simulate": RiskTier.LOCAL_WRITE,
    "knowledge.evidence": RiskTier.LOCAL_WRITE,
    "knowledge.draft": RiskTier.LOCAL_WRITE,
    "execution.transfer_plan": RiskTier.CLUSTER_PLAN,
}


def test_default_registry_has_expected_tools() -> None:
    registry = default_registry()
    assert set(registry.names()) == set(EXPECTED_TOOLS)
    by_name = {spec.name: spec for spec in registry.specs()}
    for name, tier in EXPECTED_TOOLS.items():
        assert by_name[name].risk_tier == tier


def test_no_cluster_execute_tool_is_registered() -> None:
    registry = default_registry()
    for spec in registry.specs():
        assert spec.risk_tier != RiskTier.CLUSTER_EXECUTE


def test_no_shell_like_tool_names_exist() -> None:
    # Guard against tools that would run arbitrary shell or drive the cluster.
    # The read-only "execution.*" preview/plan namespace is explicitly allowed.
    registry = default_registry()
    forbidden = ("shell", "ssh", "mpirun", "launch", "cancel", "retry")
    for name in registry.names():
        assert not any(token in name for token in forbidden)
    # The only cluster tool is the offline transfer plan; nothing connects.
    cluster_tools = {
        spec.name
        for spec in registry.specs()
        if spec.risk_tier == RiskTier.CLUSTER_PLAN
    }
    assert cluster_tools == {"execution.transfer_plan"}


def test_registering_cluster_execute_is_rejected() -> None:
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="cluster_execute"):
        registry.register(
            name="execution.launch",
            description="should never exist",
            risk_tier=RiskTier.CLUSTER_EXECUTE,
            produces_files=False,
            input_model=StructureShowInput,
            run=lambda data: {},
        )


def test_duplicate_registration_is_rejected() -> None:
    registry = ToolRegistry()
    registry.register(
        name="structure.show",
        description="parse a POSCAR",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=StructureShowInput,
        run=lambda data: {},
    )
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(
            name="structure.show",
            description="parse a POSCAR again",
            risk_tier=RiskTier.READ_ONLY,
            produces_files=False,
            input_model=StructureShowInput,
            run=lambda data: {},
        )


def test_get_unknown_tool_raises_keyerror() -> None:
    registry = default_registry()
    with pytest.raises(KeyError, match="unknown tool"):
        registry.get("ssh.run")


def test_specs_are_json_serializable_and_carry_schema() -> None:
    registry = default_registry()
    for spec in registry.specs():
        assert isinstance(spec, ToolSpec)
        payload = spec.model_dump(mode="json")
        assert payload["input_schema"]["additionalProperties"] is False
        assert isinstance(spec.description, str) and spec.description
