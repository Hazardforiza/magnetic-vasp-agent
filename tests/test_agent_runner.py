from __future__ import annotations

from pathlib import Path

from magnetic_vasp_agent.agent.approval import ApprovalPolicy
from magnetic_vasp_agent.agent.runner import evaluate_call, run_plan
from magnetic_vasp_agent.agent.tool_models import (
    AgentPlan,
    ApprovalState,
    sha256_of,
    ToolCall,
)

FIXTURES = Path(__file__).parent / "fixtures"
MN3AL_POSCAR = str(FIXTURES / "Mn3Al" / "POSCAR")
MN3AL_KPOINTS = str(FIXTURES / "Mn3Al" / "KPOINTS")
FEO_MCIF = str(FIXTURES / "generated_mcif" / "candidate_01_FeO_BNS2_7.mcif")


def test_read_only_plan_completes_with_provenance() -> None:
    plan = AgentPlan(
        goal="inspect Mn3Al inputs and one FeO candidate",
        calls=[
            ToolCall(call_id="T1", tool="structure.show", arguments={"poscar": MN3AL_POSCAR}),
            ToolCall(call_id="T2", tool="kpoints.show", arguments={"kpoints": MN3AL_KPOINTS}),
            ToolCall(call_id="T3", tool="mcif.show", arguments={"mcif": FEO_MCIF}),
        ],
    )
    transcript = run_plan(plan)

    assert transcript.status == "completed"
    assert transcript.plan_sha256 == plan.digest()
    assert transcript.cluster_execution == "disabled"
    assert transcript.human_review_required is True
    assert [r.call_id for r in transcript.results] == ["T1", "T2", "T3"]
    for result in transcript.results:
        assert result.status == "ok"
        assert result.approval_state == ApprovalState.EXECUTED
        assert result.output is not None
        assert result.output_sha256 == sha256_of(result.output)


def test_unknown_tool_halts_and_is_blocked() -> None:
    plan = AgentPlan(
        goal="try to escape",
        calls=[ToolCall(call_id="T1", tool="ssh.run", arguments={"cmd": "rm -rf /"})],
    )
    transcript = run_plan(plan)
    assert transcript.status == "halted"
    assert transcript.results[0].status == "blocked"
    assert "unknown tool" in transcript.results[0].message


def test_bad_arguments_are_validation_errors() -> None:
    plan = AgentPlan(
        goal="bad args",
        calls=[ToolCall(call_id="T1", tool="structure.show", arguments={"nope": 1})],
    )
    transcript = run_plan(plan)
    assert transcript.status == "halted"
    assert transcript.results[0].status == "validation_error"
    assert transcript.results[0].approval_state == ApprovalState.REJECTED


def test_missing_file_becomes_error_not_crash() -> None:
    plan = AgentPlan(
        goal="missing file",
        calls=[ToolCall(call_id="T1", tool="structure.show", arguments={"poscar": "no/such/POSCAR"})],
    )
    transcript = run_plan(plan)
    assert transcript.status == "halted"
    assert transcript.results[0].status == "error"


def test_run_halts_at_first_failure() -> None:
    plan = AgentPlan(
        goal="second call never runs",
        calls=[
            ToolCall(call_id="T1", tool="structure.show", arguments={"poscar": "no/POSCAR"}),
            ToolCall(call_id="T2", tool="kpoints.show", arguments={"kpoints": MN3AL_KPOINTS}),
        ],
    )
    transcript = run_plan(plan)
    assert transcript.status == "halted"
    assert len(transcript.results) == 1


def test_cluster_plan_requires_approval_and_halts() -> None:
    plan = AgentPlan(
        goal="transfer without approval",
        calls=[
            ToolCall(
                call_id="T1",
                tool="execution.transfer_plan",
                arguments={"bundle": "b", "profile": "p", "request": "r"},
            )
        ],
    )
    transcript = run_plan(plan)
    assert transcript.status == "halted"
    assert transcript.results[0].status == "blocked"
    assert transcript.results[0].approval_state == ApprovalState.AWAITING_APPROVAL


def test_cluster_plan_with_approval_passes_the_gate() -> None:
    # Once approved the gate no longer blocks; the tool then fails on missing
    # files rather than being refused, proving approval (not execution) succeeded.
    call = ToolCall(
        call_id="T1",
        tool="execution.transfer_plan",
        arguments={"bundle": "b", "profile": "p", "request": "r"},
    )
    policy = ApprovalPolicy(approved_cluster_plan_call_ids=["T1"], approver="researcher")
    approval, result = evaluate_call(call, policy=policy)
    assert approval.state == ApprovalState.APPROVED
    assert result.status != "blocked"


def test_local_write_tool_produces_files(tmp_path) -> None:
    output = tmp_path / "sim"
    plan = AgentPlan(
        goal="run a local simulation",
        calls=[
            ToolCall(
                call_id="T1",
                tool="execution.simulate",
                arguments={"output": str(output), "scenario": "complete"},
            )
        ],
    )
    transcript = run_plan(plan)
    assert transcript.status == "completed"
    assert transcript.results[0].status == "ok"
    assert transcript.results[0].approval_state == ApprovalState.EXECUTED
    assert output.exists()


def test_local_write_can_be_disabled_by_policy(tmp_path) -> None:
    output = tmp_path / "sim"
    plan = AgentPlan(
        goal="blocked local write",
        calls=[
            ToolCall(
                call_id="T1",
                tool="execution.simulate",
                arguments={"output": str(output), "scenario": "complete"},
            )
        ],
    )
    transcript = run_plan(plan, policy=ApprovalPolicy(allow_local_writes=False))
    assert transcript.status == "halted"
    assert transcript.results[0].status == "blocked"
    assert not output.exists()
