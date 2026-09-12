# ruff: noqa: B008 - Typer requires option declarations in function signatures.
from __future__ import annotations

import json
from pathlib import Path

import typer

from magnetic_vasp_agent.agent.approval import ApprovalPolicy
from magnetic_vasp_agent.agent.runner import run_plan
from magnetic_vasp_agent.agent.tool_models import AgentPlan
from magnetic_vasp_agent.agent.tools import default_registry

agent_app = typer.Typer(
    no_args_is_help=True,
    help="Expose the deterministic core as gated tools; cluster execution stays disabled.",
)


def _error(exc: Exception) -> None:
    typer.echo(f"Agent error: {exc}", err=True)
    raise typer.Exit(2)


def _write_json(payload: str, output: Path | None) -> None:
    if output is not None:
        if output.exists() or output.is_symlink():
            raise FileExistsError("output exists; choose a new path")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    typer.echo(payload)


@agent_app.command("tools")
def list_tools() -> None:
    """List registered tools with their risk tiers; read-only, no LLM."""
    try:
        specs = [spec.model_dump(mode="json") for spec in default_registry().specs()]
        typer.echo(json.dumps(specs, indent=2, ensure_ascii=False))
    except (ValueError, OSError) as exc:
        _error(exc)


@agent_app.command("run-plan")
def run_typed_plan(
    plan: Path = typer.Option(..., "--plan", exists=True, dir_okay=False),
    policy: Path | None = typer.Option(None, "--policy", exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Run a hand-written typed plan through the registry and approval gate."""
    try:
        agent_plan = AgentPlan.model_validate_json(plan.read_bytes())
        approval_policy = (
            ApprovalPolicy.model_validate_json(policy.read_bytes())
            if policy
            else ApprovalPolicy()
        )
        transcript = run_plan(agent_plan, policy=approval_policy)
        _write_json(transcript.model_dump_json(indent=2), output)
    except (ValueError, OSError) as exc:
        _error(exc)
    if transcript.status != "completed":
        raise typer.Exit(1)


@agent_app.command("chat")
def chat(
    goal: str = typer.Option(..., "--goal"),
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    policy: Path | None = typer.Option(None, "--policy", exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Turn a natural-language goal into gated tool calls via LangGraph (needs 'agent' extra)."""
    try:
        from magnetic_vasp_agent.agent.graph import (
            langgraph_available,
            run_goal_from_config_path,
        )

        if not langgraph_available():
            raise RuntimeError(
                "langgraph is not installed; install the optional 'agent' extra"
            )
        approval_policy = (
            ApprovalPolicy.model_validate_json(policy.read_bytes())
            if policy
            else ApprovalPolicy()
        )
        result = run_goal_from_config_path(goal, config, policy=approval_policy)
        _write_json(result.model_dump_json(indent=2), output)
    except (ValueError, OSError, RuntimeError) as exc:
        _error(exc)
    if result.pending_approvals:
        raise typer.Exit(1)
