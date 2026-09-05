from __future__ import annotations

import hashlib
from pathlib import Path

from magnetic_vasp_agent.models import (
    CalculationGoal,
    CandidateComposition,
    PotcarBuildManifest,
    PotentialChoice,
    PotentialPlan,
    PotentialPolicy,
    PotentialRecord,
    PotentialRegistry,
)


def load_policy(path: Path) -> PotentialPolicy:
    return PotentialPolicy.model_validate_json(path.read_text(encoding="utf-8"))


def _records_by_name(registry: PotentialRegistry) -> dict[str, PotentialRecord]:
    return {record.potential: record for record in registry.records}


def select_by_policy(
    candidate: CandidateComposition,
    goal: CalculationGoal,
    registry: PotentialRegistry,
    policy: PotentialPolicy,
) -> PotentialPlan:
    if goal.xc_family != registry.family or policy.family != registry.family:
        raise ValueError(
            "goal, policy, and POTCAR registry must use the same XC family"
        )

    records = _records_by_name(registry)
    choices: list[PotentialChoice] = []
    for element in candidate.species_order:
        preferred = policy.preferred.get(element)
        if preferred is not None:
            potential_name = preferred.potential
            rationale = preferred.rationale
            evidence = preferred.evidence
            confidence = preferred.confidence
        elif element in records:
            potential_name = element
            rationale = "Exact-element fallback; no reviewed preference is available."
            evidence = []
            confidence = 0.5
        else:
            raise ValueError(
                f"no reviewed or exact-element potential is available for {element}"
            )

        record = records.get(potential_name)
        if record is None:
            raise ValueError(
                f"policy selected {potential_name}, but it is absent from the registry"
            )
        if record.element != element:
            raise ValueError(
                f"potential {potential_name} belongs to {record.element}, not {element}"
            )
        choices.append(
            PotentialChoice(
                element=element,
                potential=potential_name,
                rationale=rationale,
                confidence=confidence,
                evidence=evidence,
            )
        )

    plan = PotentialPlan(
        candidate_id=candidate.candidate_id,
        species_order=candidate.species_order,
        goal=goal,
        choices=choices,
    )
    validate_plan(plan, registry)
    return plan


def validate_plan(
    plan: PotentialPlan, registry: PotentialRegistry
) -> list[PotentialRecord]:
    if plan.goal.xc_family != registry.family:
        raise ValueError("plan and registry use different XC families")

    records = _records_by_name(registry)
    selected: list[PotentialRecord] = []
    for expected_element, choice in zip(
        plan.species_order, plan.choices, strict=True
    ):
        record = records.get(choice.potential)
        if record is None:
            raise ValueError(f"unknown potential: {choice.potential}")
        if choice.element != expected_element or record.element != expected_element:
            raise ValueError(
                f"potential {choice.potential} does not match {expected_element}"
            )
        if record.family != plan.goal.xc_family:
            raise ValueError(
                f"potential {choice.potential} belongs to {record.family}, "
                f"not {plan.goal.xc_family}"
            )
        selected.append(record)
    return selected


def prepare_potcar(
    plan: PotentialPlan,
    registry: PotentialRegistry,
) -> tuple[bytes, PotcarBuildManifest]:
    """Verify licensed sources and prepare bytes without writing to disk."""
    selected = validate_plan(plan, registry)
    root = Path(registry.root).resolve()
    payloads: list[bytes] = []

    for record in selected:
        source = (root / Path(record.relative_path)).resolve()
        if not source.is_relative_to(root):
            raise ValueError(f"potential path escapes registry root: {source}")
        payload = source.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != record.sha256:
            raise ValueError(
                f"POTCAR source changed after inventory: {record.potential}"
            )
        payloads.append(payload)

    combined = b"".join(payloads)
    manifest = PotcarBuildManifest(
        candidate_id=plan.candidate_id,
        family=plan.goal.xc_family,
        species_order=plan.species_order,
        potential_names=[choice.potential for choice in plan.choices],
        source_sha256=[record.sha256 for record in selected],
        output_sha256=hashlib.sha256(combined).hexdigest(),
    )
    return combined, manifest


def assemble_potcar(
    plan: PotentialPlan,
    registry: PotentialRegistry,
    output: Path,
) -> PotcarBuildManifest:
    combined, manifest = prepare_potcar(plan, registry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(combined)
    return manifest
