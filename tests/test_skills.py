from __future__ import annotations

from pathlib import Path

import pytest

from magnetic_vasp_agent.agent.tools import default_registry
from magnetic_vasp_agent.knowledge import build_inventory, load_catalog
from magnetic_vasp_agent.skills import list_skills, read_skill_file, show_skill
from magnetic_vasp_agent.skills_models import SkillCatalog, SkillRecord

ROOT = Path(__file__).parents[1]


def write_skill(tmp_path: Path, skill_id: str, body: str, extra: str | None = None) -> Path:
    root = tmp_path / "skills" / skill_id
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        f"---\nname: {skill_id}\ndescription: demo skill\n---\n\n{body}\n",
        encoding="utf-8",
        newline="\n",
    )
    if extra is not None:
        references = root / "references"
        references.mkdir()
        (references / "notes.md").write_text(extra, encoding="utf-8", newline="\n")
        (references / "run.py").write_text("print('no')\n", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        SkillCatalog(
            schema_version=1,
            base_directory=".",
            skills=[
                SkillRecord(
                    id=skill_id,
                    title=skill_id,
                    path=f"skills/{skill_id}",
                    domains=["vasp"],
                )
            ],
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return catalog


def test_project_skill_catalog_lists_user_skills_and_blocks_scripts() -> None:
    catalog = ROOT / "config" / "skills" / "catalog.json"
    payload = SkillCatalog.model_validate_json(catalog.read_bytes())
    ids = {record.id for record in payload.skills}
    assert "first-principles-calculation" in ids
    assert "analyze-pdf-reference" in ids
    assert all(record.scripts_allowed is False for record in payload.skills)
    summaries = list_skills(catalog)
    by_id = {item.id: item for item in summaries}
    assert "references/workflow.md" in by_id["first-principles-calculation"].reference_files
    shown = show_skill(catalog, "first-principles-calculation")
    assert "执行边界" in shown.text
    assert shown.scripts_allowed is False
    workflow = read_skill_file(
        catalog, "first-principles-calculation", "references/workflow.md"
    )
    assert "SCF" in workflow.text or "scf" in workflow.text.casefold()


def test_skill_tools_refuse_scripts(tmp_path) -> None:
    catalog = write_skill(tmp_path, "demo-skill", "# Demo\nDo not run scripts.", extra="# note\n")
    tool = default_registry().get("skill.read")
    with pytest.raises(ValueError, match="blocked|only Markdown"):
        tool.run(tool.input_model(skill_id="demo-skill", path="references/run.py", catalog=str(catalog)))
    listed = default_registry().get("skill.list").run(
        default_registry().get("skill.list").input_model(catalog=str(catalog))
    )
    assert listed["scripts_executable"] is False
    assert listed["skills"][0]["id"] == "demo-skill"


def test_knowledge_inventory_includes_reference_demo_and_skips_system_skills() -> None:
    catalog_path = ROOT / "config" / "knowledge" / "sources.json"
    catalog, payload = load_catalog(catalog_path)
    inventory = build_inventory(catalog, catalog_path, payload)
    sources = {item.source_id: item for item in inventory.sources}
    assert sources["reference-demo-library"].documents
    assert any(
        document.relative_path.endswith("reference_data.txt")
        for document in sources["reference-demo-library"].documents
    )
    skill_paths = [document.relative_path for document in sources["agent-skill-library"].documents]
    assert any(path.endswith("first-principles-calculation/SKILL.md") for path in skill_paths)
    assert all(".system" not in path for path in skill_paths)
    assert all("/scripts/" not in path for path in skill_paths)
