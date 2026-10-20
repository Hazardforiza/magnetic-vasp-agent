"""Storage edge cases whose outcome matters to visible library readiness."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import magnetic_vasp_agent.knowledge.library as library_module
from magnetic_vasp_agent.knowledge.library import KnowledgeLibrary


def test_metadata_is_searchable_but_quotes_remain_source_text(tmp_path: Path) -> None:
    library = KnowledgeLibrary(tmp_path)
    source = "Compare interpolation with the original band structure."
    item = library.import_file("local", "method.txt", source.encode(),
                               title="DisentangleMetadataToken", doi="10.1234/DOIMetadataToken",
                               tags=["NiOMetadataToken"])
    library.index("local", item["id"])
    for query in ("DisentangleMetadataToken", "DOIMetadataToken", "NiOMetadataToken"):
        found = library.search(query, scope="local")["evidence"]
        assert found and found[0]["item_id"] == item["id"]
        assert found[0]["quote"] == source


def test_corrupt_active_index_is_visible_and_explicit_confirm_repairs_it(tmp_path: Path) -> None:
    library = KnowledgeLibrary(tmp_path)
    item = library.import_file("local", "method.txt", b"CorruptionRepairToken convergence notes.")
    library.index("local", item["id"])
    pointer = json.loads((library.local / "current.json").read_text())
    documents = library.local / "snapshots" / pointer["generation"] / "index" / "documents.jsonl"
    documents.write_text("damaged", encoding="utf-8")
    status = library.overview()["local"]
    assert status["available"] is False and status["index_ready"] is False
    assert status["indexed_count"] == 0 and status["item_count"] == 1
    assert status["error"]
    result = library.search("CorruptionRepairToken")
    assert result["evidence"] == [] and result["warnings"]
    library.index("local", item["id"])
    assert library.overview()["local"]["index_ready"] is True
    assert library.search("CorruptionRepairToken")["evidence"]


def test_same_case_notes_with_distinct_result_provenance_are_not_deduplicated(tmp_path: Path) -> None:
    library = KnowledgeLibrary(tmp_path)
    first = library.create_case("local", "Trial 1", "Same notes.", provenance={"result": "first"})
    second = library.create_case("local", "Trial 2", "Same notes.", provenance={"result": "second"})
    assert first["id"] != second["id"]


def test_publishing_existing_shared_draft_confirms_without_creating_duplicate(tmp_path: Path) -> None:
    library = KnowledgeLibrary(tmp_path / "web")
    shared = tmp_path / "shared"
    shared.mkdir()
    library.configure_shared(str(shared))
    payload = b"ExistingSharedDraftToken review notes."
    draft = library.import_file("shared", "draft.txt", payload, title="Same reviewed source")
    local = library.import_file("local", "local.txt", payload, title="Same reviewed source")
    library.index("local", local["id"])
    published = library.publish(local["id"])
    assert published["id"] == draft["id"] and published["status"] == "indexed"
    assert library.overview()["shared"]["item_count"] == 1


def test_republication_cannot_confirm_shared_text_edited_since_original_publish(tmp_path: Path) -> None:
    library = KnowledgeLibrary(tmp_path / "web")
    shared = tmp_path / "shared"
    shared.mkdir()
    library.configure_shared(str(shared))
    local = library.import_file("local", "original.txt", b"ApprovedOriginalToken original body A.")
    library.index("local", local["id"])
    published = library.publish(local["id"])
    library.update("shared", published["id"], {"text": "UnconfirmedEditedToken body B needs review."})
    replacement = library.publish(local["id"])
    assert replacement["id"] != published["id"]
    assert replacement["text"] == local["text"] and replacement["status"] == "indexed"
    assert library.get("shared", published["id"])["status"] == "draft"
    assert library.search("UnconfirmedEditedToken", scope="shared")["evidence"] == []
    assert library.search("ApprovedOriginalToken", scope="shared")["evidence"]


def test_separate_process_writers_preserve_both_confirmed_items(tmp_path: Path) -> None:
    shared = tmp_path / "team-drive"
    shared.mkdir()
    reader = KnowledgeLibrary(tmp_path / "reader")
    reader.configure_shared(str(shared))
    code = """
import sys
from pathlib import Path
from magnetic_vasp_agent.knowledge.library import KnowledgeLibrary
library=KnowledgeLibrary(Path(sys.argv[1]))
library.configure_shared(sys.argv[2])
item=library.import_file('shared','notes.txt',('CrossProcessJointToken '+sys.argv[3]).encode())
library.index('shared',item['id'])
print(item['id'])
"""
    processes = [subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path / name), str(shared), name],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) for name in ("alice", "bob")]
    identifiers = set()
    for process in processes:
        stdout, stderr = process.communicate(timeout=45)
        assert process.returncode == 0, stderr
        identifiers.add(stdout.strip())
    assert len(identifiers) == 2
    evidence = reader.search("CrossProcessJointToken", scope="shared")["evidence"]
    assert {item["item_id"] for item in evidence} == identifiers


def test_import_does_not_follow_shared_path_reconfiguration_mid_request(tmp_path: Path, monkeypatch) -> None:
    library = KnowledgeLibrary(tmp_path / "web")
    first, second = tmp_path / "team-a", tmp_path / "team-b"
    first.mkdir(); second.mkdir()
    library.configure_shared(str(first))
    original = library_module._extract

    def switch_during_extraction(filename, payload):
        library.configure_shared(str(second))
        return original(filename, payload)

    monkeypatch.setattr(library_module, "_extract", switch_during_extraction)
    with pytest.raises(library_module.LibraryError) as error:
        library.import_file("shared", "notes.txt", b"PathSwitchToken content")
    assert error.value.status_code == 409
    assert not (first / "magvasp-knowledge" / "current.json").exists()
    assert not (second / "magvasp-knowledge" / "current.json").exists()


def test_publish_does_not_follow_shared_path_reconfiguration_mid_request(tmp_path: Path, monkeypatch) -> None:
    library = KnowledgeLibrary(tmp_path / "web")
    first, second = tmp_path / "team-a", tmp_path / "team-b"
    first.mkdir(); second.mkdir()
    library.configure_shared(str(first))
    item = library.import_file("local", "notes.txt", b"PathSwitchPublicationToken content")
    library.index("local", item["id"])
    original = library.get

    def switch_during_local_read(scope, item_id):
        library.configure_shared(str(second))
        return original(scope, item_id)

    monkeypatch.setattr(library, "get", switch_during_local_read)
    with pytest.raises(library_module.LibraryError) as error:
        library.publish(item["id"])
    assert error.value.status_code == 409
    assert not (first / "magvasp-knowledge" / "current.json").exists()
    assert not (second / "magvasp-knowledge" / "current.json").exists()


def test_edited_case_notes_cannot_rewrite_the_captured_source_snapshot(tmp_path: Path) -> None:
    library = KnowledgeLibrary(tmp_path)
    provenance = {"result": {"id": "a" * 32, "summary": {
        "status_marker": "SnapshotUnconvergedToken", "electronic_converged": False,
    }}}
    item = library.create_case("local", "Trial", "Original note about failed convergence.", provenance=provenance)
    assert "SnapshotUnconvergedToken" not in item["text"]
    library.index("local", item["id"])
    library.update("local", item["id"], {"text": "EditedNoteConvergedToken: I think this converged."})
    library.index("local", item["id"])
    snapshot = library.search("SnapshotUnconvergedToken", scope="local")["evidence"]
    notes = library.search("EditedNoteConvergedToken", scope="local")["evidence"]
    assert snapshot and snapshot[0]["source_segment"] == "result_snapshot"
    assert "false" in snapshot[0]["quote"]
    assert snapshot[0]["source_segment_label"] == "关联来源快照 · 结果"
    assert notes and notes[0]["source_segment"] == "notes"
    assert notes[0]["source_segment_label"] == "笔记"
    assert all(entry["source_validated"] is False for entry in [*snapshot, *notes])
    assert library.get("local", item["id"])["provenance"] == provenance
