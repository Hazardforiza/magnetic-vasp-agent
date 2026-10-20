from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from magnetic_vasp_agent.cli import app
from magnetic_vasp_agent import knowledge_cli
from magnetic_vasp_agent.knowledge import embedding as embedding_module
from magnetic_vasp_agent.knowledge import retrieval as retrieval_module
from magnetic_vasp_agent.knowledge import build_inventory, load_catalog, write_inventory
from magnetic_vasp_agent.knowledge.models import (
    KnowledgeCatalog,
    KnowledgeSource,
    KnowledgeSourceType,
    MagneticMode,
    RagEvaluationSuite,
)
from magnetic_vasp_agent.knowledge.retrieval import (
    build_bm25_index,
    build_evidence_package,
    build_vector_index,
    evaluate_bm25,
    evaluate_vector,
    search_bm25,
    search_vector,
    tokenize,
    write_evidence_package,
)
from magnetic_vasp_agent.knowledge.retrieval_models import (
    RetrievalFilter,
    VectorIndexManifest,
)
from magnetic_vasp_agent.knowledge.wiki import (
    api_url_for_page,
    build_chunks,
    capture_wiki,
    page_name_from_source,
    parse_api_payload,
    verify_wiki_capture,
)

ROOT = Path(__file__).parents[1]


def source(
    source_id: str,
    path: str,
    *,
    location_type: str = "local_file",
    required: bool = True,
    extensions: list[str] | None = None,
) -> dict:
    return {
        "id": source_id,
        "title": source_id,
        "source_type": "personal_note",
        "location_type": location_type,
        "content_handling": "local_only",
        "path": path,
        "required": required,
        "validated": False,
        "extensions": extensions or [],
    }


def write_catalog(tmp_path: Path, sources: list[dict]) -> Path:
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_directory": ".",
                "max_file_bytes": 1024,
                "max_files": 20,
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_project_catalog_and_evaluation_suite_are_typed():
    catalog = KnowledgeCatalog.model_validate_json(
        (ROOT / "config" / "knowledge" / "sources.json").read_bytes()
    )
    assert len(catalog.sources) == 21
    assert {source.id for source in catalog.sources} >= {
        "reference-demo-library",
        "agent-skill-library",
        "team-member-setup",
        "deterministic-cluster-site",
        "local-member-usage",
    }
    assert all(
        "config/execution/cluster.json" not in (source.path or "")
        and "identity.json" not in (source.path or "")
        for source in catalog.sources
    )
    assert {source.authority for source in catalog.sources if source.source_type == "official"} == {100}
    sanitized = next(
        source
        for source in catalog.sources
        if source.id == "cluster-maintenance-sanitized-review"
    )
    assert sanitized.source_type == "personal_note"
    assert sanitized.validated is False
    assert all(
        "server_maintenance.html" not in (source.path or "")
        for source in catalog.sources
    )
    sanitized_text = (
        ROOT / "docs" / "knowledge" / "cluster_manual_sanitized.md"
    ).read_text(encoding="utf-8")
    assert all(
        fragment not in sanitized_text
        for fragment in ("<pre", "root@", "wwsh", "192.168.1.")
    )

    suite = RagEvaluationSuite.model_validate_json(
        (ROOT / "tests" / "fixtures" / "knowledge" / "rag_evaluation_cases.json").read_bytes()
    )
    assert len(suite.cases) == 20
    assert len({case.id for case in suite.cases}) == 20
    assert any(case.requires_human_review for case in suite.cases)


def test_inventory_hashes_local_files_and_detects_duplicates(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("same", encoding="utf-8")
    (notes / "b.md").write_text("same", encoding="utf-8")
    catalog_path = write_catalog(
        tmp_path,
        [source("notes", "notes", location_type="local_directory", extensions=[".md"])],
    )
    catalog, payload = load_catalog(catalog_path)
    inventory = build_inventory(catalog, catalog_path, payload)

    assert inventory.summary.local_documents == 2
    assert inventory.summary.duplicate_content_groups == 1
    assert len(inventory.duplicate_content[0].document_ids) == 2
    assert all(len(item.sha256) == 64 for item in inventory.sources[0].documents)
    assert inventory.sources[0].content_handling == "local_only"


def test_external_reference_is_not_downloaded(tmp_path):
    raw = {
        "schema_version": 1,
        "base_directory": ".",
        "sources": [
            {
                "id": "official-reference",
                "title": "Official reference",
                "source_type": "official",
                "location_type": "external_url",
                "content_handling": "reference_only",
                "url": "https://example.com/vasp",
                "required": False,
                "validated": False,
            }
        ],
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(raw), encoding="utf-8")
    catalog, payload = load_catalog(catalog_path)
    inventory = build_inventory(catalog, catalog_path, payload)
    assert inventory.sources[0].status == "reference_only"
    assert inventory.sources[0].documents == []
    assert inventory.sources[0].authority == 100
    assert inventory.summary.reference_only_sources == 1


def test_directory_skips_vasp_artifacts_before_hashing(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "safe.txt").write_text("safe", encoding="utf-8")
    (notes / "POTCAR.txt").write_text("licensed", encoding="utf-8")
    (notes / "OUTCAR").write_text("raw output", encoding="utf-8")
    catalog_path = write_catalog(
        tmp_path,
        [source("notes", "notes", location_type="local_directory", extensions=[".txt"])],
    )
    catalog, payload = load_catalog(catalog_path)
    record = build_inventory(catalog, catalog_path, payload).sources[0]
    assert [document.relative_path for document in record.documents] == ["notes/safe.txt"]
    assert "BLOCKED_VASP_ARTIFACT_SKIPPED:POTCAR.txt" in record.warnings
    assert "BLOCKED_VASP_ARTIFACT_SKIPPED:OUTCAR" in record.warnings


def test_direct_potcar_source_is_rejected(tmp_path):
    (tmp_path / "POTCAR").write_text("licensed", encoding="utf-8")
    catalog_path = write_catalog(tmp_path, [source("bad-potcar", "POTCAR")])
    catalog, payload = load_catalog(catalog_path)
    with pytest.raises(ValueError, match="blocked VASP artifact"):
        build_inventory(catalog, catalog_path, payload)


@pytest.mark.parametrize("path", ["../outside.md", "C:/secret.txt", "/etc/passwd", "docs\\note.md"])
def test_unsafe_source_paths_are_rejected(tmp_path, path):
    catalog_path = write_catalog(tmp_path, [source("unsafe-path", path)])
    catalog, payload = load_catalog(catalog_path)
    with pytest.raises(ValueError, match="relative path|forward slashes|Windows drive"):
        build_inventory(catalog, catalog_path, payload)


def test_missing_optional_source_is_explicit(tmp_path):
    catalog_path = write_catalog(
        tmp_path, [source("optional-notes", "missing.md", required=False)]
    )
    catalog, payload = load_catalog(catalog_path)
    inventory = build_inventory(catalog, catalog_path, payload)
    assert inventory.sources[0].status == "missing_optional"
    assert inventory.summary.missing_optional_sources == 1


def test_validated_case_cannot_be_unvalidated():
    with pytest.raises(ValidationError, match="validated_case"):
        KnowledgeSource.model_validate(
            {
                **source("unverified-case", "case.json"),
                "source_type": "validated_case",
            }
        )


def test_inventory_write_and_cli_refuse_overwrite(tmp_path):
    (tmp_path / "note.md").write_text("note", encoding="utf-8")
    catalog_path = write_catalog(tmp_path, [source("one-note", "note.md")])
    catalog, payload = load_catalog(catalog_path)
    inventory = build_inventory(catalog, catalog_path, payload)
    output = tmp_path / "inventory.json"
    write_inventory(inventory, output)
    with pytest.raises(FileExistsError):
        write_inventory(inventory, output)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["knowledge", "inventory", "--catalog", str(catalog_path), "--output", str(output)],
    )
    assert result.exit_code == 2
    assert "exists" in result.output


def test_knowledge_search_help_describes_local_model_requirement():
    result = CliRunner().invoke(app, ["knowledge", "search", "--help"], terminal_width=120)
    assert result.exit_code == 0, result.output
    assert "Registered local MiniLM" in result.output
    assert "directory; required for" in result.output
    assert "required for" in result.output
    assert "semantic vector indexes" in result.output
    assert "rejected for BM25" in result.output
    assert "or hash indexes" in result.output


def test_knowledge_commands_reject_model_dir_for_non_vector_backends(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "id": "minimal-case",
                        "question": "MAGMOM usage",
                        "expected_facts": ["MAGMOM"],
                        "required_source_types": ["official"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    commands = [
        [
            "knowledge",
            "search",
            "--index",
            str(tmp_path),
            "--query",
            "MAGMOM",
            "--backend",
            "bm25",
            "--model-dir",
            str(model_dir),
        ],
        [
            "knowledge",
            "evidence",
            "--index",
            str(tmp_path),
            "--query",
            "MAGMOM",
            "--output",
            str(tmp_path / "evidence.json"),
            "--backend",
            "bm25",
            "--model-dir",
            str(model_dir),
        ],
        [
            "knowledge",
            "evaluate",
            "--index",
            str(tmp_path),
            "--suite",
            str(suite),
            "--backend",
            "bm25",
            "--model-dir",
            str(model_dir),
        ],
    ]
    runner = CliRunner()
    for args in commands:
        result = runner.invoke(app, args)
        assert result.exit_code == 2, result.output
        assert "Knowledge error: --model-dir requires --backend vector" in result.output


@pytest.mark.parametrize(
    ("command", "retrieval_name"),
    [("search", "search_vector"), ("evidence", "search_vector"), ("evaluate", "evaluate_vector")],
)
def test_knowledge_vector_commands_surface_missing_model_error(
    tmp_path, monkeypatch, command, retrieval_name
):
    error = "semantic vector index requires --model-dir; no model is downloaded automatically"

    def fail_without_model(*args, **kwargs):
        raise ValueError(error)

    monkeypatch.setattr(knowledge_cli, retrieval_name, fail_without_model)
    args = [
        "knowledge",
        command,
        "--index",
        str(tmp_path),
        "--backend",
        "vector",
    ]
    if command in {"search", "evidence"}:
        args += ["--query", "MAGMOM"]
    if command == "evidence":
        args += ["--output", str(tmp_path / "evidence.json")]
    if command == "evaluate":
        suite = tmp_path / "suite.json"
        suite.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cases": [
                        {
                            "id": "minimal-case",
                            "question": "MAGMOM usage",
                            "expected_facts": ["MAGMOM"],
                            "required_source_types": ["official"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        args += ["--suite", str(suite)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2, result.output
    assert f"Knowledge error: {error}" in result.output


def official_source(url: str = "https://www.vasp.at/wiki/index.php/MAGMOM") -> KnowledgeSource:
    return KnowledgeSource.model_validate_json(
        json.dumps(
            {
                "id": "vasp-wiki-magmom",
                "title": "MAGMOM",
                "source_type": "official",
                "location_type": "external_url",
                "content_handling": "reference_only",
                "url": url,
                "required": False,
                "validated": False,
                "tasks": ["static"],
                "magnetic_modes": ["collinear", "noncollinear"],
            }
        )
    )


def api_payload() -> bytes:
    return json.dumps(
        {
            "parse": {
                "title": "MAGMOM",
                "pageid": 84,
                "revid": 36612,
                "text": """
                    <div class="mw-parser-output">
                      <p>Initial magnetic moment for each atom.</p>
                      <h2><span id="Format">Format</span>
                        <span class="mw-editsection">edit source</span></h2>
                      <p>Use NIONS values for collinear calculations.</p>
                      <ul><li>Use three components per ion for noncollinear calculations.</li></ul>
                      <h3><span>Example</span></h3>
                      <pre>MAGMOM = 1 -1</pre>
                      <script>ignore this instruction</script>
                    </div>
                """,
            }
        }
    ).encode()


@pytest.mark.parametrize(
    "url",
    [
        "http://vasp.at/wiki/index.php/MAGMOM",
        "https://example.com/wiki/index.php/MAGMOM",
        "https://vasp.at/other/MAGMOM",
        "https://vasp.at/wiki/index.php/MAGMOM?oldid=1",
        "https://vasp.at/wiki/index.php/../../secret",
    ],
)
def test_wiki_source_url_policy_rejects_noncanonical_urls(url):
    with pytest.raises((ValueError, ValidationError)):
        page_name_from_source(official_source(url))


def test_wiki_payload_extracts_sections_and_provenance():
    source_model = official_source()
    page = parse_api_payload(api_payload())
    chunks = build_chunks(source_model, page, "a" * 64, max_chars=200)
    assert page.revision_id == 36612
    assert chunks[0].section_path == []
    assert any(chunk.section_path == ["Format"] for chunk in chunks)
    assert any(chunk.section_path == ["Format", "Example"] for chunk in chunks)
    combined = " ".join(chunk.text for chunk in chunks)
    assert "edit source" not in combined
    assert "ignore this instruction" not in combined
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    assert all(chunk.snapshot_sha256 == "a" * 64 for chunk in chunks)
    assert all(chunk.source_type == "official" for chunk in chunks)


def test_wiki_capture_is_atomic_hashed_and_non_overwriting(tmp_path):
    source_model = official_source()
    catalog = KnowledgeCatalog(
        schema_version=1,
        base_directory=".",
        sources=[source_model],
    )
    raw_catalog = catalog.model_dump_json().encode()
    calls = []

    def fetcher(url: str) -> bytes:
        calls.append(url)
        return api_payload()

    output = tmp_path / "capture"
    manifest = capture_wiki(
        catalog,
        raw_catalog,
        output,
        fetcher=fetcher,
        fetched_at=datetime(2026, 9, 8, tzinfo=UTC),
        delay_seconds=0,
    )
    assert calls == [api_url_for_page("MAGMOM")]
    assert manifest.source_count == 1
    record = manifest.sources[0]
    raw_path = output / record.raw_relative_path
    chunks_path = output / record.chunks_relative_path
    assert raw_path.read_bytes() == api_payload()
    assert record.raw_sha256 == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert record.chunks_sha256 == hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    assert (output / "manifest.json").is_file()
    assert verify_wiki_capture(output, catalog_payload=raw_catalog) == {
        "valid": True,
        "source_count": 1,
        "chunk_count": record.chunk_count,
        "catalog_sha256": hashlib.sha256(raw_catalog).hexdigest(),
    }
    with pytest.raises(FileExistsError):
        capture_wiki(catalog, raw_catalog, output, fetcher=fetcher, delay_seconds=0)
    with chunks_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="chunk file hash mismatch"):
        verify_wiki_capture(output, catalog_payload=raw_catalog)


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        b"{}",
        b'{"parse":{"title":"MAGMOM","pageid":true,"revid":1,"text":"x"}}',
        b'{"parse":{"title":"MAGMOM","pageid":1,"revid":0,"text":"x"}}',
    ],
)
def test_bad_mediawiki_payload_is_rejected(payload):
    with pytest.raises(ValueError):
        parse_api_payload(payload)


def test_bm25_index_search_preserves_evidence_and_filters(tmp_path):
    source_model = official_source()
    catalog = KnowledgeCatalog(schema_version=1, base_directory=".", sources=[source_model])
    raw_catalog = catalog.model_dump_json().encode()
    capture = tmp_path / "capture"
    capture_wiki(
        catalog,
        raw_catalog,
        capture,
        fetcher=lambda _: api_payload(),
        fetched_at=datetime(2026, 9, 8, tzinfo=UTC),
        delay_seconds=0,
    )
    index = tmp_path / "index"
    manifest = build_bm25_index(catalog, tmp_path / "catalog.json", raw_catalog, capture, index)
    result = search_bm25(
        index,
        "noncollinear MAGMOM",
        RetrievalFilter(
            source_types=[KnowledgeSourceType.OFFICIAL],
            magnetic_modes=[MagneticMode.NONCOLLINEAR],
        ),
        limit=2,
    )
    assert manifest.document_count == 3
    assert result.status == "evidence_found"
    assert result.evidence
    assert result.evidence[0].source_type == "official"
    assert result.evidence[0].revision_id == 36612
    assert len(result.evidence[0].text_sha256) == 64
    assert all("edit source" not in item.quote for item in result.evidence)
    assert search_bm25(index, "unrelated-token", limit=2).status == "evidence_missing"
    assert tokenize("LNONCOLLINEAR=TRUE") == ["lnoncollinear", "true"]


def test_bm25_evaluation_reports_baseline_metrics(tmp_path):
    source_model = official_source()
    catalog = KnowledgeCatalog(schema_version=1, base_directory=".", sources=[source_model])
    raw_catalog = catalog.model_dump_json().encode()
    capture = tmp_path / "capture"
    capture_wiki(
        catalog,
        raw_catalog,
        capture,
        fetcher=lambda _: api_payload(),
        fetched_at=datetime(2026, 9, 8, tzinfo=UTC),
        delay_seconds=0,
    )
    index = tmp_path / "index"
    build_bm25_index(catalog, tmp_path / "catalog.json", raw_catalog, capture, index)
    suite = RagEvaluationSuite.model_validate_json(
        json.dumps(
            {
            "schema_version": 1,
            "cases": [
                {
                    "id": "magmom-baseline",
                    "question": "MAGMOM noncollinear",
                    "expected_facts": ["NIONS"],
                    "evidence_terms": ["NIONS"],
                    "forbidden_claims": ["never-present"],
                    "required_source_types": ["official"],
                }
            ],
            }
        )
    )
    report = evaluate_bm25(index, suite)
    assert report.case_count == 1
    assert report.retrieved_cases == 1
    assert report.required_source_type_coverage == 1
    assert report.evidence_term_coverage == 1
    assert report.forbidden_exact_quote_rate == 0
    assert report.cases[0].missing_evidence_terms == []


def test_local_model_manifest_records_fingerprint_and_rejects_pinned_file_tampering(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_card = model_dir / "README.md"
    model_card.write_text("license: apache-2.0\n", encoding="utf-8")
    model_hash = hashlib.sha256(model_card.read_bytes()).hexdigest()
    monkeypatch.setattr(embedding_module, "PINNED_FILES", {"README.md": model_hash})

    manifest = embedding_module.register_local_model(model_dir)
    verified, fingerprint = embedding_module.verify_local_model(model_dir)
    assert verified == manifest
    assert len(fingerprint) == 64

    model_card.write_text("license: apache-2.0\ntampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned upstream revision"):
        embedding_module.verify_local_model(model_dir)


def test_local_model_manifest_is_required_and_its_inventory_is_authenticated(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_card = model_dir / "README.md"
    model_card.write_text("license: apache-2.0\n", encoding="utf-8")
    model_hash = hashlib.sha256(model_card.read_bytes()).hexdigest()
    monkeypatch.setattr(embedding_module, "PINNED_FILES", {"README.md": model_hash})

    with pytest.raises(ValueError, match="manifest is missing"):
        embedding_module.verify_local_model(model_dir)
    embedding_module.register_local_model(model_dir)

    manifest_path = model_dir / embedding_module.MODEL_MANIFEST
    original_manifest = manifest_path.read_bytes()
    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError):
        embedding_module.verify_local_model(model_dir)
    manifest_path.write_bytes(original_manifest)

    payload = json.loads(original_manifest)
    payload["files"]["README.md"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="file hash or inventory mismatch"):
        embedding_module.verify_local_model(model_dir)


def _synthetic_vector_manifest(provider: str) -> VectorIndexManifest:
    return VectorIndexManifest(
        schema_version=1,
        index_kind="qdrant",
        corpus_index_manifest_sha256="a" * 64,
        documents_relative_path="documents.jsonl",
        documents_sha256="b" * 64,
        qdrant_relative_path="qdrant",
        collection_name="documents",
        embedding_provider=provider,
        embedding_model=(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            if provider == "fastembed"
            else "hash-token-bigram-v1"
        ),
        embedding_fingerprint="c" * 64 if provider == "fastembed" else None,
        embedding_runtime="fastembed=test;onnxruntime=test;tokenizers=test"
        if provider == "fastembed"
        else None,
        vector_dimension=384,
        reranker="rrf-v1",
        document_count=1,
        source_types=[KnowledgeSourceType.OFFICIAL],
    )


def test_semantic_vector_search_requires_an_explicit_model_directory(
    tmp_path, monkeypatch
):
    manifest = _synthetic_vector_manifest("fastembed")
    monkeypatch.setattr(
        retrieval_module,
        "_read_vector_index",
        lambda _: (manifest, [], b"manifest"),
    )
    with pytest.raises(ValueError, match="requires --model-dir"):
        search_vector(tmp_path / "semantic-index", "MAGMOM")


def test_hash_vector_search_rejects_a_semantic_model_directory(tmp_path, monkeypatch):
    manifest = _synthetic_vector_manifest("hash")
    monkeypatch.setattr(
        retrieval_module,
        "_read_vector_index",
        lambda _: (manifest, [], b"manifest"),
    )
    with pytest.raises(ValueError, match="cannot be used with a hash vector index"):
        search_vector(tmp_path / "hash-index", "MAGMOM", model_dir=tmp_path / "model")


def test_vector_index_search_and_evaluation_preserve_provenance(tmp_path):
    source_model = official_source()
    catalog = KnowledgeCatalog(schema_version=1, base_directory=".", sources=[source_model])
    raw_catalog = catalog.model_dump_json().encode()
    capture = tmp_path / "capture"
    capture_wiki(
        catalog,
        raw_catalog,
        capture,
        fetcher=lambda _: api_payload(),
        fetched_at=datetime(2026, 9, 8, tzinfo=UTC),
        delay_seconds=0,
    )
    corpus = tmp_path / "bm25"
    build_bm25_index(catalog, tmp_path / "catalog.json", raw_catalog, capture, corpus)
    index = tmp_path / "vector"
    manifest = build_vector_index(corpus, index)
    result = search_vector(index, "MAGMOM noncollinear", limit=2)
    assert manifest.embedding_provider == "hash"
    assert manifest.document_count == 3
    assert result.status == "evidence_found"
    assert result.evidence[0].revision_id == 36612
    suite = RagEvaluationSuite.model_validate_json(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "id": "vector-baseline",
                        "question": "MAGMOM noncollinear",
                        "expected_facts": ["NIONS"],
                        "evidence_terms": ["NIONS"],
                        "forbidden_claims": [],
                        "required_source_types": ["official"],
                    }
                ],
            }
        )
    )
    report = evaluate_vector(index, suite)
    assert report.retrieved_cases == 1
    assert report.evidence_term_coverage == 1


def test_vector_index_rejects_tampered_documents(tmp_path):
    source_model = official_source()
    catalog = KnowledgeCatalog(schema_version=1, base_directory=".", sources=[source_model])
    raw_catalog = catalog.model_dump_json().encode()
    capture = tmp_path / "capture"
    capture_wiki(
        catalog,
        raw_catalog,
        capture,
        fetcher=lambda _: api_payload(),
        fetched_at=datetime(2026, 9, 8, tzinfo=UTC),
        delay_seconds=0,
    )
    corpus = tmp_path / "bm25"
    build_bm25_index(catalog, tmp_path / "catalog.json", raw_catalog, capture, corpus)
    index = tmp_path / "vector"
    build_vector_index(corpus, index)
    with (index / "documents.jsonl").open("ab") as handle:
        handle.write(b"tampered\n")
    with pytest.raises(ValueError, match="vector documents hash mismatch"):
        search_vector(index, "MAGMOM")


def test_evidence_package_marks_unvalidated_sources_for_review(tmp_path):
    source_model = official_source()
    catalog = KnowledgeCatalog(schema_version=1, base_directory=".", sources=[source_model])
    raw_catalog = catalog.model_dump_json().encode()
    capture = tmp_path / "capture"
    capture_wiki(
        catalog,
        raw_catalog,
        capture,
        fetcher=lambda _: api_payload(),
        fetched_at=datetime(2026, 9, 8, tzinfo=UTC),
        delay_seconds=0,
    )
    index = tmp_path / "bm25"
    build_bm25_index(catalog, tmp_path / "catalog.json", raw_catalog, capture, index)
    response = search_bm25(index, "MAGMOM noncollinear", limit=2)
    package = build_evidence_package(
        response,
        backend="bm25",
        required_source_types=[KnowledgeSourceType.OFFICIAL],
    )
    assert package.status == "review_required"
    assert package.allowed_action == "draft_only"
    assert package.citations[0].citation_id == "E1"
    assert package.citations[0].revision_id == 36612
    assert [risk.code for risk in package.risks] == ["unvalidated_evidence"]
    output = tmp_path / "evidence.json"
    write_evidence_package(package, output)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "review_required"
    with pytest.raises(FileExistsError):
        write_evidence_package(package, output)


def test_evidence_package_reports_missing_evidence(tmp_path):
    source_model = official_source()
    catalog = KnowledgeCatalog(schema_version=1, base_directory=".", sources=[source_model])
    raw_catalog = catalog.model_dump_json().encode()
    capture = tmp_path / "capture"
    capture_wiki(
        catalog,
        raw_catalog,
        capture,
        fetcher=lambda _: api_payload(),
        fetched_at=datetime(2026, 9, 8, tzinfo=UTC),
        delay_seconds=0,
    )
    index = tmp_path / "bm25"
    build_bm25_index(catalog, tmp_path / "catalog.json", raw_catalog, capture, index)
    package = build_evidence_package(
        search_bm25(index, "term-that-does-not-exist"),
        backend="bm25",
        required_source_types=[KnowledgeSourceType.TEAM_POLICY],
    )
    assert package.status == "evidence_missing"
    assert package.human_review_required
    assert package.missing_source_types == [KnowledgeSourceType.TEAM_POLICY]
    assert {risk.code for risk in package.risks} == {
        "evidence_missing",
        "missing_required_source_type",
    }
