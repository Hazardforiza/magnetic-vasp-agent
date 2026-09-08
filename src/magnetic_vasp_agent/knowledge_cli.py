# ruff: noqa: B008 - Typer requires option declarations in function signatures.
from __future__ import annotations

import json
from pathlib import Path

import typer

from magnetic_vasp_agent.knowledge import (
    build_inventory,
    capture_wiki,
    load_catalog,
    verify_wiki_capture,
    write_inventory,
)
from magnetic_vasp_agent.knowledge.models import (
    CalculationTask,
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
    write_evidence_package,
)
from magnetic_vasp_agent.knowledge.retrieval_models import RetrievalFilter
from magnetic_vasp_agent.knowledge.embedding import register_local_model

_BACKEND_HELP = "Retrieval backend: bm25 or vector."
_MODEL_DIR_HELP = (
    "Registered local MiniLM directory; required for semantic vector indexes "
    "and rejected for BM25 or hash indexes."
)

knowledge_app = typer.Typer(
    no_args_is_help=True,
    help="Inventory, capture, and search governed VASP knowledge with traceable evidence.",
)


def _error(exc: Exception) -> None:
    typer.echo(f"Knowledge error: {exc}", err=True)
    raise typer.Exit(2)


@knowledge_app.command("inventory")
def inventory(
    catalog_path: Path = typer.Option(
        ..., "--catalog", exists=True, dir_okay=False
    ),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Hash and classify declared sources; never parse, embed, or upload content."""
    try:
        catalog, payload = load_catalog(catalog_path)
        report = build_inventory(catalog, catalog_path, payload)
        if output is not None:
            write_inventory(report, output)
        typer.echo(report.model_dump_json(indent=2))
    except (ValueError, OSError) as exc:
        _error(exc)


@knowledge_app.command("verify-wiki")
def verify_vasp_wiki(
    capture: Path = typer.Option(..., "--capture", exists=True, file_okay=False),
    catalog_path: Path = typer.Option(
        ..., "--catalog", exists=True, dir_okay=False
    ),
) -> None:
    """Verify capture scope, source revisions, chunk provenance, and every hash."""
    try:
        _, payload = load_catalog(catalog_path)
        report = verify_wiki_capture(capture, catalog_payload=payload)
        typer.echo(json.dumps(report, indent=2) + "\n", nl=False)
    except (ValueError, OSError) as exc:
        _error(exc)


@knowledge_app.command("capture-wiki")
def capture_vasp_wiki(
    catalog_path: Path = typer.Option(
        ..., "--catalog", exists=True, dir_okay=False
    ),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Capture only allowlisted official VASP Wiki sources and extract cited chunks."""
    try:
        catalog, payload = load_catalog(catalog_path)
        report = capture_wiki(catalog, payload, output)
        typer.echo(report.model_dump_json(indent=2))
    except (ValueError, OSError) as exc:
        _error(exc)


@knowledge_app.command("build-bm25")
def build_bm25(
    catalog_path: Path = typer.Option(
        ..., "--catalog", exists=True, dir_okay=False
    ),
    capture: Path = typer.Option(..., "--capture", exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Build a local BM25 index from verified Wiki chunks and approved Markdown."""
    try:
        catalog, payload = load_catalog(catalog_path)
        report = build_bm25_index(catalog, catalog_path, payload, capture, output)
        typer.echo(report.model_dump_json(indent=2))
    except (ValueError, OSError) as exc:
        _error(exc)


@knowledge_app.command("build-vector")
def build_vector(
    corpus: Path = typer.Option(
        ..., "--corpus", exists=True, file_okay=False,
        help="Verified BM25 corpus used as the vector index source.",
    ),
    output: Path = typer.Option(..., "--output"),
    model_dir: Path | None = typer.Option(
        None, "--model-dir", exists=True, file_okay=False, help=_MODEL_DIR_HELP
    ),
) -> None:
    """Build Qdrant with hash vectors or an explicitly supplied offline model."""
    try:
        report = build_vector_index(corpus, output, model_dir=model_dir)
        typer.echo(report.model_dump_json(indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        _error(exc)


@knowledge_app.command("register-model")
def register_model(
    model_dir: Path = typer.Option(..., "--model-dir", exists=True, file_okay=False),
) -> None:
    """Hash pre-staged files from the documented fixed MiniLM revision; no download."""
    try:
        typer.echo(register_local_model(model_dir).model_dump_json(indent=2))
    except (ValueError, OSError) as exc:
        _error(exc)


def _parse_enum_list(value: str, enum_type: type) -> list:
    if not value.strip():
        return []
    try:
        return [enum_type(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid filter value: {value}") from exc


@knowledge_app.command("search")
def search_knowledge(
    index: Path = typer.Option(..., "--index", exists=True, file_okay=False),
    query: str = typer.Option(..., "--query"),
    limit: int = typer.Option(5, "--limit", min=1, max=100),
    source_types: str = typer.Option("", "--source-types", help="Comma-separated source types."),
    tasks: str = typer.Option("", "--tasks", help="Comma-separated calculation tasks."),
    magnetic_modes: str = typer.Option("", "--magnetic-modes", help="Comma-separated magnetic modes."),
    validated_only: bool = typer.Option(False, "--validated-only"),
    backend: str = typer.Option("bm25", "--backend", help=_BACKEND_HELP),
    model_dir: Path | None = typer.Option(
        None, "--model-dir", exists=True, file_okay=False, help=_MODEL_DIR_HELP
    ),
) -> None:
    """Search a local retrieval index and return traceable evidence."""
    try:
        filters = RetrievalFilter(
            source_types=_parse_enum_list(source_types, KnowledgeSourceType),
            tasks=_parse_enum_list(tasks, CalculationTask),
            magnetic_modes=_parse_enum_list(magnetic_modes, MagneticMode),
            validated_only=validated_only,
        )
        if backend == "bm25":
            if model_dir is not None:
                raise ValueError("--model-dir requires --backend vector")
            response = search_bm25(index, query, filters, limit)
        elif backend == "vector":
            response = search_vector(index, query, filters, limit, model_dir=model_dir)
        else:
            raise ValueError("--backend must be bm25 or vector")
        typer.echo(response.model_dump_json(indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        _error(exc)


@knowledge_app.command("evidence")
def create_evidence_package(
    index: Path = typer.Option(..., "--index", exists=True, file_okay=False),
    query: str = typer.Option(..., "--query"),
    output: Path = typer.Option(..., "--output"),
    limit: int = typer.Option(5, "--limit", min=1, max=100),
    backend: str = typer.Option("bm25", "--backend", help=_BACKEND_HELP),
    required_source_types: str = typer.Option(
        "", "--required-source-types", help="Comma-separated required source types."
    ),
    tasks: str = typer.Option("", "--tasks", help="Comma-separated calculation tasks."),
    magnetic_modes: str = typer.Option(
        "", "--magnetic-modes", help="Comma-separated magnetic modes."
    ),
    validated_only: bool = typer.Option(False, "--validated-only"),
    human_review: bool = typer.Option(False, "--human-review"),
    model_dir: Path | None = typer.Option(
        None, "--model-dir", exists=True, file_okay=False, help=_MODEL_DIR_HELP
    ),
) -> None:
    """Create an immutable, cited evidence package for a future answer draft."""
    try:
        required = _parse_enum_list(required_source_types, KnowledgeSourceType)
        filters = RetrievalFilter(
            source_types=required,
            tasks=_parse_enum_list(tasks, CalculationTask),
            magnetic_modes=_parse_enum_list(magnetic_modes, MagneticMode),
            validated_only=validated_only,
        )
        if backend == "bm25":
            if model_dir is not None:
                raise ValueError("--model-dir requires --backend vector")
            response = search_bm25(index, query, filters, limit)
        elif backend == "vector":
            response = search_vector(index, query, filters, limit, model_dir=model_dir)
        else:
            raise ValueError("--backend must be bm25 or vector")
        package = build_evidence_package(
            response,
            backend=backend,
            required_source_types=required,
            human_review_requested=human_review,
        )
        write_evidence_package(package, output)
        typer.echo(package.model_dump_json(indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        _error(exc)


@knowledge_app.command("evaluate")
def evaluate_knowledge(
    index: Path = typer.Option(..., "--index", exists=True, file_okay=False),
    suite: Path = typer.Option(..., "--suite", exists=True, dir_okay=False),
    limit: int = typer.Option(5, "--limit", min=1, max=100),
    backend: str = typer.Option("bm25", "--backend", help=_BACKEND_HELP),
    model_dir: Path | None = typer.Option(
        None, "--model-dir", exists=True, file_okay=False, help=_MODEL_DIR_HELP
    ),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Run offline retrieval coverage evaluation; this is not answer validation."""
    try:
        if output is not None and (output.exists() or output.is_symlink()):
            raise FileExistsError("evaluation report exists; choose a new output path")
        evaluation = RagEvaluationSuite.model_validate_json(suite.read_bytes())
        if backend == "bm25":
            if model_dir is not None:
                raise ValueError("--model-dir requires --backend vector")
            report = evaluate_bm25(index, evaluation, limit=limit)
        elif backend == "vector":
            report = evaluate_vector(index, evaluation, limit=limit, model_dir=model_dir)
        else:
            raise ValueError("--backend must be bm25 or vector")
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(report.model_dump_json(indent=2) + "\n")
        typer.echo(report.model_dump_json(indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        _error(exc)
