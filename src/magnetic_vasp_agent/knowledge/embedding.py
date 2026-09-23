"""Pinned, local-only multilingual embeddings; model preparation is separate."""
from __future__ import annotations

import hashlib
import json
import math
import stat
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from pydantic import Field

from magnetic_vasp_agent.knowledge.models import StrictModel

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REPOSITORY = "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
MODEL_REVISION = "faf4aa4225822f3bc6376869cb1164e8e3feedd0"
MODEL_DIMENSION = 384
MODEL_MANIFEST = "magvasp-model.json"
PINNED_FILES = {
    "README.md": "c96c17d91a9e83ec840da5a517277f5164315bd765e21da4216a568008c4f74e",
    "config.json": "c8ec081fdad2df991bf5abbf18418fec7a5cdaa421f60ffb060a30040b8c376f",
    "model_optimized.onnx": "634d0f66c29dc934c8fa72b8a4fe91dd4d420a22f1d82a241058d4316e659a99",
    "ort_config.json": "99881c45e073696289224931dd48694398bc6bcd1fe7cb7018bca1e0cc00e1fc",
    "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    "tokenizer.json": "fa685fc160bbdbab64058d4fc91b60e62d207e8dc60b9af5c002c5ab946ded00",
    "tokenizer_config.json": "0666eebf692422757e1dddf3c9fb1ded73ba3dc726c5828671fc89e45bf3609f",
    "unigram.json": "da145b5e7700ae40f16691ec32a0b1fdc1ee3298db22a31ea55f57a966c4a65d",
}


class ModelFile(StrictModel):
    size_bytes: int = Field(gt=0, le=512 * 1024 * 1024)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class LocalModelManifest(StrictModel):
    schema_version: Literal[1]
    model: Literal[MODEL_NAME]
    repository: Literal[MODEL_REPOSITORY]
    revision: Literal[MODEL_REVISION]
    license: Literal["apache-2.0"]
    dimension: Literal[384]
    files: dict[str, ModelFile]


def _model_files(folder: Path) -> dict[str, ModelFile]:
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("local model must be a real directory")
    files = {}
    total = 0
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    for path in sorted(folder.rglob("*")):
        if path.is_symlink() or (
            reparse_point
            and getattr(path.lstat(), "st_file_attributes", 0) & reparse_point
        ):
            raise ValueError("local model must not contain links")
        if path.is_dir():
            continue
        relative = path.relative_to(folder).as_posix()
        if relative == MODEL_MANIFEST:
            continue
        if not path.is_file() or len(files) >= 100:
            raise ValueError("invalid local model files")
        size = path.stat().st_size
        total += size
        if not 0 < size <= 512 * 1024 * 1024 or total > 1024 * 1024 * 1024:
            raise ValueError("local model exceeds size limits")
        with path.open("rb") as handle:
            sha = hashlib.file_digest(handle, "sha256").hexdigest()
        files[relative] = ModelFile(size_bytes=size, sha256=sha)
    if not PINNED_FILES.keys() <= files.keys():
        raise ValueError("local model files are incomplete")
    if any(files[name].sha256 != sha for name, sha in PINNED_FILES.items()):
        raise ValueError("model files differ from the pinned upstream revision")
    return files


def register_local_model(folder: Path) -> LocalModelManifest:
    """Record a pre-staged pinned model; no download or model code is executed.

    Loaded files must match the reviewed fixed revision's SHA256 values.
    """
    output = folder / MODEL_MANIFEST
    if output.exists() or output.is_symlink():
        raise FileExistsError("local model manifest exists")
    files = _model_files(folder)
    card = (folder / "README.md").read_text(encoding="utf-8")
    if "license: apache-2.0" not in card.lower():
        raise ValueError("model card must declare apache-2.0")
    manifest = LocalModelManifest(
        schema_version=1, model=MODEL_NAME, repository=MODEL_REPOSITORY,
        revision=MODEL_REVISION, license="apache-2.0", dimension=384, files=files,
    )
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def verify_local_model(folder: Path) -> tuple[LocalModelManifest, str]:
    path = folder / MODEL_MANIFEST
    if folder.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("local model manifest is missing or linked; register the model first")
    if path.stat().st_size > 128 * 1024:
        raise ValueError("local model manifest is too large")
    manifest = LocalModelManifest.model_validate_json(path.read_bytes())
    if _model_files(folder) != manifest.files:
        raise ValueError("local model file hash or inventory mismatch")
    canonical = json.dumps(manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return manifest, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LocalEmbedder:
    def __init__(self, folder: Path, *, expected_fingerprint: str | None = None,
                 expected_runtime: str | None = None):
        manifest, self.fingerprint = verify_local_model(folder)
        if expected_fingerprint is not None and self.fingerprint != expected_fingerprint:
            raise ValueError("local model fingerprint does not match vector index")
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("local embeddings require: uv sync --extra rag --extra semantic --frozen") from exc
        self.runtime = f"fastembed={version('fastembed')};onnxruntime={version('onnxruntime')};tokenizers={version('tokenizers')}"
        if expected_runtime is not None and self.runtime != expected_runtime:
            raise ValueError("embedding runtime differs from vector index; rebuild the index")
        self.dimension = manifest.dimension
        self.model = TextEmbedding(
            model_name=manifest.model, specific_model_path=str(folder.resolve()),
            local_files_only=True, providers=["CPUExecutionProvider"], cuda=False,
            threads=2,
        )

    def _vectors(self, raw, count: int) -> list[list[float]]:
        vectors = []
        for item in raw:
            values = [float(value) for value in item]
            if len(values) != self.dimension or not all(math.isfinite(v) for v in values):
                raise ValueError("embedding has invalid dimensions or non-finite values")
            norm = math.sqrt(sum(v * v for v in values))
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("embedding has zero or non-finite norm")
            vectors.append([v / norm for v in values])
        if len(vectors) != count:
            raise ValueError("embedding count mismatch")
        return vectors

    def documents(self, texts: list[str]) -> list[list[float]]:
        return self._vectors(self.model.passage_embed(texts, batch_size=16), len(texts))

    def query(self, text: str) -> list[float]:
        return self._vectors(self.model.query_embed(text), 1)[0]
