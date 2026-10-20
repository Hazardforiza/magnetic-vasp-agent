"""User-visible knowledge intake, publication, and retrieval boundaries."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from magnetic_vasp_agent.web import create_app


def _client(root: Path) -> tuple[TestClient, dict[str, str]]:
    app = create_app(
        root=root,
        registry=root / "missing-registry.json",
        policy=root / "missing-policy.json",
        llm_config=root / "missing-llm.json",
        execution_profile=root / "missing-cluster.json",
    )
    client = TestClient(app, base_url="http://localhost")
    response = client.get("/api/status")
    assert response.status_code == 200, response.text
    return client, {"X-CSRF-Token": response.json()["csrf_token"]}


@pytest.fixture
def knowledge_client(tmp_path: Path) -> tuple[TestClient, dict[str, str]]:
    return _client(tmp_path / "web")


def _import(
    client: TestClient,
    headers: dict[str, str],
    text: str,
    *,
    scope: str = "local",
    filename: str = "research.md",
    title: str = "Research notes",
) -> dict:
    response = client.post(
        "/api/knowledge/import",
        headers=headers,
        data={"scope": scope, "title": title, "tags": "FeO,Wannier"},
        files={"file": (filename, text.encode("utf-8"), "text/plain")},
    )
    assert response.status_code in (200, 201), response.text
    return response.json()


def _index(client: TestClient, headers: dict[str, str], item: dict) -> dict:
    response = client.post(
        f"/api/knowledge/{item['scope']}/{item['id']}/index",
        headers=headers,
        json={},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _search(client: TestClient, headers: dict[str, str], query: str, scope: str = "all") -> dict:
    response = client.post(
        "/api/knowledge/search",
        headers=headers,
        json={"query": query, "scope": scope, "limit": 10},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _configure_shared(client: TestClient, headers: dict[str, str], path: Path) -> None:
    response = client.put("/api/knowledge/shared", headers=headers, json={"path": str(path)})
    assert response.status_code == 200, response.text


def _pdf(pages: list[str]) -> bytes:
    """Build a small text PDF with real page objects and an accurate xref table."""
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    page_numbers = []
    for text in pages:
        page_number = len(objects) + 1
        page_numbers.append(page_number)
        stream_number = page_number + 1
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {stream_number} 0 R >>"
            ).encode("ascii")
        )
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    kids = " ".join(f"{number} 0 R" for number in page_numbers)
    objects[1] = f"<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>".encode("ascii")
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(payload))
        payload.extend(f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(payload)


def test_draft_edit_confirm_retrieve_and_restart(tmp_path: Path) -> None:
    root = tmp_path / "web"
    client, headers = _client(root)
    draft = _import(client, headers, "ObsoleteAlphaToken notes before review.")
    assert draft["scope"] == "local"
    assert draft["kind"] == "literature"
    assert draft["status"] == "draft"
    assert draft["validation_status"] == "unverified"
    assert _search(client, headers, "ObsoleteAlphaToken", "local")["evidence"] == []

    edited = client.patch(
        f"/api/knowledge/local/{draft['id']}",
        headers=headers,
        json={
            "title": "Reviewed Wannier disentanglement notes",
            "doi": "10.1234/example",
            "tags": ["Wannier", "FeO"],
            "text": "ConfirmedBetaToken: compare interpolated and DFT bands before accepting a Wannier model.",
        },
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["status"] == "draft"
    assert "ConfirmedBetaToken" in edited.json()["text"]
    assert _search(client, headers, "ConfirmedBetaToken", "local")["evidence"] == []

    indexed = _index(client, headers, draft)
    assert indexed["status"] == "indexed"
    assert indexed["validation_status"] == "unverified"
    evidence = _search(client, headers, "ConfirmedBetaToken", "local")["evidence"]
    assert evidence
    assert {entry["item_id"] for entry in evidence} == {draft["id"]}
    assert all(entry["source_validated"] is False for entry in evidence)
    assert all(entry["scope"] == "local" for entry in evidence)
    assert "ConfirmedBetaToken" in evidence[0]["quote"]
    assert _search(client, headers, "ObsoleteAlphaToken", "local")["evidence"] == []

    restarted, restarted_headers = _client(root)
    assert restarted.get(f"/api/knowledge/local/{draft['id']}").json()["status"] == "indexed"
    assert _search(restarted, restarted_headers, "ConfirmedBetaToken", "local")["evidence"]

    # Editing confirmed evidence removes the stale passage until another explicit confirmation.
    changed = restarted.patch(
        f"/api/knowledge/local/{draft['id']}", headers=restarted_headers,
        json={"text": "ReplacementGammaToken notes need a fresh review."},
    )
    assert changed.status_code == 200, changed.text
    assert changed.json()["status"] == "draft"
    assert _search(restarted, restarted_headers, "ConfirmedBetaToken", "local")["evidence"] == []
    assert _search(restarted, restarted_headers, "ReplacementGammaToken", "local")["evidence"] == []


def test_shared_requires_configuration_and_explicit_publication(tmp_path: Path) -> None:
    client, headers = _client(tmp_path / "web")
    local = _import(client, headers, "PrivateLocalToken must stay on this computer.")
    unpublished = client.post(f"/api/knowledge/local/{local['id']}/publish", headers=headers, json={})
    assert unpublished.status_code in (409, 422, 503), unpublished.text
    indexed = _index(client, headers, local)
    assert _search(client, headers, "PrivateLocalToken", "shared")["evidence"] == []

    shared = tmp_path / "team-drive"
    shared.mkdir()
    _configure_shared(client, headers, shared)
    assert _search(client, headers, "PrivateLocalToken", "shared")["evidence"] == []
    publication = client.post(f"/api/knowledge/local/{local['id']}/publish", headers=headers, json={})
    assert publication.status_code in (200, 201), publication.text
    published = publication.json()
    assert published["scope"] == "shared"
    assert published["validation_status"] == "unverified"
    assert published["status"] == "indexed"
    shared_evidence = _search(client, headers, "PrivateLocalToken", "shared")["evidence"]
    assert shared_evidence and all(entry["scope"] == "shared" for entry in shared_evidence)
    assert client.get(f"/api/knowledge/local/{indexed['id']}").status_code == 200

    # Disconnecting a team drive must leave the local library usable.
    disconnected = client.put("/api/knowledge/shared", headers=headers, json={"path": None})
    assert disconnected.status_code == 200, disconnected.text
    assert _search(client, headers, "PrivateLocalToken", "local")["evidence"]
    assert _search(client, headers, "PrivateLocalToken", "shared")["evidence"] == []


def test_two_instances_share_confirmed_documents_without_losing_concurrent_updates(tmp_path: Path) -> None:
    first, first_headers = _client(tmp_path / "first")
    second, second_headers = _client(tmp_path / "second")
    shared = tmp_path / "team-drive"
    shared.mkdir()
    for client, headers in ((first, first_headers), (second, second_headers)):
        _configure_shared(client, headers, shared)

    first_item = _import(first, first_headers, "JointCorpusToken AliceCaseToken convergence notes.", scope="shared")
    second_item = _import(second, second_headers, "JointCorpusToken BobCaseToken interpolation notes.", scope="shared")
    assert _search(first, first_headers, "JointCorpusToken", "shared")["evidence"] == []

    def confirm(client: TestClient, headers: dict[str, str], item: dict):
        return client.post(f"/api/knowledge/shared/{item['id']}/index", headers=headers, json={})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(confirm, first, first_headers, first_item),
            pool.submit(confirm, second, second_headers, second_item),
        ]
        responses = [future.result() for future in futures]
    for response, client, headers, item in zip(
        responses, (first, second), (first_headers, second_headers), (first_item, second_item), strict=True,
    ):
        if response.status_code == 409:
            assert _index(client, headers, item)["status"] == "indexed"
        else:
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "indexed"

    for client, headers in ((first, first_headers), (second, second_headers)):
        evidence = _search(client, headers, "JointCorpusToken", "shared")["evidence"]
        assert {entry["item_id"] for entry in evidence} == {first_item["id"], second_item["id"]}
        assert _search(client, headers, "JointCorpusToken", "local")["evidence"] == []

    private = _index(first, first_headers, _import(first, first_headers, "ComputerPrivateToken private notes."))
    assert _search(second, second_headers, "ComputerPrivateToken")["evidence"] == []
    assert second.get(f"/api/knowledge/local/{private['id']}").status_code == 404
    restarted, restarted_headers = _client(tmp_path / "second")
    assert len(_search(restarted, restarted_headers, "JointCorpusToken", "shared")["evidence"]) == 2


def test_pdf_keeps_real_page_citations_and_rejects_unextractable_files(
    knowledge_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, headers = knowledge_client
    response = client.post(
        "/api/knowledge/import", headers=headers,
        data={"scope": "local", "title": "Two-page methods paper"},
        files={"file": ("methods.pdf", _pdf(["FirstPageToken background.", "SecondPageToken Wannier checks."]), "application/pdf")},
    )
    assert response.status_code in (200, 201), response.text
    draft = response.json()
    assert draft["page_count"] == 2
    assert [page["page"] for page in draft["pages"]] == [1, 2]
    _index(client, headers, draft)
    evidence = _search(client, headers, "SecondPageToken", "local")["evidence"]
    assert evidence and evidence[0]["page"] == 2
    assert "SecondPageToken" in evidence[0]["quote"]

    for filename, payload in (("broken.pdf", b"%PDF-1.4\ninvalid"), ("scan.pdf", _pdf([""]))):
        failed = client.post(
            "/api/knowledge/import", headers=headers, data={"scope": "local"},
            files={"file": (filename, payload, "application/pdf")},
        )
        assert failed.status_code in (400, 422), failed.text
        assert failed.json().get("detail")
    items = client.get("/api/knowledge").json()["items"]
    assert {item["id"] for item in items} == {draft["id"]}


def test_case_confirmation_does_not_assert_scientific_validation(
    knowledge_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, headers = knowledge_client
    response = client.post(
        "/api/knowledge/cases", headers=headers,
        json={
            "scope": "local", "title": "Unconverged FeO trial",
            "text": "CaseFailureToken: electronic SCF did not converge; this result must not be used as a validated reference.",
            "tags": ["FeO", "failed"],
        },
    )
    assert response.status_code in (200, 201), response.text
    case = response.json()
    assert case["kind"] == "case" and case["status"] == "draft"
    assert case["validation_status"] == "unverified"
    _index(client, headers, case)
    evidence = _search(client, headers, "CaseFailureToken", "local")["evidence"]
    assert evidence and all(entry["source_validated"] is False for entry in evidence)
    forged = client.patch(
        f"/api/knowledge/local/{case['id']}", headers=headers,
        json={"validation_status": "validated", "source_validated": True},
    )
    assert forged.status_code == 422, forged.text


def test_result_case_keeps_source_evidence_without_copying_private_paths(tmp_path: Path) -> None:
    root = tmp_path / "web"
    client, headers = _client(root)
    result_id = "a" * 32
    result_root = root / "results" / result_id
    result_root.mkdir()
    source_hash = "b" * 64
    (result_root / "summary.json").write_text(
        json.dumps({
            "directory": "PRIVATE_DIRECTORY_MUST_NOT_BE_PUBLISHED",
            "warnings": ["PRIVATE_DIAGNOSTIC_PATH_MUST_NOT_BE_PUBLISHED"],
            "final_energy": -12.5,
            "outcar": {
                "source_sha256": source_hash, "vasp_version": "5.4.4",
                "electronic_converged": False,
                "path": "PRIVATE_OUTCAR_PATH_MUST_NOT_BE_PUBLISHED",
                "raw_content": "PRIVATE_RAW_OUTPUT_MUST_NOT_BE_PUBLISHED",
            },
        }),
        encoding="utf-8",
    )
    response = client.post(
        "/api/knowledge/cases", headers=headers,
        json={
            "scope": "local", "title": "Imported convergence failure",
            "text": "ResultCaseToken: do not accept these unconverged results.",
            "tags": ["failed"], "result_id": result_id,
        },
    )
    assert response.status_code in (200, 201), response.text
    case = response.json()
    assert source_hash in str(case["provenance"])
    assert "5.4.4" in str(case["provenance"])
    assert "PRIVATE_" not in response.text
    assert case["validation_status"] == "unverified"
    _index(client, headers, case)
    assert "PRIVATE_" not in str(_search(client, headers, "ResultCaseToken", "local"))
    missing = client.post(
        "/api/knowledge/cases", headers=headers,
        json={"scope": "local", "title": "Missing result", "text": "notes", "result_id": "c" * 32},
    )
    assert missing.status_code == 404, missing.text
    assert len(client.get("/api/knowledge").json()["items"]) == 1


def test_duplicate_upload_and_untrusted_paths_do_not_create_extra_documents(
    knowledge_client: tuple[TestClient, dict[str, str]], tmp_path: Path,
) -> None:
    client, headers = knowledge_client
    item = _import(client, headers, "DuplicateSourceToken reproducible method.")
    duplicate = _import(client, headers, "DuplicateSourceToken reproducible method.", filename="renamed.md")
    assert duplicate["id"] == item["id"]
    assert duplicate["duplicate"] is True
    assert len(client.get("/api/knowledge").json()["items"]) == 1
    for filename in ("../../escape.md", "POTCAR", "OUTCAR", "vasprun.xml"):
        rejected = client.post(
            "/api/knowledge/import", headers=headers, data={"scope": "local"},
            files={"file": (filename, b"private raw calculation data", "text/plain")},
        )
        assert rejected.status_code in (400, 422), (filename, rejected.text)
    assert client.get("/api/knowledge/local/not-a-library-id").status_code == 404
    relative = client.put("/api/knowledge/shared", headers=headers, json={"path": "../team-drive"})
    assert relative.status_code == 422, relative.text
    assert len(client.get("/api/knowledge").json()["items"]) == 1


def test_knowledge_writes_require_local_csrf_token(
    knowledge_client: tuple[TestClient, dict[str, str]], tmp_path: Path,
) -> None:
    client, headers = knowledge_client
    assert client.post(
        "/api/knowledge/import", data={"scope": "local"},
        files={"file": ("notes.txt", b"notes", "text/plain")},
    ).status_code == 403
    assert client.put("/api/knowledge/shared", json={"path": str(tmp_path)}).status_code == 403
    assert client.post(
        "/api/knowledge/cases", json={"scope": "local", "title": "trial", "text": "notes"},
    ).status_code == 403
    foreign = client.post(
        "/api/knowledge/search", headers={**headers, "Origin": "https://foreign.example"},
        json={"query": "notes", "scope": "all", "limit": 5},
    )
    assert foreign.status_code == 403

