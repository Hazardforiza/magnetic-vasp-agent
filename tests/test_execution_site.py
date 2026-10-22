from pathlib import Path

from magnetic_vasp_agent.execution_site import (
    SiteTemplate,
    UserIdentity,
    compose_cluster_profile,
    compose_from_files,
    load_site_template,
)
from magnetic_vasp_agent.cli import app
from typer.testing import CliRunner

ROOT = Path(__file__).parents[1]
SITE = ROOT / "config/execution/site.team.json"


def test_shared_site_has_office_and_tunnel_but_no_member_account() -> None:
    site = load_site_template(SITE)
    assert site.site_id == "team-example"
    assert site.gateway_hostname == "login01"
    assert "node01" in site.allowed_nodes
    assert {item.id for item in site.connections} == {"office", "tunnel"}
    for path in (
        SITE,
        ROOT / "docs/knowledge/team_cluster_policy.md",
        ROOT / "docs/knowledge/member_setup.md",
    ):
        text = path.read_text(encoding="utf-8")
        assert "cluster_user@" not in text
        assert "/home/cluster_user" not in text


def test_compose_office_uses_workstation_gateway() -> None:
    site = load_site_template(SITE)
    profile = compose_cluster_profile(
        site, UserIdentity(user="alice", connection="office")
    )
    assert profile.user == "alice"
    assert profile.gateway == "gateway.example.org"
    assert profile.port == 22
    assert profile.remote_root == "/home/alice/magvasp/runs"
    assert profile.execution_enabled is False


def test_compose_tunnel_uses_localhost() -> None:
    site = load_site_template(SITE)
    profile = compose_cluster_profile(
        site,
        UserIdentity(user="alice", connection="tunnel", execution_enabled=True),
    )
    assert profile.gateway == "127.0.0.1"
    assert profile.port == 2222
    assert profile.remote_root == "/home/alice/magvasp/runs"
    assert profile.execution_enabled is True


def test_compose_rejects_foreign_remote_root() -> None:
    site = load_site_template(SITE)
    try:
        compose_cluster_profile(
            site,
            UserIdentity(
                user="alice",
                connection="office",
                remote_root="/home/another-user/magvasp/runs",
            ),
        )
    except ValueError as exc:
        assert "username" in str(exc)
    else:
        raise AssertionError("foreign remote_root was accepted")


def test_compose_rejects_unknown_connection() -> None:
    site = load_site_template(SITE)
    try:
        compose_cluster_profile(site, UserIdentity(user="alice", connection="wifi"))
    except ValueError as exc:
        assert "unknown connection" in str(exc)
    else:
        raise AssertionError("unknown connection was accepted")


def test_example_identity_composes_without_enabling_execution() -> None:
    profile = compose_from_files(SITE, ROOT / "config/execution/identity.example.json")
    assert profile.user == "cluster_user"
    assert profile.gateway == "gateway.example.org"
    assert profile.execution_enabled is False


def test_cli_compose_writes_private_profile(tmp_path) -> None:
    output = tmp_path / "cluster.json"
    result = CliRunner().invoke(
        app,
        [
            "execution",
            "compose",
            "--site",
            str(SITE),
            "--user",
            "bob",
            "--connection",
            "office",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    profile = SiteTemplate.model_validate_json(SITE.read_bytes())  # sanity: site still valid
    assert profile.site_id == "team-example"
    text = output.read_text(encoding="utf-8")
    assert '"user": "bob"' in text
    assert "gateway.example.org" in text
    assert "/home/bob/magvasp/runs" in text
