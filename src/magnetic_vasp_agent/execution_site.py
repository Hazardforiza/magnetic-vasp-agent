"""Compose a runtime ClusterProfile from a shared site plus one person's identity.

The site file is what the group can share. The identity file stays on one
computer and never goes into the knowledge index or the LLM prompt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from magnetic_vasp_agent.execution_models import WANNIER_MODULE, ClusterProfile, ExecutionModel

_USER_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SECRET_MARKERS = (
    "password=",
    "passwd=",
    "BEGIN OPENSSH",
    "BEGIN RSA PRIVATE",
    "MAGVASP_LLM_API_KEY",
    "-----BEGIN",
)


class SiteConnection(ExecutionModel):
    id: str = Field(min_length=1, max_length=32)
    gateway: str = Field(min_length=1, max_length=253)
    port: int = Field(ge=1, le=65535)
    notes: str | None = Field(default=None, max_length=300)

    @field_validator("id")
    @classmethod
    def safe_connection_id(cls, value: str) -> str:
        if re.fullmatch(r"[a-z][a-z0-9-]{0,31}", value) is None:
            raise ValueError("connection id must be lowercase kebab-case")
        return value


class SiteTemplate(ExecutionModel):
    schema_version: Literal[1] = 1
    site_id: str = Field(min_length=1, max_length=64)
    transport: Literal["ssh_direct_mpi"] = "ssh_direct_mpi"
    gateway_hostname: str
    node_port: int = Field(22, ge=1, le=65535)
    allowed_nodes: tuple[str, ...]
    vasp_module: str
    wannier_module: str = WANNIER_MODULE
    lmod_path: str | None = None
    modulepath: str | None = None
    mpi_path: str
    vasp_paths: dict[str, str]
    connections: tuple[SiteConnection, ...]
    remote_root_template: str
    notes: tuple[str, ...] = ()

    @field_validator("site_id")
    @classmethod
    def safe_site_id(cls, value: str) -> str:
        if re.fullmatch(r"[a-z][a-z0-9-]{0,63}", value) is None:
            raise ValueError("site_id must be lowercase kebab-case")
        return value

    @field_validator("remote_root_template")
    @classmethod
    def template_uses_user(cls, value: str) -> str:
        if "{user}" not in value:
            raise ValueError("remote_root_template must contain {user}")
        if any(token in value for token in ("{", "}")) and value.count("{user}") != 1:
            extra = value.replace("{user}", "")
            if "{" in extra or "}" in extra:
                raise ValueError("remote_root_template may only interpolate {user}")
        return value

    @model_validator(mode="after")
    def unique_connections(self) -> "SiteTemplate":
        ids = [item.id for item in self.connections]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("site connections must be a non-empty unique list")
        return self


class UserIdentity(ExecutionModel):
    schema_version: Literal[1] = 1
    user: str
    connection: str
    execution_enabled: bool = False
    remote_root: str | None = None
    gateway: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    reviewer: str | None = None

    @field_validator("user", "connection", "reviewer", "gateway")
    @classmethod
    def safe_optional_token(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if _USER_TOKEN.fullmatch(value) is None:
            raise ValueError("identity tokens must be nonempty shell-safe host or username tokens")
        return value


def default_remote_root(user: str, template: str) -> str:
    return template.format(user=user)


def connection_named(site: SiteTemplate, connection_id: str) -> SiteConnection:
    matches = [item for item in site.connections if item.id == connection_id]
    if len(matches) != 1:
        known = ", ".join(item.id for item in site.connections)
        raise ValueError(f"unknown connection {connection_id!r}; site defines: {known}")
    return matches[0]


def compose_cluster_profile(site: SiteTemplate, identity: UserIdentity) -> ClusterProfile:
    site = SiteTemplate.model_validate_json(site.model_dump_json())
    identity = UserIdentity.model_validate_json(identity.model_dump_json())
    hop = connection_named(site, identity.connection)
    remote_root = identity.remote_root or default_remote_root(identity.user, site.remote_root_template)
    if identity.user not in remote_root.split("/"):
        raise ValueError("remote_root must include the cluster username as a path component")
    return ClusterProfile(
        transport=site.transport,
        gateway=identity.gateway or hop.gateway,
        gateway_hostname=site.gateway_hostname,
        user=identity.user,
        port=identity.port if identity.port is not None else hop.port,
        node_port=site.node_port,
        allowed_nodes=site.allowed_nodes,
        vasp_module=site.vasp_module,
        wannier_module=site.wannier_module,
        lmod_path=site.lmod_path,
        modulepath=site.modulepath,
        mpi_path=site.mpi_path,
        vasp_paths=site.vasp_paths,
        execution_enabled=identity.execution_enabled,
        remote_root=remote_root,
    )


def load_site_template(path: Path) -> SiteTemplate:
    return SiteTemplate.model_validate_json(Path(path).read_bytes())


def load_user_identity(path: Path) -> UserIdentity:
    return UserIdentity.model_validate_json(Path(path).read_bytes())


def compose_from_files(site: Path, identity: Path) -> ClusterProfile:
    return compose_cluster_profile(load_site_template(site), load_user_identity(identity))


def write_cluster_profile(profile: ClusterProfile, output: Path, *, overwrite: bool = False) -> None:
    output = Path(output)
    if output.exists() or output.is_symlink():
        if not overwrite:
            raise FileExistsError("composed profile exists; pass overwrite or choose a new path")
        if output.is_symlink() or not output.is_file():
            raise ValueError("refusing to overwrite a non-regular composed profile")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(profile.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    if overwrite and output.is_file():
        output.write_bytes(payload)
        return
    with output.open("xb") as handle:
        handle.write(payload)


def reject_secret_text(text: str | None, *, label: str) -> str | None:
    if text is None:
        return text
    folded = text.replace(" ", "")
    if any(marker.casefold() in text.casefold() or marker.casefold() in folded.casefold() for marker in _SECRET_MARKERS):
        raise ValueError(f"{label} looks like a secret and cannot be stored in usage knowledge")
    return text
