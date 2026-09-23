"""Constrained, auditable VASPKIT execution adapters.

Only fixed menu input or a bounded numeric task is accepted. No shell is
involved. ``remote_ssh`` is deliberately modelled but disabled here; remote
execution belongs to the separately approved cluster control plane.
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from magnetic_vasp_agent.vaspkit_plot import parse_band, parse_klabels, parse_tdos


class VaspkitBackend(str, Enum):
    NATIVE = "native"
    WSL = "wsl"
    REMOTE_SSH = "remote_ssh"


TASK_INPUTS = {
    "band_kpoints": "303\n",
    "band_dat": "211\n",
    "tdos_dat": "111\n",
}

_VERSION = re.compile(r"\b(\d+\.\d+(?:\.\d+)?)\b")
_WSL_DISTRIBUTION = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_MAX_CAPTURE_CHARS = 4000


@dataclass(frozen=True)
class _FileState:
    size: int
    modified_ns: int


@dataclass(frozen=True)
class _Execution:
    completed: subprocess.CompletedProcess[str]
    runtime: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("calculation directory must not be a symlink")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError("calculation directory must be a regular directory")
    return resolved


def _executable(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("VASPKIT executable must not be a symlink")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError("VASPKIT executable must be a regular file")
    return resolved


def _file_format(path: Path) -> str:
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic == b"\x7fELF":
        return "elf"
    if magic[:2] == b"MZ":
        return "pe"
    if magic[:2] == b"#!":
        return "script"
    return "unknown"


def _snapshot(directory: Path) -> dict[str, _FileState]:
    states: dict[str, _FileState] = {}
    for path in directory.iterdir():
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            states[path.name] = _FileState(stat.st_size, stat.st_mtime_ns)
    return states


def _was_written(path: Path, before: dict[str, _FileState]) -> bool:
    if path.name not in before:
        return True
    stat = path.stat()
    return before[path.name] != _FileState(stat.st_size, stat.st_mtime_ns)


def _output(path: Path, before: dict[str, _FileState]) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "new": path.name not in before,
        "changed": _was_written(path, before),
    }


def _version_from_runtime(stdout: str, stderr: str) -> tuple[str | None, str | None]:
    for line in (stdout + "\n" + stderr).splitlines():
        if "vaspkit" not in line.casefold():
            continue
        match = _VERSION.search(line)
        if match:
            return match.group(1), "runtime_banner"
    return None, None


def _version_from_package(executable: Path) -> tuple[str | None, str | None]:
    for candidate in (executable.parent / "changelog", executable.parent.parent / "changelog"):
        if candidate.is_file() and not candidate.is_symlink():
            try:
                first_line = candidate.read_text(encoding="utf-8").splitlines()[0]
            except (OSError, UnicodeError, IndexError):
                continue
            match = re.fullmatch(r"Version\s+(\d+\.\d+(?:\.\d+)?)\s+.*", first_line)
            if match:
                return match.group(1), "package_changelog"
    return None, None


def _wsl_base(wsl: str, distribution: str | None) -> list[str]:
    if distribution is not None and not _WSL_DISTRIBUTION.fullmatch(distribution):
        raise ValueError("WSL distribution contains unsupported characters")
    command = [wsl]
    if distribution:
        command.extend(["--distribution", distribution])
    return command


def _wsl_query(command: list[str], *, description: str) -> str:
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"WSL backend unavailable while reading {description}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-1000:]
        raise RuntimeError(
            f"WSL backend unavailable while reading {description}"
            + (f": {detail}" if detail else "")
        )
    value = completed.stdout.strip()
    if not value:
        raise RuntimeError(f"WSL returned an empty {description}")
    return value


def _execute(
    directory: Path,
    executable: Path,
    *,
    backend: VaspkitBackend,
    menu: str,
    arguments: list[str] | None = None,
    timeout: int,
    wsl_distribution: str | None,
) -> _Execution:
    if backend == VaspkitBackend.REMOTE_SSH:
        raise ValueError(
            "remote_ssh VASPKIT execution is disabled; use an approved remote control-plane adapter"
        )
    directory = _directory(directory)
    executable = _executable(executable)
    executable_format = _file_format(executable)
    environment: dict[str, str | None] = {
        "host_system": platform.system(),
        "host_release": platform.release(),
        "wsl_distribution": None,
        "wsl_kernel": None,
    }
    arguments = arguments or []

    if backend == VaspkitBackend.NATIVE:
        if os.name == "nt" and executable_format == "elf":
            raise ValueError(
                "native backend cannot execute a Linux ELF program on Windows; configure backend=wsl"
            )
        command = [str(executable), *arguments]
    elif backend == VaspkitBackend.WSL:
        if os.name != "nt":
            raise ValueError("wsl backend is available only from a Windows host")
        wsl = shutil.which("wsl.exe")
        if wsl is None:
            raise RuntimeError("WSL backend unavailable: wsl.exe was not found")
        base = _wsl_base(wsl, wsl_distribution)
        linux_directory = _wsl_query(
            [*base, "--exec", "wslpath", "-a", str(directory)],
            description="calculation-directory path",
        )
        linux_executable = _wsl_query(
            [*base, "--exec", "wslpath", "-a", str(executable)],
            description="VASPKIT executable path",
        )
        environment["wsl_distribution"] = wsl_distribution or "default"
        environment["wsl_kernel"] = _wsl_query(
            [*base, "--exec", "uname", "-srmo"], description="kernel version"
        )
        command = [
            *base,
            "--cd",
            linux_directory,
            "--exec",
            linux_executable,
            *arguments,
        ]
    else:  # pragma: no cover
        raise ValueError(f"unsupported VASPKIT backend: {backend}")

    try:
        completed = subprocess.run(
            command,
            input=menu,
            text=True,
            cwd=directory if backend == VaspkitBackend.NATIVE else None,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"VASPKIT {backend.value} execution failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-1000:]
        raise RuntimeError(
            f"VASPKIT failed with exit code {completed.returncode}"
            + (f": {detail}" if detail else "")
        )

    version, version_source = _version_from_runtime(completed.stdout, completed.stderr)
    if version is None:
        version, version_source = _version_from_package(executable)
    runtime = {
        "backend": backend.value,
        "executable": {
            "path": str(executable),
            "sha256": _sha256(executable),
            "size_bytes": executable.stat().st_size,
            "format": executable_format,
            "version": version,
            "version_source": version_source,
        },
        "environment": environment,
    }
    return _Execution(completed=completed, runtime=runtime)


def _validate_kpath(path: Path) -> None:
    try:
        lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines()]
    except UnicodeDecodeError as exc:
        raise ValueError("KPATH.in must be ASCII text") from exc
    if len(lines) < 6 or lines[2].casefold() != "line-mode" or lines[3].casefold() != "reciprocal":
        raise ValueError("KPATH.in is not a VASP line-mode reciprocal path")
    try:
        points_per_segment = int(lines[1])
    except ValueError as exc:
        raise ValueError("KPATH.in has an invalid interpolation count") from exc
    if points_per_segment < 2:
        raise ValueError("KPATH.in interpolation count must be at least 2")
    coordinates = []
    for line in lines[4:]:
        if not line:
            continue
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(f"KPATH.in has an invalid coordinate row: {line!r}")
        try:
            point = tuple(float(value) for value in fields[:3])
        except ValueError as exc:
            raise ValueError(f"KPATH.in has a non-numeric coordinate: {line!r}") from exc
        if not all(math.isfinite(value) for value in point):
            raise ValueError("KPATH.in contains a non-finite coordinate")
        coordinates.append(point)
    if len(coordinates) < 2 or len(coordinates) % 2:
        raise ValueError("KPATH.in must contain endpoint pairs")


def _changed_required(path: Path, before: dict[str, _FileState]) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"VASPKIT completed but expected output is missing: {path.name}")
    if not _was_written(path, before):
        raise ValueError(
            f"VASPKIT completed but expected output was not written by this run: {path.name}"
        )


def _fixed_outputs(directory: Path, task: str, before: dict[str, _FileState]) -> list[Path]:
    if task == "band_kpoints":
        path = directory / "KPATH.in"
        _changed_required(path, before)
        _validate_kpath(path)
        return [path]
    if task == "tdos_dat":
        path = directory / "TDOS.dat"
        _changed_required(path, before)
        parse_tdos(path)
        return [path]

    files = {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    labels = files.get("KLABELS", directory / "KLABELS")
    _changed_required(labels, before)
    candidate_names = [
        "REFORMATTED_BAND.dat",
        "BAND.dat",
        "band.dat",  # compatibility with the original adapter
    ]
    written = [
        files[name]
        for name in candidate_names
        if name in files and _was_written(files[name], before)
    ]
    if not written:
        raise ValueError(
            "VASPKIT completed but wrote no recognized band data "
            "(REFORMATTED_BAND.dat or BAND.dat)"
        )
    for path in written:
        table = parse_band(path)
        parse_klabels(labels, table.kpath)
    return [*written, labels]


def run_vaspkit(
    directory: Path,
    task: str,
    executable: Path,
    *,
    backend: VaspkitBackend | str = VaspkitBackend.NATIVE,
    wsl_distribution: str | None = None,
) -> dict:
    """Run one scientifically validated fixed VASPKIT adapter."""

    if task not in TASK_INPUTS:
        raise ValueError("task must be band_kpoints, band_dat, or tdos_dat")
    try:
        selected_backend = VaspkitBackend(backend)
    except ValueError as exc:
        raise ValueError("backend must be native, wsl, or remote_ssh") from exc
    directory = _directory(directory)
    before = _snapshot(directory)
    execution = _execute(
        directory,
        executable,
        backend=selected_backend,
        menu=TASK_INPUTS[task],
        timeout=300,
        wsl_distribution=wsl_distribution,
    )
    paths = _fixed_outputs(directory, task, before)
    outputs = {path.name: _output(path, before) for path in paths}
    return {
        "schema_version": 2,
        "task": task,
        "directory": str(directory),
        "runtime": execution.runtime,
        "outputs": outputs,
        "stdout": execution.completed.stdout[-_MAX_CAPTURE_CHARS:],
        "stderr": execution.completed.stderr[-_MAX_CAPTURE_CHARS:],
    }


def run_vaspkit_task(
    directory: Path,
    task_id: int,
    executable: Path,
    inputs: list[str] | None = None,
    *,
    backend: VaspkitBackend | str = VaspkitBackend.NATIVE,
    wsl_distribution: str | None = None,
) -> dict:
    """Run a numeric task without claiming scientific output validation."""

    if not 0 <= task_id <= 9999:
        raise ValueError("VASPKIT task id must be between 0 and 9999")
    if inputs is not None and len(inputs) > 32:
        raise ValueError("VASPKIT task accepts at most 32 bounded menu inputs")
    if any("\n" in item or "\r" in item or len(item) > 256 for item in (inputs or [])):
        raise ValueError("VASPKIT menu inputs must be single lines of at most 256 characters")
    try:
        selected_backend = VaspkitBackend(backend)
    except ValueError as exc:
        raise ValueError("backend must be native, wsl, or remote_ssh") from exc
    directory = _directory(directory)
    executable = _executable(executable)
    menu_text = executable.read_bytes()
    if not re.search(rb"\b" + str(task_id).encode() + rb"\)\s", menu_text):
        raise ValueError(f"VASPKIT task {task_id} is not present in this executable")
    before = _snapshot(directory)
    menu = "\n".join(str(item) for item in (inputs or [])) + ("\n" if inputs else "")
    execution = _execute(
        directory,
        executable,
        backend=selected_backend,
        menu=menu,
        arguments=["-task", str(task_id)],
        timeout=900,
        wsl_distribution=wsl_distribution,
    )
    changed: dict[str, dict] = {}
    for path in directory.iterdir():
        if path.is_file() and not path.is_symlink() and _was_written(path, before):
            changed[path.name] = _output(path, before)
    return {
        "schema_version": 2,
        "task_id": task_id,
        "directory": str(directory),
        "runtime": execution.runtime,
        "scientific_validation": "not_validated_generic_entry",
        "changed_outputs": changed,
        "stdout": execution.completed.stdout[-_MAX_CAPTURE_CHARS:],
        "stderr": execution.completed.stderr[-_MAX_CAPTURE_CHARS:],
    }
