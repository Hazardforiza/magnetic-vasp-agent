from __future__ import annotations

from pathlib import Path

def list_examples(root: Path) -> dict:
    """Return the versioned VASPKIT example families available locally."""
    root = root.resolve()
    examples = root / "examples"
    if not examples.is_dir():
        raise ValueError("VASPKIT examples directory is missing")
    families = []
    for path in sorted(examples.iterdir(), key=lambda p: p.name.casefold()):
        if path.is_dir() and not path.is_symlink():
            families.append({"name": path.name, "examples": sum(1 for p in path.rglob("*") if p.is_dir())})
    supported = {
        "band_kpoints": {
            "task": 303,
            "required_outputs": ["KPATH.in"],
            "validation": "vasp_line_mode_kpath",
        },
        "band_dat": {
            "task": 211,
            "required_outputs": ["KLABELS"],
            "required_any": ["REFORMATTED_BAND.dat", "BAND.dat"],
            "compatibility_outputs": ["band.dat"],
            "validation": "parsed_band_and_matching_klabels",
        },
        "tdos_dat": {
            "task": 111,
            "required_outputs": ["TDOS.dat"],
            "validation": "parsed_vaspkit_tdos",
        },
    }
    return {
        "schema_version": 2,
        "vaspkit_root": str(root),
        "families": families,
        "supported_adapters": supported,
        "unsupported_families": [item["name"] for item in families if item["name"] not in {"3D_band"}],
        "execution_integrated": True,
        "backends": {
            "native": "enabled_on_compatible_hosts",
            "wsl": "enabled_when_a_distribution_is_installed",
            "remote_ssh": "disabled_requires_control_plane_integration",
        },
        "safety": (
            "Named adapters validate freshly written outputs. The generic numeric "
            "entry records changes but does not claim scientific validation."
        ),
    }
