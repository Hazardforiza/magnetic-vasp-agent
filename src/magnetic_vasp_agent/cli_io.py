"""Small helpers for keeping the CLI text boundary deterministic."""

from __future__ import annotations

import sys
from typing import Any, TextIO

from typer.core import TyperGroup


def configure_utf8_stdio(*streams: TextIO | None) -> None:
    """Prefer UTF-8 for CLI output, including redirected Windows streams.

    Python may initialize Windows standard streams with the active console code
    page (for example, ``gbk``).  Reconfiguring the text wrappers at the CLI
    boundary keeps human-readable and JSON output byte-stable when it is
    piped or captured.  Test runners and embedding callers can provide stream
    objects without ``reconfigure``; those are left untouched.
    """

    if not streams:
        streams = (sys.stdout, sys.stderr)
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace", newline="\n")
        except (AttributeError, OSError, ValueError):
            # Some capture and proxy streams expose ``reconfigure`` but do not
            # permit changing their encoding.  Click will still write through
            # those streams using their existing capture contract.
            continue


class Utf8TyperGroup(TyperGroup):
    """Configure output before argument parsing, help, or command execution."""

    def main(self, *args: Any, **kwargs: Any) -> Any:
        configure_utf8_stdio()
        return super().main(*args, **kwargs)
