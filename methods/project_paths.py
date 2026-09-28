"""Portable paths shared by project-adapted baseline runners."""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the checkout root, or an explicitly selected external workspace."""
    override = os.environ.get("HSMAD_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]

