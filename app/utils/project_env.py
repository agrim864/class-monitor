from __future__ import annotations

import os
from pathlib import Path


def load_project_env(project_root: Path, *, override: bool = False) -> Path | None:
    """Load a simple root .env file into os.environ.

    This keeps the project dependency-light and is enough for the repo's
    script-style configuration needs.
    """

    env_path = project_root / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value

    return env_path
