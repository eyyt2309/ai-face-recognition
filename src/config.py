"""Shared configuration helpers."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolveModelPath(value, variableName):
    """Resolve a model path, treating relative paths as project-root relative."""

    if not value:
        raise RuntimeError(
            f"{variableName} is not set in the project .env file"
        )

    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"{variableName} points to a missing file: {path}"
        )
    return path
