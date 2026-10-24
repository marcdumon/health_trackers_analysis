"""Copies of source files into the project: a file is copied when the copy is missing or differs in size or change time,
through a .part name, so a copy is never half written. The source is never changed."""

import shutil
from datetime import datetime
from pathlib import Path


def copy_file(source: Path, target: Path) -> bool:
    """Copy one file when needed; True when it was copied."""
    info = source.stat()
    if target.exists():
        known = target.stat()
        if known.st_size == info.st_size and known.st_mtime_ns == info.st_mtime_ns:
            return False
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(f'{target.name}.part')
    shutil.copy2(source, part)
    part.replace(target)
    return True


def copy_version(source: Path, folder: Path, prefix: str) -> Path | None:
    """Keep a full export as a new file named by its change time, so every version stays; None when that version is
    already there."""
    target = folder / f'{prefix}_{datetime.fromtimestamp(source.stat().st_mtime):%Y%m%d_%H%M%S}{source.suffix}'
    return target if copy_file(source, target) else None
