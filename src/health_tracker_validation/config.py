"""The settings of config.toml and the folders of the project."""

import tomllib
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
FIGURES = ROOT / 'figures'
SECRETS = ROOT / 'secrets'


@dataclass(frozen=True)
class Settings:
    """The study period and the places the data comes from."""

    start: datetime
    tz: ZoneInfo
    polar_phone_dir: Path
    polar_pc_dir: Path
    automation_desk_db: Path
    health_connect_zip: Path
    sleep_as_android_zip: Path


@cache
def settings() -> Settings:
    """The settings of config.toml."""
    raw = tomllib.loads((ROOT / 'config.toml').read_text())
    sources = {key: Path(value) for key, value in raw['sources'].items()}
    return Settings(start=datetime.fromisoformat(raw['study']['start']), tz=ZoneInfo(raw['study']['timezone']), **sources)
