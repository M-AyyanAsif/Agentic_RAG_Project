"""Create timestamped SQLite backups for disaster recovery."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from backend.core.config import get_settings


def main() -> None:
    settings = get_settings()
    source = Path(settings.sqlite_path)
    backup_dir = Path(settings.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        print(f"Database file not found: {source}")
        return

    suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f"indus_guardian_{suffix}.db"
    shutil.copy2(source, target)
    print(f"Backup created: {target}")


if __name__ == "__main__":
    main()
