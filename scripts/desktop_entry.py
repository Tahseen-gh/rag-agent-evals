"""PyInstaller entrypoint: launches the Streamlit UI as a standalone desktop app.

Streamlit execs ui/app.py off disk at runtime, not as frozen bytecode, so
our own source (ui/, rag/, ml/, evals/, fastapi_app/, data/) ships as plain
--add-data files - only third-party deps get frozen normally. See
.github/workflows/build-windows.yml.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bundle_root() -> Path:
    # sys._MEIPASS is where PyInstaller extracts --add-data to; falls back
    # to repo root when run unfrozen (e.g. `python scripts/desktop_entry.py`)
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else Path(__file__).resolve().parent.parent


def main() -> None:
    root = _bundle_root()
    sys.path.insert(0, str(root))
    os.chdir(root)  # streamlit does some cwd-relative lookups

    import streamlit.web.cli as stcli

    app_path = root / "ui" / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(
            f"Expected bundled app entrypoint at {app_path}, but it's missing. "
            "This means the --add-data 'ui;ui' bundling step didn't run or "
            "didn't land where this script expects."
        )

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
        "--server.headless=false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
