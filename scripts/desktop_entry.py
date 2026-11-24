"""PyInstaller entrypoint: launches the Streamlit UI as a standalone desktop app.

Streamlit execs ui/app.py as a real script file at runtime (its file-watcher
and script-runner read it off disk), rather than importing it as frozen
bytecode - so unlike a normal PyInstaller build, this app's own source tree
(ui/, rag/, ml/, evals/, fastapi_app/, data/) has to ship as plain files via
--add-data rather than being frozen into the executable. Only third-party
dependencies (streamlit, torch, sentence-transformers, ...) get frozen
normally. See .github/workflows/build-windows.yml for the exact build.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bundle_root() -> Path:
    # PyInstaller sets sys._MEIPASS to the extraction root: a temp dir for
    # --onefile, or the dist folder itself for --onedir. Either way, the
    # --add-data trees land directly under it. Unfrozen (running this file
    # directly with `python scripts/desktop_entry.py` for local testing),
    # fall back to the repo root.
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else Path(__file__).resolve().parent.parent


def main() -> None:
    root = _bundle_root()
    sys.path.insert(0, str(root))
    # Several modules (rag/ingestion.py, rag/backends/learned.py, ml/reranker.py)
    # resolve data/model-artifact paths from Path(__file__).resolve().parents[N],
    # not cwd, so this isn't load-bearing for those - but Streamlit itself and
    # some of its dependencies do use cwd-relative lookups, so set it anyway to
    # match a normal `streamlit run` invocation from the repo root.
    os.chdir(root)

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
