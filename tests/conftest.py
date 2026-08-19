from pathlib import Path
import sys

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLUGIN_ROOT.parent
CORE_ROOT = REPO_ROOT / "vipr-core"

for path in (str(PLUGIN_ROOT), str(CORE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
