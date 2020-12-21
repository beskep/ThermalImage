import sys
from pathlib import Path

PRJ_DIR = Path(__file__).parents[1].resolve()
SRC_DIR = PRJ_DIR.joinpath('src')

SCRIPT_DIR = PRJ_DIR.joinpath('scripts')
DATA_DIR = PRJ_DIR.joinpath('data')

_SRC_DIR = SRC_DIR.as_posix()
if _SRC_DIR not in sys.path:
  sys.path.insert(0, _SRC_DIR)
