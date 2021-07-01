import sys
from pathlib import Path

import yaml
from loguru import logger
from rich.logging import RichHandler

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
  # pyinstaller
  PRJ_DIR = Path(getattr(sys, '_MEIPASS'))
  SRC_DIR = PRJ_DIR.joinpath('src')
else:
  SRC_DIR = Path(__file__).parent.resolve()
  PRJ_DIR = SRC_DIR.parent

SCRIPT_DIR = PRJ_DIR.joinpath('scripts')
DATA_DIR = PRJ_DIR.joinpath('data')

_SRC_DIR = SRC_DIR.as_posix()
if _SRC_DIR not in sys.path:
  sys.path.insert(0, _SRC_DIR)


def get_config():
  config_path = SRC_DIR.joinpath('config.yaml')
  with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

  return config


def set_logger():
  logger.remove()

  if any('debug' in x.lower() for x in sys.argv):
    level = 'DEBUG'
  else:
    level = 'INFO'

  rich_handler = RichHandler(show_time=True)
  logger.add(rich_handler, level=level, format='{message}', enqueue=True)
  logger.add('pano.log',
             level='DEBUG',
             rotation='1 week',
             retention='1 month',
             encoding='utf-8-sig',
             enqueue=True)
