import logging
import logging.config
import sys
from pathlib import Path

import yaml
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


def get_logger():
  logger = logging.getLogger('IR')

  return logger


# logger 설정
if not logging.getLogger().handlers:
  config = get_config()

  logging.config.dictConfig(config['logging'])

  rich_handler = RichHandler(level=logging.INFO, show_time=True)
  logging.getLogger('IR').addHandler(rich_handler)

  try:
    from kivy.logger import Logger as kvlogger

    kvlogger.handlers = [
        handler for handler in kvlogger.handlers
        if not isinstance(handler, logging.StreamHandler)
    ]
    kvlogger.addHandler(rich_handler)
  except ModuleNotFoundError:
    pass

  logger = get_logger()
  logger.info('project dir: %s', PRJ_DIR)
  logger.info('src dir: %s', SRC_DIR)
