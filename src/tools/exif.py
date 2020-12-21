from subprocess import check_output

import utils

import yaml

EXIFTOOL_PATH = utils.SRC_DIR.joinpath('exiftool.exe')
if not EXIFTOOL_PATH.exists():
  raise FileNotFoundError(EXIFTOOL_PATH)


def run_exif_tool(*args):
  args = (EXIFTOOL_PATH.as_posix(),) + args
  res = check_output(args)

  return res


def get_exif_tags(image_path: str, *args):
  tag_byte = run_exif_tool(image_path, '-j', *args)
  tag = yaml.load(tag_byte.decode(), Loader=yaml.FullLoader)

  return tag


def get_exif_binary(image_path: str, tag):
  res = run_exif_tool(tag, '-b', image_path)

  return res
