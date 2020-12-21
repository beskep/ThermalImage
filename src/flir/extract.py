from pathlib import Path

import utils

import numpy as np

import flir._flir_image_extractor as flirext
from tools import imagetools

EXIFTOOL_PATH = utils.SRC_DIR.joinpath('exiftool.exe')
if not EXIFTOOL_PATH.exists():
  raise FileNotFoundError(EXIFTOOL_PATH)


class FlirExtractor:

  def __init__(self, path=None, exiftool_path=None, debug=False):
    if exiftool_path is None:
      exiftool_path = EXIFTOOL_PATH.as_posix()

    # TODO: extractor 새로 만들기 / Atmospheric 변수 Exif 추출해서 적용
    self._extractor = flirext.FlirImageExtractor(exiftool_path=exiftool_path,
                                                 is_debug=debug)

    if path is not None:
      if not Path(path).exists():
        raise FileNotFoundError(path)

      self._extractor.process_image(path)

    self._path = path

  @property
  def extractor(self):
    return self._extractor

  def process_image(self, path):
    path = Path(path).resolve()
    if not path.exists():
      raise FileNotFoundError(path)

    self._extractor.process_image(path.as_posix())
    self._path = path

  def _check_path(self, path):
    if path is not None:
      path = Path(path).resolve()

      if path != self._path:
        self.process_image(path)
        self._path = path

    elif self._path is None:
      raise ValueError('이미지가 지정되지 않음')

  def extract_data(self, path=None):
    self._check_path(path)

    ir_array: np.ndarray = self.extractor.thermal_image_np
    if ir_array is None:
      raise ValueError('IR image 추출 실패')

    vis_array: np.ndarray = self.extractor.rgb_image_np
    if vis_array is None:
      raise ValueError('Vis image 추출 실패')

    return ir_array.copy(), vis_array.copy()

  def extract_ir(self, path=None):
    self._check_path(path)

    ir_array: np.ndarray = self.extractor.thermal_image_np
    if ir_array is None:
      raise ValueError('IR image 추출 실패')

    return ir_array.copy()

  def extract_vis(self, path=None):
    self._check_path(path)
    vis_array: np.ndarray = self.extractor.rgb_image_np
    if vis_array is None:
      raise ValueError('Vis image 추출 실패')

    return vis_array.copy()

  def write_data(self, ir_path, vis_path, image_path=None, csv_fmt='%.3e'):
    ir_array, vis_array = self.extract_data(image_path)

    if ir_path:
      ir_path = Path(ir_path)
      ir_suffix = ir_path.suffix

      if ir_suffix == '.npy':
        np.save(ir_path.as_posix(), ir_array)

      elif ir_suffix == '.csv':
        np.savetxt(ir_path.as_posix(), ir_array, fmt=csv_fmt, delimiter=',')

    if vis_path:
      imagetools.imsave(vis_path, vis_array)
