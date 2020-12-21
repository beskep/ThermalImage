import cv2 as cv
import numpy as np
from matplotlib.colors import ListedColormap

import tools.exif


class Colormap(ListedColormap):
  # TODO: FLIR 파일로부터 읽는 함수

  @classmethod
  def from_uint8_colors(cls, colors: np.ndarray, space='rgb'):
    if space.lower() not in {'rgb', 'ycrcb'}:
      raise ValueError(space)

    if space == 'ycrcb':
      colors = colors.reshape([1, -1, 3])
      colors = cv.cvtColr(colors, code=cv.COLOR_YCrCb2RGB)

    colors = colors.astype('float').reshape([-1, 3]) / 255.0
    instance = cls(colors=colors)

    return instance

  @classmethod
  def from_flir_file(cls, path):
    colors = tools.exif.get_exif_binary(image_path=path, tag='-Palette')
    colors = np.array(list(colors)).astype('uint8')
    instance = cls.from_uint8_colors(colors=colors, space='ycrcb')

    return instance

  @classmethod
  def from_uint8_text(cls, path, space='rgb'):
    colors = np.loadtxt(path)
    instance = cls.from_uint8_colors(colors=colors, space=space)

    return instance
