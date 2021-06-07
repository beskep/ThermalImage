"""
Stitcher용 전처리 함수
"""

import cv2 as cv
import skimage.exposure as skexp
from loguru import logger

from tools.imagetools import normalize_rgb_image_hist


def preprocess_ir(image, temperature_threshold=-30.0):
  mask = (image > temperature_threshold).astype('uint8')

  image = skexp.equalize_hist(image)
  image = skexp.rescale_intensity(image=image, out_range='uint8')
  image = cv.bilateralFilter(image, d=-1, sigmaColor=20, sigmaSpace=10)

  return image, mask


def preprocess_vis(image):
  image = normalize_rgb_image_hist(image)
  image = cv.bilateralFilter(image, d=-1, sigmaColor=20, sigmaSpace=10)

  return image, None


def _pass_fn(image):
  return image


def _bilateral_filter(image):
  return cv.bilateralFilter(image, d=-1, sigmaColor=20, sigmaSpace=10)


def _gaussian_filter(image):
  return cv.GaussianBlur(image, ksize=(5, 5), sigmaX=0)


class PanoramaPreprocess:

  def __init__(self,
               is_numeric,
               mask_threshold=-30.0,
               constrast='equalization',
               noise='bilateral'):
    if constrast not in (None, 'equalization', 'normalization'):
      raise ValueError
    if noise not in (None, 'bilateral', 'gaussian'):
      raise ValueError

    self._is_numeric = is_numeric
    self._mask_threshold = mask_threshold

    if constrast == 'equalization':
      self._contrast_fn = skexp.equalize_hist
      if not is_numeric:
        logger.warning('실화상에 Histogram equalization을 적용합니다. '
                       '예상치 못한 오류가 발생할 수 있습니다.')
    elif constrast == 'normalization':
      if is_numeric:
        logger.warning('열화상에 Histogram normalization을 적용할 수 없습니다. '
                       '명암 보정을 적용하지 않습니다')
        self._contrast_fn = _pass_fn
      else:
        self._contrast_fn = normalize_rgb_image_hist
    else:
      self._contrast_fn = _pass_fn

    if noise == 'bilateral':
      self._noise_fn = _bilateral_filter
    elif noise == 'gaussian':
      self._noise_fn = _gaussian_filter
    else:
      self._noise_fn = _pass_fn

  def __call__(self, image):
    if self._is_numeric:
      mask = (self._mask_threshold < image).astype('uint8')
    else:
      mask = None

    image = self._contrast_fn(image)
    image = skexp.rescale_intensity(image, out_range='uint8')
    image = self._noise_fn(image)

    return image, mask
