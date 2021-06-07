from pathlib import Path
from typing import Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.io import imsave

__all__ = ('read_image_cv', 'show_images_cv', 'show_images_mpl',
           'array_to_image', 'imsave', 'normalize_rgb_image_hist')

sns.set_style('white')


def read_image_cv(path, flags=cv.IMREAD_COLOR, to_rgb=False) -> np.ndarray:
  path = Path(path).resolve()
  if not path.exists():
    raise FileNotFoundError(path)

  image = cv.imread(filename=path.as_posix(), flags=flags)

  if to_rgb and image.ndim == 3:
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

  return image


def show_images_cv(images: list, titles=None, wait_and_destroy=True):
  if not titles or (len(images) != len(titles)):
    titles = ['title {}'.format(x) for x in range(len(images))]

  for image, title in zip(images, titles):
    cv.imshow(title, image)

  if wait_and_destroy:
    cv.waitKey()
    cv.destroyAllWindows()


def show_images_mpl(images: list,
                    titles=None,
                    aspect=(16, 9),
                    figsize=(16, 9),
                    cmap='gray',
                    show=True) -> Tuple[plt.Figure, plt.Axes]:
  images_count = len(images)
  if images_count == 0:
    raise ValueError

  if images_count == 1:
    nrows, ncols = 1, 1
  else:
    nrows = max(round(np.sqrt(images_count * aspect[1] / aspect[0])), 1)
    ncols = int(max(np.ceil(images_count / nrows), 1))

  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

  axes_flatten = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
  for idx, image, in enumerate(images):
    axes_flatten[idx].imshow(image, cmap=cmap)
    axes_flatten[idx].set_xticks([])
    axes_flatten[idx].set_yticks([])

  if titles is not None:
    for idx, title in enumerate(titles):
      axes_flatten[idx].set_title(title)

  fig.tight_layout()

  if show:
    plt.show()

  return fig, axes


def normalize_array(arr: np.ndarray):
  """
  0~1 범위로 normalize
  skimage.exposure.rescale_intensity 쓸 것
  """
  arr = arr - np.nanmin(arr)
  arr /= np.nanmax(arr)

  return arr


def array_to_image(arr: np.ndarray, fill_na=0.0):
  """
  skimage.exposure.rescale_intensity 쓸 것
  """
  # image = arr - np.nanmin(arr)
  # image = (255 * image / np.nanmax(image))
  image = normalize_array(arr.astype('float')) * 255

  if fill_na is not None:
    image = np.nan_to_num(image, nan=fill_na)

  return image.astype(np.uint8)


def normalize_rgb_image_hist(image):
  image_yuv = cv.cvtColor(image, cv.COLOR_RGB2YUV)
  image_yuv[:, :, 0] = cv.normalize(image_yuv[:, :, 0],
                                    dst=None,
                                    alpha=0,
                                    beta=255,
                                    norm_type=cv.NORM_MINMAX)
  equalized = cv.cvtColor(image_yuv, cv.COLOR_YUV2RGB)

  return equalized


def _mask_range(mask, axis):
  mask_ = np.any(mask, axis=axis)
  c1 = np.argmax(mask_)
  c2 = len(mask_) - np.argmax(mask_[::-1])

  return int(c1), int(c2)


def mask_bbox(mask: np.ndarray, morphology_open=True):
  """
  마스크 bounding box 좌표 찾기
  https://stackoverflow.com/questions/39206986/numpy-get-rectangle-area-just-the-size-of-mask/48346079"""
  if mask.ndim != 2:
    raise ValueError

  if morphology_open:
    kernel = np.ones(shape=(3, 3), dtype='uint8')
    mask_ = cv.morphologyEx(src=mask, op=cv.MORPH_OPEN, kernel=kernel)
  else:
    mask_ = mask

  xx = _mask_range(mask=mask_, axis=0)
  yy = _mask_range(mask=mask_, axis=1)

  return xx + yy
