"""
2021-06-09
spherical로 투영한 파노라마를 plane 투영으로 바꾸기 시도
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import warp

import cv2 as cv


def print_range(coord: np.ndarray):
  xrange = (np.min(coord[:, 0]), np.max(coord[:, 0]))
  yrange = (np.min(coord[:, 1]), np.max(coord[:, 1]))

  print(f'xrange: {xrange} | yrange: {yrange}')


def spherical2plane(image: np.ndarray, f: float, center=None):
  if image.ndim != 2:
    raise ValueError

  if center is None:
    center = np.array(image.shape[::-1]) / 2

  def inv_map(coord: np.ndarray):
    print_range(coord)
    coord -= center
    print_range(coord)

    # r0 = np.sqrt(np.sum(np.square(coord), axis=1))
    # theta = np.arctan2(coord[:, 1], coord[:, 0])
    # r1 = f * np.arctan(r0 / f) * 2
    # coord = np.vstack([r1 * np.cos(theta), r1 * np.sin(theta)]).T
    # print_range(coord)

    r0 = np.sqrt(np.sum(np.square(coord), axis=1))
    theta = np.arctan2(coord[:, 1], coord[:, 0])
    r1 = f * np.sin(2 * np.arctan(r0 / (2 * f))) * 2
    coord = np.vstack([r1 * np.cos(theta), r1 * np.sin(theta)]).T
    print_range(coord)

    coord += center
    print_range(coord)

    print('x range: [{}, {}]'.format(np.min(coord[:, 0]), np.max(coord[:, 0])))
    print('y range: [{}, {}]'.format(np.min(coord[:, 1]), np.max(coord[:, 1])))

    return coord

  warped_image = warp(image=image, inverse_map=inv_map)

  return warped_image


if __name__ == '__main__':
  path = Path(r'D:\test\panorama\MainBldgFrontPanTiltTripodResult\panorama.png')
  assert path.exists()

  image = imread(path.as_posix())
  assert image.ndim == 2

  warped = spherical2plane(image=image, f=1215.230182027216)

  plt.imshow(warped)
  plt.show()

  # cases_count = 5
  # xcenters = (image.shape[1] * np.arange(0,
  #                                        (cases_count + 2)) / (cases_count + 2))
  # xcenters = xcenters[1:-1]

  # fig, axes = plt.subplots(2, 3, figsize=(16, 9))
  # for idx, xc in enumerate(xcenters):
  #   img = spherical2plane(image=image,
  #                         f=1215.230182027216,
  #                         center=(image.shape[0] / 2, xc))
  #   axes.ravel()[idx].imshow(img)

  # plt.show()
