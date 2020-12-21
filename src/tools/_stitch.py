"""
2020.10.26
버리는게 나을듯
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def warp_perspective_points(points: np.ndarray, matrix: np.ndarray):
  assert points.shape[1] == 2
  assert matrix.shape == (3, 3)

  points_matrix = np.hstack(
      [points, np.full([points.shape[0], 1], fill_value=1)])

  xyz = matrix @ points_matrix.T
  xy = xyz[:2, :] / xyz[2, :]
  res = xy[:2, :].T
  return res


class MatchingImage:

  def __init__(
      self,
      image: np.ndarray,
      preprocess: list,
      detector: cv.Feature2D,
      mask=None,
  ):
    self.original_image = image
    self.image = self.original_image.copy()
    self.mask = mask

    if preprocess:
      for fn in preprocess:
        self.image = fn(self.image)

    if mask is not None:
      mask = mask.astype(self.image.dtype)

    kp, desc = detector.detectAndCompute(self.image, mask=mask)
    self.keypoints = kp
    self.desc = desc


class Stitcher:

  def __init__(self):
    self._detector = None
    self._matcher = None
    self.preprocess = []
    self.match_thershold = 0.75
    self.min_match_count = 4
    self.homography_method = cv.RHO

  @property
  def detector(self) -> cv.Feature2D:
    if self._detector is None:
      self._detector = cv.ORB_create()
    return self._detector

  @detector.setter
  def detector(self, value):
    self._detector = value

  @property
  def matcher(self) -> cv.DescriptorMatcher:
    if self._matcher is None:
      self._matcher = cv.FlannBasedMatcher_create()
    return self._matcher

  @matcher.setter
  def matcher(self, value):
    self._matcher = value

  def match(self,
            image1: np.ndarray,
            image2: np.ndarray,
            mask1=None,
            mask2=None):
    img1 = MatchingImage(image=image1,
                         preprocess=self.preprocess,
                         detector=self.detector,
                         mask=mask1)
    img2 = MatchingImage(image=image2,
                         preprocess=self.preprocess,
                         detector=self.detector,
                         mask=mask2)

    if any(x.desc is None for x in [img1, img2]):
      matches = None
      matches_mask = None
    else:
      matches = self.matcher.knnMatch(queryDescriptors=np.float32(img1.desc),
                                      trainDescriptors=np.float32(img2.desc),
                                      k=2)
      matches_mask = [
          [1, 0] if
          (x.distance < y.distance * self.match_thershold) else [0, 0]
          for x, y in matches
      ]

    return img1, img2, matches, matches_mask

  def find_homography(self, src_points, dst_points, **kwargs):
    homography_matrix, _ = cv.findHomography(srcPoints=src_points,
                                             dstPoints=dst_points,
                                             method=self.homography_method,
                                             **kwargs)
    return homography_matrix


class MatchingImages:

  def __init__(self,
               image1: np.ndarray,
               image2: np.ndarray,
               stitcher: Stitcher,
               mask1=None,
               mask2=None,
               draw_original=True):

    self.stitcher = stitcher

    self.matching_image1: MatchingImage = None
    self.matching_image2: MatchingImage = None
    self.matches = None
    self.match_mask = None
    self.homography_matrix = None
    self.draw_original = draw_original
    self.warp_image1 = None
    self.warp_image2 = None

    self._match(image1, image2, mask1, mask2)

  @property
  def draw_image1(self):
    return (self.matching_image1.original_image
            if self.draw_original else self.matching_image1.image)

  @property
  def draw_image2(self):
    return (self.matching_image2.original_image
            if self.draw_original else self.matching_image2.image)

  def _match(self, image1, image2, mask1, mask2):
    img1, img2, matches, mask = self.stitcher.match(image1=image1,
                                                    image2=image2,
                                                    mask1=mask1,
                                                    mask2=mask2)
    self.matching_image1 = img1
    self.matching_image2 = img2
    self.matches = matches
    self.match_mask = mask

    if matches is not None:
      good_matches = [x[0] for x, y in zip(matches, mask) if y[0]]
      if not good_matches:
        good_matches = matches

      src_points = np.array(
          [img1.keypoints[x.queryIdx].pt for x in good_matches],
          dtype=np.float32).reshape([-1, 1, 2])
      dst_points = np.array(
          [img2.keypoints[x.trainIdx].pt for x in good_matches],
          dtype=np.float32).reshape([-1, 1, 2])

      self.homography_matrix = self.stitcher.find_homography(
          src_points=src_points, dst_points=dst_points)
    else:
      self.homography_matrix = None

  def draw_matches_image(self, matches_mask=True):
    if isinstance(matches_mask, bool) and matches_mask:
      matches_mask = self.match_mask

    res = cv.drawMatchesKnn(img1=self.draw_image1,
                            keypoints1=self.matching_image1.keypoints,
                            img2=self.draw_image2,
                            keypoints2=self.matching_image2.keypoints,
                            matches1to2=self.matches,
                            outImg=None,
                            matchColor=[255, 55, 55],
                            singlePointColor=[55, 55, 255],
                            matchesMask=matches_mask)
    return res

  def warp_perspective(self):
    if self.homography_matrix is None:
      raise ValueError('homography matrix is None')

    h1, w1 = self.matching_image1.original_image.shape[:2]
    shape2 = self.matching_image2.original_image.shape[:2]
    src_points = np.array([
        [0, 0],
        [0, w1 - 1],
        [h1 - 1, 0],
        [h1 - 1, w1 - 1],
    ])
    warp_shape = warp_perspective_points(src_points, self.homography_matrix)
    warp_shape_min = np.min(warp_shape, axis=0).flatten()
    dst_shift = np.max([[0, 0], -warp_shape_min], axis=0).astype(np.int)
    all_points = np.vstack([warp_shape_min, [0, 0], shape2])
    res_shape = np.ceil(
        np.max(all_points, axis=0) - np.min(all_points, axis=0)).flatten()

    warp_image1 = cv.warpPerspective(src=self.draw_image1,
                                     M=self.homography_matrix,
                                     dsize=(int(res_shape[1]),
                                            int(res_shape[0])))

    warp_image2 = np.full_like(warp_image1, fill_value=np.nan)
    warp_image2[dst_shift[0]:(shape2[0] + dst_shift[0]),
                dst_shift[1]:(shape2[1] + dst_shift[1])] = self.draw_image2

    self.warp_image1 = warp_image1
    self.warp_image2 = warp_image2

    return warp_image1, warp_image2

  def plot_warp_images(self, figsize=None):
    warp_image1, warp_image2 = self.warp_perspective()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    axes[0, 0].imshow(self.draw_image1)
    axes[0, 1].imshow(self.draw_image2)
    axes[1, 0].imshow(warp_image1)
    axes[1, 1].imshow(warp_image2)

    fig.tight_layout()

    return fig, axes

  def draw_stitched_image(self):
    if any(x is None for x in [self.warp_image1, self.warp_image2]):
      self.warp_perspective()

    return ((self.warp_image1.astype(np.float) / 2.0) +
            (self.warp_image2.astype(np.float) / 2.0))
