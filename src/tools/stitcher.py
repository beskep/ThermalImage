from typing import List, Union

import cv2 as cv
import numpy as np
from loguru import logger
from skimage.exposure import rescale_intensity

_AVAILABLE_WARPER = (
    'affine',
    'compressedPlaneA1.5B1',
    'compressedPlaneA2B1',
    'compressedPlanePortraitA1.5B1',
    'compressedPlanePortraitA2B1',
    'cylindrical',
    'fisheye',
    'mercator',
    'paniniA1.5B1',
    'paniniA2B1',
    'paniniPortraitA1.5B1',
    'paniniPortraitA2B1',
    'plane',
    'spherical',
    'stereographic',
    'transverseMercator',
)


class StitchingImages:

  def __init__(self, arrays: list, preprocess=None):
    """Stitching 대상 이미지

    Parameters
    ----------
    arrays : list of np.ndarray
        원본 이미지 dtype 상관 없음
    preprocess : callable, optional
        preprocessing function, by default None
        (image, mask)를 반환해야 함

    Raises
    ------
    ValueError
        preprocess가 None이나 callable이 아닌 경우
    """
    if (preprocess is not None) and not callable(preprocess):
      raise ValueError

    self.arrays = arrays
    self._preprocess = preprocess

    minmax = np.array([[np.min(x), np.max(x)] for x in arrays])
    self._in_range = (np.min(minmax[:, 0]), np.max(minmax[:, 1]))

  @property
  def arrays(self):
    """이미지 원본"""
    return self._arrays

  @arrays.setter
  def arrays(self, value):
    ndim = value[0].ndim
    if not all(x.ndim == ndim for x in value):
      raise ValueError('영상의 채널 수가 동일하지 않음')

    self._arrays = value
    self._arrays_count = len(value)
    self._ndim = ndim

  @property
  def count(self):
    return self._arrays_count

  @property
  def ndim(self):
    return self._ndim

  def set_preprocess(self, fn):
    self._preprocess = fn

  def select_images(self, indices):
    self.arrays = [self.arrays[x] for x in indices]

  def scale(self, image, out_range):
    res = rescale_intensity(image=image,
                            in_range=self._in_range,
                            out_range=out_range)
    return res

  def unscale(self, image, out_range):
    res = rescale_intensity(image=image,
                            in_range=out_range,
                            out_range=self._in_range)
    return res

  def preprocess(self):
    if self._preprocess is None:
      images = self.arrays
      masks = [None for _ in range(self.count)]
    else:
      prep = [self._preprocess(x.copy()) for x in self._arrays]
      images = [x[0] for x in prep]
      masks = [x[1] for x in prep]

    if any(x.dtype != np.uint8 for x in images):
      images = [self.scale(x, out_range=np.uint8) for x in images]

    return images, masks


class Stitcher:

  def __init__(self,
               mode='pano',
               features_finder=None,
               compose_scale=1.0,
               work_scale=1.0,
               try_cuda=False):
    self._mode = None
    self._estimator = None
    self._features_finder = None
    self._features_matcher = None
    self._bundle_adjuster = None
    self._refine_mask = None
    self._warper = None
    self._warper_type = None
    self._blend_type = 'no'
    self._blend_strength = 0.05

    self._compose_scale = compose_scale
    self._work_scale = work_scale
    self._compose_work_aspect = compose_scale / work_scale
    self._try_cuda = try_cuda

    self.features_finder = features_finder
    self.set_mode(mode.lower())

  @property
  def estimator(self) -> cv.detail_Estimator:
    return self._estimator

  @estimator.setter
  def estimator(self, value: cv.detail_Estimator):
    if not isinstance(value, cv.detail_Estimator):
      raise TypeError

    self._estimator = value

  @property
  def features_finder(self) -> cv.Feature2D:
    if self._features_finder is None:
      self._features_finder = cv.ORB_create()

    return self._features_finder

  @features_finder.setter
  def features_finder(self, value: cv.Feature2D):
    self._features_finder = value

  @property
  def features_matcher(self) -> cv.detail_FeaturesMatcher:
    return self._features_matcher

  @features_matcher.setter
  def features_matcher(self, value: cv.detail_FeaturesMatcher):
    self._features_matcher = value

  @property
  def bundle_adjuster(self) -> cv.detail_BundleAdjusterBase:
    return self._bundle_adjuster

  @bundle_adjuster.setter
  def bundle_adjuster(self, value: cv.detail_BundleAdjusterBase):
    self._bundle_adjuster = value

  @property
  def refine_mask(self) -> np.ndarray:
    if self._refine_mask is None:
      self.set_bundle_adjuster_refine_mask()

    return self._refine_mask

  @property
  def warper(self) -> Union[None, cv.PyRotationWarper]:
    return self._warper

  @property
  def warper_type(self) -> str:
    return self._warper_type

  @warper_type.setter
  def warper_type(self, value: str):
    if value not in _AVAILABLE_WARPER:
      raise ValueError(value)

    self._warper_type = value

  @property
  def blend_type(self) -> str:
    return self._blend_type

  @blend_type.setter
  def blend_type(self, value: str):
    if value.lower() not in ('multiband', 'feather', 'no'):
      raise ValueError

    self._blend_type = value.lower()

  @property
  def blend_strength(self):
    return self._blend_strength

  @blend_strength.setter
  def blend_strength(self, value: float):
    if not (0.0 <= value <= 1.0):
      raise ValueError(
          'blender strength not in [0, 1], value: {}'.format(value))

    self._blend_strength = value

  def set_features_matcher(self,
                           matcher='affine',
                           confidence=None,
                           range_width=-1):
    """
    Parameters
    ----------
    matcher: str
        matcher type
    confidence: float, optional
        Confidence for feature matching step.
        The default is 0.3 for ORB and 0.65 for other feature types.
    range_width
        uses range_width to limit number of images to match with

    Returns
    -------
    None
    """
    if confidence is None:
      if (self._features_matcher is None or
          isinstance(self._features_matcher, cv.ORB)):
        confidence = 0.30
      else:
        confidence = 0.65

    if matcher == 'affine':
      matcher = cv.detail_AffineBestOf2NearestMatcher(
          full_affine=False,
          try_use_gpu=self._try_cuda,
          match_conf=confidence,
      )
    elif range_width == -1:
      matcher = cv.detail.BestOf2NearestMatcher_create(
          try_use_gpu=self._try_cuda,
          match_conf=confidence,
      )
    else:
      matcher = cv.detail_BestOf2NearestRangeMatcher(
          range_width=range_width,
          try_use_gpu=self._try_cuda,
          match_conf=confidence,
      )
    self._features_matcher = matcher

  def set_bundle_adjuster_refine_mask(self,
                                      fx=True,
                                      skew=True,
                                      ppx=True,
                                      aspect=True,
                                      ppy=True):
    """
    Set refinement mask for bundle adjustment
    """
    refine_mask = np.zeros([3, 3], dtype=np.uint8)

    masks = [fx, skew, ppx, aspect, ppy]
    rows = [0, 0, 0, 1, 1]
    cols = [0, 1, 2, 1, 2]
    for mask, row, col in zip(masks, rows, cols):
      if mask:
        refine_mask[row, col] = 1

    self._refine_mask = refine_mask

  def set_warper(self, scale):
    self._warper = cv.PyRotationWarper(type=self.warper_type, scale=scale)

  @staticmethod
  def available_warper_types():
    return _AVAILABLE_WARPER[:]

  def set_mode(self, mode: str):
    if mode.startswith('pano'):
      self.estimator = cv.detail_HomographyBasedEstimator()
      self.set_features_matcher('pano')
      self.bundle_adjuster = cv.detail_BundleAdjusterRay()
      self.warper_type = 'spherical'
    elif mode == 'scan':
      self.estimator = cv.detail_AffineBasedEstimator()
      self.set_features_matcher('affine')
      self.bundle_adjuster = cv.detail_BundleAdjusterAffinePartial()
      self.warper_type = 'affine'
    else:
      raise ValueError(mode)

    self._mode = mode

  def find_features(self, image, mask):
    if self.features_finder is None:
      raise ValueError('features_finder가 지정되지 않음')

    features = cv.detail.computeImageFeatures2(
        featuresFinder=self.features_finder, image=image, mask=mask)

    return features

  def stitch(self,
             images: StitchingImages,
             masks: list = None,
             image_names: list = None):
    if image_names is None:
      image_names = ['image {}'.format(x + 1) for x in range(images.count)]

    prep_images, prep_masks = images.preprocess()

    if masks is None:
      masks = prep_masks
    else:
      for mask, pmask in zip(masks, prep_masks):
        mask = np.logical_and(mask, pmask)

    # camera matrix 계산
    cameras, indices, matches_graph = self.calculate_camera_matrix(
        images=prep_images, image_names=image_names)

    if len(indices) != len(prep_images):
      images.select_images(indices=indices)
      logger.debug('Stitching에 필요 없는 이미지 제거 (indices: {})', indices)

    # warp
    warped_images, warped_masks, rois = self.warp_images(images=images.arrays,
                                                         masks=masks,
                                                         cameras=cameras)

    # stitch
    scaled_images = [images.scale(x, out_range='int16') for x in warped_images]
    stitched_image, stitched_mask = self.blend(images=scaled_images,
                                               masks=warped_masks,
                                               rois=rois)
    stitched_image[np.logical_not(stitched_mask)] = np.median(stitched_image)

    if images.ndim == 2:
      # 열화상인 경우 첫 번째 채널만 추출
      stitched_image = stitched_image[:, :, 0]

    unscaled_image = images.unscale(image=stitched_image, out_range='int16')

    return unscaled_image, stitched_mask, matches_graph, indices

  def calculate_camera_matrix(self, images: List[np.ndarray], image_names):
    logger.debug('feature finding and matching')
    # note: find_features에는 마스크 적용하지 않음
    # (~mask에 0 대입한 영상으로 feature 탐색)
    features = [self.find_features(image=image, mask=None) for image in images]

    pairwise_matches = self.features_matcher.apply2(features=features)
    self.features_matcher.collectGarbage()

    indices = cv.detail.leaveBiggestComponent(features=features,
                                              pairwise_matches=pairwise_matches,
                                              conf_threshold=0.3)
    if len(indices) < 2:
      raise ValueError('Need more images (valid images are less than two)')

    indices = [x[0] for x in indices]
    images = [images[x] for x in indices]

    logger.debug('matches graph')
    matches_graph = cv.detail.matchesGraphAsString(
        pathes=image_names,
        pairwise_matches=pairwise_matches,
        conf_threshold=1.0)

    logger.debug('estimator')
    estimate_status, cameras = self.estimator.apply(
        features=features, pairwise_matches=pairwise_matches, cameras=None)
    if not estimate_status:
      raise ValueError('Homography estimation failed')

    logger.debug('bundle adjuster')
    self.bundle_adjuster.setConfThresh(1)
    self.bundle_adjuster.setRefinementMask(self.refine_mask)

    for cam in cameras:
      cam.R = cam.R.astype(np.float32)

    adjuster_status, cameras = self.bundle_adjuster.apply(
        features=features, pairwise_matches=pairwise_matches, cameras=cameras)
    if not adjuster_status:
      raise ValueError('Camera parameters adjusting failed')

    # logger.debug('wave correction')
    # if self.flag_wave_correction:
    #   # FIXME: wave correction -> cv assert 에러
    #   rmats = [np.copy(x.R) for x in cameras]
    #   rmats_correct = [
    #       cv.detail.waveCorrect(x, cv.detail.WAVE_CORRECT_HORIZ) for x in rmats
    #   ]
    #   for rmat, cam in zip(rmats_correct, cameras):
    #     cam.R = rmat

    return cameras, indices, matches_graph

  def warp_image(self, image, mask, camera):
    if self._compose_work_aspect != 1.0:
      camera.focal *= self._compose_work_aspect
      camera.ppx *= self._compose_work_aspect
      camera.ppy *= self._compose_work_aspect

    size = (int(image.shape[1] * self._compose_scale),
            int(image.shape[0] * self._compose_scale))
    kmat = camera.K().astype(np.float32)
    roi = self.warper.warpRoi(src_size=size, K=kmat, R=camera.R)

    if abs(self._compose_scale - 1) > 0.1:
      # float32에도 돌아가나?
      img = cv.resize(src=image,
                      dsize=None,
                      fx=self._compose_scale,
                      fy=self._compose_scale,
                      interpolation=cv.INTER_LINEAR_EXACT)
      if mask is not None:
        mask = cv.resize(src=mask,
                         dsize=None,
                         fx=self._compose_scale,
                         fy=self._compose_scale,
                         interpolation=cv.INTER_LINEAR_EXACT)
    else:
      img = image

    kmat = camera.K().astype(np.float32)
    rmat = camera.R
    # note: (roi[0], roi[1]) == corner
    corner, warped_image = self.warper.warp(src=img,
                                            K=kmat,
                                            R=rmat,
                                            interp_mode=cv.INTER_LINEAR,
                                            border_mode=cv.BORDER_CONSTANT)

    if mask is None:
      mask = np.ones(shape=img.shape[:2], dtype=np.uint8)

    _, warped_mask = self.warper.warp(src=mask,
                                      K=kmat,
                                      R=rmat,
                                      interp_mode=cv.INTER_LINEAR,
                                      border_mode=cv.BORDER_CONSTANT)

    return warped_image, warped_mask, roi

  def warp_images(self, images, masks, cameras):
    self.set_warper(scale=np.median([x.focal for x in cameras]))

    warped_images = []
    warped_masks = []
    rois = []
    for idx, (image, mask, camera) in enumerate(zip(images, masks, cameras)):
      try:
        wi, wm, roi = self.warp_image(image, mask, camera)
        warped_images.append(wi)
        warped_masks.append(wm)
        rois.append(roi)
      except cv.error:
        logger.error(f'{idx+1}번 영상의 과도한 변형으로 인한 오류 발생. 해당 영상을 제외합니다.')

    return warped_images, warped_masks, rois

  def blend(self, images, masks, rois):
    """영상 간 밝기 조정

    Parameters
    ----------
    images : list of CV_16S images
        int16 형식만 입력받음
        1채널일 경우 자동으로 3채널 이미지로 변환
    masks : list of np.ndarray
        대상 영역 마스크 목록
    rois : list of tuple
        ROI 리스트

    Returns
    -------
    (np.ndarray, np.ndarray)
        (stitch 된 영상, 마스크)
    """
    corners = [(x[0], x[1]) for x in rois]
    dst_size = cv.detail.resultRoi(corners=corners, images=images)

    # blend width 계산, blender type 결정
    blend_width = (np.sqrt(dst_size[2] * dst_size[3]) * self._blend_strength)
    blend_type = 'no' if blend_width < 1 else self.blend_type
    logger.debug('blend type: {}', blend_type)

    # blender 생성
    if blend_type == 'no':
      blender = cv.detail.Blender_createDefault(type=cv.detail.Blender_NO,
                                                try_gpu=self._try_cuda)
    elif blend_type == 'multiband':
      blender = cv.detail_MultiBandBlender()
      bands_count = (np.log2(blend_width) - 1.0).astype(np.int)
      blender.setNumBands(bands_count)
    elif blend_type == 'feather':
      blender = cv.detail_FeatherBlender()
      blender.setSharpness(1.0 / blend_width)
    else:
      raise ValueError

    # blend
    blender.prepare(dst_size)
    for image, mask, corner in zip(images, masks, corners):
      if image.ndim == 2:
        image = np.repeat(image[:, :, np.newaxis], repeats=3, axis=2)

      blender.feed(img=image, mask=mask, tl=corner)

    stitched_image, stitched_mask = blender.blend(dst=None, dst_mask=None)

    return stitched_image, stitched_mask
