from typing import Union

import cv2 as cv
import numpy as np

import tools.imagetools as imt

AVAILABLE_WARPER = (
    'spherical',
    'plane',
    'affine',
    'cylindrical',
    'fisheye',
    'stereographic',
    'compressedPlaneA2B1',
    'compressedPlaneA1.5B1',
    'compressedPlanePortraitA2B1',
    'compressedPlanePortraitA1.5B1',
    'paniniA2B1',
    'paniniA1.5B1',
    'paniniPortraitA2B1',
    'paniniPortraitA1.5B1',
    'mercator',
    'transverseMercator',
)
AVAILABLE_SEAM_FINDER = (
    'gc_color',
    'gc_colorgrad',
    'dp_color',
    'dp_colorgrad',
    'voronoi',
    'no',
)


class Stitcher:

  def __init__(self, mode='pano', try_cuda=False):
    self._mode = None
    self._estimator = None
    self._flag_wave_correction = None
    self._wave_correction_type = None
    self._features_matcher = None
    self._features_finder = None
    self._bundle_adjuster = None
    self._warper = None
    self._warper_type = None
    self._exposure_compensator = None
    self._seam_finder = None
    self._confidence_threshold = 1.0
    self._refine_mask = None
    self._blend_type = 'multiband'
    self._blend_strength = 5

    self._cuda = try_cuda

    if mode[:4] == 'pano':
      self.set_mode('pano')
    elif mode[:4] == 'scan':
      self.set_mode('scan')
    else:
      msg = 'mode는 scan, pano 중 하나여야 함 (입력: {})'.format(mode)
      raise ValueError(msg)

  @property
  def estimator(self) -> cv.detail_Estimator:
    return self._estimator

  @estimator.setter
  def estimator(self, value: cv.detail_Estimator):
    self._estimator = value

  @property
  def flag_wave_correction(self) -> bool:
    return self._flag_wave_correction

  @flag_wave_correction.setter
  def flag_wave_correction(self, value=False):
    self._flag_wave_correction = value

  @property
  def wave_correction_type(self) -> str:
    # TODO: wave correction 적용
    return self._wave_correction_type

  @wave_correction_type.setter
  def wave_correction_type(self, value: str):
    if value.lower() not in ('horiz', 'no', 'vert'):
      raise ValueError
    self._wave_correction_type = value.lower()

  @property
  def features_matcher(self) -> cv.detail_FeaturesMatcher:
    return self._features_matcher

  @features_matcher.setter
  def features_matcher(self, value: cv.detail_FeaturesMatcher):
    self._features_matcher = value

  def set_features_matcher(self,
                           matcher='affine',
                           confidence=None,
                           range_width=-1,
                           try_cuda=False):
    """
    Parameters
    ----------
    matcher
        matcher
    confidence
        Confidence for feature matching step.
        The default is 0.3 for ORB and 0.65 for other feature types.
    range_width
        uses range_width to limit number of images to match with
    try_cuda
        try CUDA

    Returns
    -------
    None
    """
    if confidence is None:
      if (self._features_matcher is None or
          isinstance(self._features_matcher, cv.ORB)):
        confidence = 0.3
      else:
        confidence = 0.65

    if matcher == 'affine':
      matcher = cv.detail_AffineBestOf2NearestMatcher(False, try_cuda,
                                                      confidence)
    elif range_width == -1:
      matcher = cv.detail.BestOf2NearestMatcher_create(try_cuda, confidence)
    else:
      matcher = cv.detail_BestOf2NearestRangeMatcher(range_width, try_cuda,
                                                     confidence)
    self._features_matcher = matcher

  @property
  def features_finder(self) -> Union[None, cv.Feature2D]:
    return self._features_finder

  @features_finder.setter
  def features_finder(self, value: cv.Feature2D):
    self._features_finder = value

  @property
  def confidence_threshold(self) -> float:
    return self._confidence_threshold

  @confidence_threshold.setter
  def confidence_threshold(self, value: float):
    self._confidence_threshold = value

  @property
  def bundle_adjuster(self) -> cv.detail_BundleAdjusterBase:
    return self._bundle_adjuster

  @bundle_adjuster.setter
  def bundle_adjuster(self, value: cv.detail_BundleAdjusterBase):
    self._bundle_adjuster = value

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
    if fx == 'x':
      refine_mask[0, 0] = 1
    if skew == 'x':
      refine_mask[0, 1] = 1
    if ppx == 'x':
      refine_mask[0, 2] = 1
    if aspect == 'x':
      refine_mask[1, 1] = 1
    if ppy == 'x':
      refine_mask[1, 2] = 1

    self._refine_mask = refine_mask

  @property
  def refine_mask(self) -> np.ndarray:
    if self._refine_mask is None:
      self.set_bundle_adjuster_refine_mask()

    return self._refine_mask

  @property
  def warper_type(self) -> str:
    return self._warper_type

  @warper_type.setter
  def warper_type(self, value: str):
    if value not in AVAILABLE_WARPER:
      raise ValueError(value)

    self._warper_type = value

  @property
  def warper(self) -> Union[None, cv.PyRotationWarper]:
    return self._warper

  def set_warper(self, scale, warp_type=None):
    if warp_type is not None:
      self.warper_type = warp_type

    self._warper = cv.PyRotationWarper(type=self.warper_type, scale=scale)

  @property
  def exposure_compensator(self) -> Union[None, cv.detail_ExposureCompensator]:
    return self._exposure_compensator

  @exposure_compensator.setter
  def exposure_compensator(self, value: Union[None,
                                              cv.detail_ExposureCompensator]):
    self._exposure_compensator = value

  @property
  def seam_finder(self) -> Union[None, cv.detail_SeamFinder]:
    return self._seam_finder

  @seam_finder.setter
  def seam_finder(self, seam_finder: str):
    seam_finder = seam_finder.lower()
    if seam_finder not in AVAILABLE_SEAM_FINDER:
      raise ValueError('{} is not available'.format(seam_finder))

    if seam_finder == 'no':
      sf = None
    elif seam_finder == 'gc_color':
      sf = cv.detail_GraphCutSeamFinder('COST_COLOR')
    elif seam_finder == 'gc_colorgrad':
      sf = cv.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
    elif seam_finder == 'dp_color':
      sf = cv.detail_DpSeamFinder('COLOR')
    elif seam_finder == 'dp_colorgrad':
      sf = cv.detail_DpSeamFinder('COLOR_GRAD')
    elif seam_finder == 'voronoi':
      sf = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM)
    else:
      raise ValueError(seam_finder)

    self._seam_finder = sf

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
  def blend_strength(self, value):
    if not (0 <= value <= 100):
      raise ValueError

    self._blend_strength = value

  def set_mode(self, mode):
    if mode == 'pano':
      self.estimator = cv.detail_HomographyBasedEstimator()
      self.flag_wave_correction = True
      self.set_features_matcher('pano', try_cuda=self._cuda)
      self.bundle_adjuster = cv.detail_BundleAdjusterRay()
      self.warper_type = 'spherical'
      self.exposure_compensator = cv.detail_BlocksGainCompensator()
    elif mode == 'scan':
      self.estimator = cv.detail_AffineBasedEstimator()
      self.flag_wave_correction = False
      self.set_features_matcher('affine', try_cuda=self._cuda)
      self.bundle_adjuster = cv.detail_BundleAdjusterAffinePartial()
      self.warper_type = 'affine'
      # self.exposure_compensator = cv.detail_NoExposureCompensator()
      self.exposure_compensator = None
    else:
      raise ValueError(mode)

    self._mode = mode

  def stitch(self,
             images: list,
             image_names: list = None,
             masks=None,
             graph_path=None):
    if masks is None:
      masks = [None] * len(images)
      # FIXME: mask만 입력하면 정확도가 급감함
    if image_names is None:
      image_names = ['image {}'.format(x + 1) for x in range(len(images))]
    elif len(images) != len(image_names):
      raise ValueError
    # TODO: 결과 mask 대신 투명 layer로 표시하기

    features = []
    for image, mask in zip(images, masks):
      feature = cv.detail.computeImageFeatures2(
          featuresFinder=self.features_finder, image=image, mask=mask)
      features.append(feature)

    pairwise_matches = self.features_matcher.apply2(features=features)
    self.features_matcher.collectGarbage()

    if graph_path is not None:
      matches_graph = cv.detail.matchesGraphAsString(
          pathes=image_names,
          pairwise_matches=pairwise_matches,
          conf_threshold=self.confidence_threshold)
      with open(graph_path, 'w') as f:
        f.write(matches_graph)

    indices = cv.detail.leaveBiggestComponent(features=features,
                                              pairwise_matches=pairwise_matches,
                                              conf_threshold=0.3)

    subset_images = [images[indices[x, 0]] for x in range(len(indices))]
    # subset_names = [image_names[indices[x, 0]] for x in range(len(indices))]
    subset_full_sizes = [(x.shape[1], x.shape[0]) for x in subset_images]

    if len(subset_images) < 2:
      raise ValueError('Need more images (valid images are less than two)')

    estimate_status, cameras = self.estimator.apply(
        features=features, pairwise_matches=pairwise_matches, cameras=None)
    if not estimate_status:
      raise ValueError('Homography estimation failed')
    if len(cameras) != len(indices):
      # TODO: 왜지 ㅅㅂ
      raise ValueError

    for cam in cameras:
      cam.R = cam.R.astype(np.float32)

    self.bundle_adjuster.setConfThresh(1)
    self.bundle_adjuster.setRefinementMask(self.refine_mask)

    adjuster_status, cameras = self.bundle_adjuster.apply(
        features=features, pairwise_matches=pairwise_matches, cameras=cameras)
    if not adjuster_status:
      raise ValueError('Camera parameters adjusting failed')

    if self.flag_wave_correction:
      rmats = [np.copy(x.R) for x in cameras]
      rmats_correct = [
          cv.detail.waveCorrect(x, cv.detail.WAVE_CORRECT_HORIZ) for x in rmats
      ]
      for rmat, cam in zip(rmats_correct, cameras):
        cam.R = rmat

    masks = [
        cv.UMat(255 * np.ones(img.shape[:2], dtype=np.uint8))
        for img in subset_images
    ]

    focals = [x.focal for x in cameras]
    warped_image_scale = np.median(focals)
    self.set_warper(scale=warped_image_scale)

    corners = []
    warped_masks = []
    warped_images = []
    sizes = []
    for idx in range(len(subset_images)):
      kmat = cameras[idx].K().astype(np.float32)
      corner, image_wp = self.warper.warp(src=subset_images[idx],
                                          K=kmat,
                                          R=cameras[idx].R,
                                          interp_mode=cv.INTER_LINEAR,
                                          border_mode=cv.BORDER_REFLECT)
      corners.append(corner)
      warped_images.append(image_wp)
      sizes.append((image_wp.shape[1], image_wp.shape[0]))
      p, mask_wp = self.warper.warp(src=masks[idx],
                                    K=kmat,
                                    R=cameras[idx].R,
                                    interp_mode=cv.INTER_NEAREST,
                                    border_mode=cv.BORDER_CONSTANT)
      warped_masks.append(mask_wp)

    if self.exposure_compensator is not None:
      self.exposure_compensator.feed(corners=corners,
                                     images=warped_images,
                                     masks=warped_masks)

    if self.seam_finder is not None:
      images_warped_float = [x.astype(np.float32) for x in warped_images]
      self.seam_finder.find(src=images_warped_float,
                            corners=corners,
                            masks=warped_masks)

    compose_scale = 1
    corners = []
    sizes = []
    blender = None
    work_scale = 1.0
    for idx in range(len(warped_images)):
      compose_work_aspect = compose_scale / work_scale
      warped_image_scale *= compose_work_aspect

      assert len(cameras) == len(subset_images)
      for camera, size in zip(cameras, subset_full_sizes):
        if compose_work_aspect != 1.0:
          camera.focal *= compose_work_aspect
          camera.ppx *= compose_work_aspect
          camera.ppy *= compose_work_aspect

        sz = (size[0] * compose_scale, size[1] * compose_scale)
        kmat = camera.K().astype(np.float32)
        roi = self.warper.warpRoi(src_size=sz, K=kmat, R=camera.R)
        corners.append(roi[:2])
        sizes.append(roi[2:4])

      if abs(compose_scale - 1) > 0.1:
        img = cv.resize(src=subset_images[idx],
                        dsize=None,
                        fx=compose_scale,
                        fy=compose_scale,
                        interpolation=cv.INTER_LINEAR_EXACT)
      else:
        img = subset_images[idx]

      kmat = cameras[idx].K().astype(np.float32)
      rmat = cameras[idx].R
      corner, warped_img = self.warper.warp(
          src=img,
          K=kmat,
          R=rmat,
          interp_mode=cv.INTER_LINEAR,
          # border_mode=cv.BORDER_REFLECT
          border_mode=cv.BORDER_CONSTANT)
      # TODO: border_mode 옵션으로 결정?

      masks = 255 * np.ones(img.shape[:2], dtype=np.uint8)

      p, warped_mask = self.warper.warp(src=masks,
                                        K=kmat,
                                        R=rmat,
                                        interp_mode=cv.INTER_NEAREST,
                                        border_mode=cv.BORDER_CONSTANT)
      if self.exposure_compensator is not None:
        self.exposure_compensator.apply(index=idx,
                                        corner=corners[idx],
                                        image=warped_img,
                                        mask=warped_mask)

      warped_img_s = warped_img.astype(np.int16)
      dilated_mask = cv.dilate(src=warped_masks[idx], kernel=None)
      seam_mask = cv.resize(src=dilated_mask,
                            dsize=(warped_mask.shape[1], warped_mask.shape[0]),
                            interpolation=cv.INTER_LINEAR_EXACT)
      warped_mask = cv.bitwise_and(seam_mask, warped_mask)

      if blender is None:
        # None이 아니면??
        blender = cv.detail.Blender_createDefault(type=cv.detail.Blender_NO,
                                                  try_gpu=False)
        dst_sz = cv.detail.resultRoi(corners=corners, images=warped_images)

        blend_width = (np.sqrt(dst_sz[2] * dst_sz[3]) * self.blend_strength /
                       100)
        if blend_width < 1:
          blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
        elif self.blend_type == 'multiband':
          blender = cv.detail_MultiBandBlender()
          blender.setNumBands(
              (np.log(blend_width) / np.log(2.0) - 1.0).astype(np.int))
        elif self.blend_type == 'feather':
          blender = cv.detail_FeatherBlender()
          blender.setSharpness(1.0 / blend_width)

        blender.prepare(dst_sz)

      feed_img = cv.UMat(warped_img_s)
      blender.feed(img=feed_img, mask=warped_mask, tl=corners[idx])

    result_image, result_mask = blender.blend(dst=None, dst_mask=None)

    return result_image, result_mask


class IRStitcher(Stitcher):
  """전처리 전 이미지 저장, 이용
  테스트용 클래스
  """

  def __init__(self, mode='pano', try_cuda=False):
    super(IRStitcher, self).__init__(mode=mode, try_cuda=try_cuda)

    self._orig_images = None
    self._prep_images = None
    self._preprocess = None

  @property
  def preprocess(self):
    return self._preprocess

  @preprocess.setter
  def preprocess(self, value: list):
    if not callable(value):
      raise ValueError('{} is not callable'.format(value))

    self._preprocess = value

  def _find_features(self, image, mask):
    features = cv.detail.computeImageFeatures2(
        featuresFinder=self.features_finder, image=image, mask=mask)

    return features

  def stitch(self, images: list, image_names: list = None, masks=None):
    images_count = len(images)

    for idx in range(images_count):
      if images[idx].ndim == 2:
        images[idx] = np.repeat(images[idx][:, :, np.newaxis], 3, axis=2)
    assert all(image.ndim == 3 for image in images)

    if masks is None:
      masks = [None for x in range(images_count)]

    if image_names is None:
      image_names = ['image {}'.format(x + 1) for x in range(images_count)]

    prep_images = [self.preprocess(x) for x in images]

    features = [
        self._find_features(image, mask)
        for image, mask in zip(prep_images, masks)
    ]

    pairwise_matches = self.features_matcher.apply2(features=features)
    self.features_matcher.collectGarbage()

    indices = cv.detail.leaveBiggestComponent(features=features,
                                              pairwise_matches=pairwise_matches,
                                              conf_threshold=0.3)
    if len(indices) < 2:
      raise ValueError('Need more images (valid images are less than two)')

    indices_int = [indices[x, 0] for x in range(len(indices))]
    # unselected_images = [
    #     images[x] for x in range(images_count) if x not in indices_int
    # ]
    orig_images = [images[x] for x in indices_int]
    prep_images = [prep_images[x] for x in indices_int]
    full_sizes = [(x.shape[1], x.shape[0]) for x in prep_images]

    # estimator
    estimate_status, cameras = self.estimator.apply(
        features=features, pairwise_matches=pairwise_matches, cameras=None)
    if not estimate_status:
      raise ValueError('Homography estimation failed')

    for cam in cameras:
      cam.R = cam.R.astype(np.float32)

    # bundle adjuster
    self.bundle_adjuster.setConfThresh(1)
    self.bundle_adjuster.setRefinementMask(self.refine_mask)

    adjuster_status, cameras = self.bundle_adjuster.apply(
        features=features, pairwise_matches=pairwise_matches, cameras=cameras)
    if not adjuster_status:
      raise ValueError('Camera parameters adjusting failed')

    # wave correction
    if self.flag_wave_correction:
      # FIXME: error
      rmats = [np.copy(x.R) for x in cameras]
      rmats_correct = [
          cv.detail.waveCorrect(x, cv.detail.WAVE_CORRECT_HORIZ) for x in rmats
      ]
      for rmat, cam in zip(rmats_correct, cameras):
        cam.R = rmat

    # warp
    masks = [
        cv.UMat(255 * np.ones(img.shape[:2], dtype=np.uint8))
        for img in prep_images
    ]

    focals = [x.focal for x in cameras]
    warped_image_scale = np.median(focals)
    self.set_warper(scale=warped_image_scale)

    corners = []
    warped_masks = []
    warped_images = []
    sizes = []
    for idx in range(len(prep_images)):
      kmat = cameras[idx].K().astype(np.float32)
      corner, image_wp = self.warper.warp(src=orig_images[idx],
                                          K=kmat,
                                          R=cameras[idx].R,
                                          interp_mode=cv.INTER_LINEAR,
                                          border_mode=cv.BORDER_REFLECT)

      corners.append(corner)
      warped_images.append(image_wp)
      sizes.append((image_wp.shape[1], image_wp.shape[0]))
      _, mask_wp = self.warper.warp(src=masks[idx],
                                    K=kmat,
                                    R=cameras[idx].R,
                                    interp_mode=cv.INTER_NEAREST,
                                    border_mode=cv.BORDER_CONSTANT)
      warped_masks.append(mask_wp)

    # exposure_compensator
    if self.exposure_compensator is not None:
      self.exposure_compensator.feed(corners=corners,
                                     images=warped_images,
                                     masks=warped_masks)

    # seam finder
    if self.seam_finder is not None:
      images_warped_float = [x.astype(np.float32) for x in warped_images]
      self.seam_finder.find(src=images_warped_float,
                            corners=corners,
                            masks=warped_masks)

    # test
    # imt.show_images_mpl(warped_images)

    # 모르겠다 ㅅㅂ
    compose_scale = 1
    corners = []
    sizes = []
    blender = None
    work_scale = 1.0
    warped_images_test = []  # test
    for idx in range(len(warped_images)):
      compose_work_aspect = compose_scale / work_scale
      warped_image_scale *= compose_work_aspect

      assert len(cameras) == len(orig_images)
      for camera, size in zip(cameras, full_sizes):
        if compose_work_aspect != 1.0:
          camera.focal *= compose_work_aspect
          camera.ppx *= compose_work_aspect
          camera.ppy *= compose_work_aspect

        sz = (size[0] * compose_scale, size[1] * compose_scale)
        kmat = camera.K().astype(np.float32)
        roi = self.warper.warpRoi(src_size=sz, K=kmat, R=camera.R)
        corners.append(roi[:2])
        sizes.append(roi[2:4])

      if abs(compose_scale - 1) > 0.1:
        img = cv.resize(src=orig_images[idx],
                        dsize=None,
                        fx=compose_scale,
                        fy=compose_scale,
                        interpolation=cv.INTER_LINEAR_EXACT)
      else:
        img = orig_images[idx]

      kmat = cameras[idx].K().astype(np.float32)
      rmat = cameras[idx].R
      corner, warped_img = self.warper.warp(src=img,
                                            K=kmat,
                                            R=rmat,
                                            interp_mode=cv.INTER_LINEAR,
                                            border_mode=cv.BORDER_CONSTANT)

      # test
      warped_images_test.append(warped_img)

      masks = 255 * np.ones(img.shape[:2], dtype=np.uint8)

      _, warped_mask = self.warper.warp(src=masks,
                                        K=kmat,
                                        R=rmat,
                                        interp_mode=cv.INTER_NEAREST,
                                        border_mode=cv.BORDER_CONSTANT)
      if self.exposure_compensator is not None:
        self.exposure_compensator.apply(index=idx,
                                        corner=corners[idx],
                                        image=warped_img,
                                        mask=warped_mask)

      # test
      imt.show_images_mpl([warped_img, warped_mask])

      warped_img_s = warped_img.astype(np.int16)
      dilated_mask = cv.dilate(src=warped_masks[idx], kernel=None)
      seam_mask = cv.resize(src=dilated_mask,
                            dsize=(warped_mask.shape[1], warped_mask.shape[0]),
                            interpolation=cv.INTER_LINEAR_EXACT)
      warped_mask = cv.bitwise_and(seam_mask, warped_mask)

      if blender is None:
        # blender = cv.detail.Blender_createDefault(type=cv.detail.Blender_NO,
        #                                           try_gpu=False)
        dst_sz = cv.detail.resultRoi(corners=corners, images=warped_images)

        blend_width = (np.sqrt(dst_sz[2] * dst_sz[3]) * self.blend_strength /
                       100)
        # if blend_width < 1:
        #   self.blend_type = 'no'

        if self.blend_type == 'no':
          blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO,
                                                    try_gpu=False)
        elif self.blend_type == 'multiband':
          blender = cv.detail_MultiBandBlender()
          blender.setNumBands(
              (np.log(blend_width) / np.log(2.0) - 1.0).astype(np.int))
        elif self.blend_type == 'feather':
          blender = cv.detail_FeatherBlender()
          blender.setSharpness(1.0 / blend_width)

        blender.prepare(dst_sz)

      feed_img = cv.UMat(warped_img_s)
      blender.feed(img=feed_img, mask=warped_mask, tl=corners[idx])

    # return warped_images_test, None

    result_image, result_mask = blender.blend(dst=None, dst_mask=None)

    return result_image, result_mask
