import os

import cv2 as cv
import numpy as np
import yaml


class CameraCalibration:
  # TODO: matrix 추출하고 저장하는 코드도 추가하기
  #  (script/camera_calibration_matrix.py)

  def __init__(self, params_path):
    if not os.path.exists(params_path):
      raise FileNotFoundError(params_path)

    ext = os.path.splitext(params_path)[1]
    if ext == '.npz':
      npzfile = np.load(params_path)

      self._imsize = tuple(npzfile['imsize'])
      self._matrix = npzfile['matrix']
      self._dist_coeff = npzfile['dist_coeff']
    elif ext in ['.yaml', '.yml']:
      with open(params_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

      self._imsize = tuple(params['image_size'])
      self._matrix = np.array(params['matrix'])
      self._dist_coeff = params['dist_coeff']
    else:
      raise ValueError

  @property
  def imsize(self):
    return self._imsize

  @property
  def matrix(self):
    return self._matrix

  @property
  def dist_coeff(self):
    return self._dist_coeff

  def calibrate(self, image):
    new_matrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix=self._matrix,
                                                   distCoeffs=self._dist_coeff,
                                                   imageSize=self._imsize,
                                                   alpha=1,
                                                   newImgSize=self._imsize)

    calibrated = cv.undistort(image,
                              cameraMatrix=self._matrix,
                              distCoeffs=self._dist_coeff,
                              dst=None,
                              newCameraMatrix=new_matrix)

    return calibrated

  def mask(self):
    blank = np.full(shape=self._imsize[::-1], fill_value=255, dtype=np.uint8)
    mask = self.calibrate(blank)

    return mask
