from pathlib import Path

import utils

import numpy as np
from skimage import io

import tools.imagetools as imt
from tools.camera_calibration import CameraCalibration

if __name__ == '__main__':
  params_dir = utils.DATA_DIR.joinpath('FLIR_E95_Calibration')
  params_npz = params_dir.joinpath('parameters.npz')
  params_yml = params_dir.joinpath('parameters.yaml')
  assert params_npz.exists()
  assert params_yml.exists()

  cali_npz = CameraCalibration(params_npz)
  cali_yml = CameraCalibration(params_yml)

  assert cali_npz.imsize == cali_yml.imsize
  assert np.all(np.isclose(cali_npz.matrix, cali_yml.matrix))
  assert np.all(np.isclose(cali_npz.dist_coeff, cali_yml.dist_coeff))

  # image = cv.imread(
  #     dirs.ROOT_DIR.joinpath('data/FLIR/IR_2020-10-19_0017.jpg').as_posix())
  #
  # imt.show_images_mpl([cali_npz.mask(), cali_npz.calibrate(image)])

  fnames = ['IR_2020-11-10_0056.png', 'IR_2020-11-10_0061.png']
  data_dir = Path(r'D:\01. 업무\02. 용역\2020 [KICT] 열화상 파노라마\02. 연구\03. 결과'
                  r'\2020.11.10_건기연 직접 촬영_체스보드 보정\원본 (열화상 전처리 후)')
  fnames = [data_dir.joinpath(x) for x in fnames]
  for fname in fnames:
    assert fname.exists(), fname
  # images = [cv.imread(x.as_posix(), flags=cv.IMREAD_GRAYSCALE) for x in fnames]
  images = [io.imread(x.as_posix()) for x in fnames]

  grid = np.zeros_like(images[0], dtype='uint8')
  gap = 35
  for row in range(0, images[0].shape[0], gap):
    grid[row] = 255
  for col in range(0, images[0].shape[1], gap):
    grid[:, col] = 255
  grid[grid.shape[0] - 1] = 255
  grid[:, (grid.shape[1] - 1)] = 255

  # imt.show_images_mpl([cali_npz.calibrate(x) for x in images + [grid]])

  fig, axes = imt.show_images_mpl(
      [cali_npz.calibrate(x) for x in [grid, images[1]]], show=False)
  fig.savefig(utils.PRJ_DIR.joinpath('result/calibration.png'), dpi=200)
