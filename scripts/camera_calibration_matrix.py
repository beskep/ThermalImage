from pathlib import Path

import utils

import cv2 as cv
import numpy as np
import yaml
from rich.progress import track

import tools.imagetools as imt


def main(fnames, save_dir, pattern_size=(3, 3), show=False):
  save_dir = Path(save_dir).resolve()

  criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)
  objp = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
  objp[:, :2] = np.mgrid[0:pattern_size[0],
                         0:pattern_size[1]].T.reshape([-1, 2])

  obj_points = []
  img_points = []
  images = []
  imsize = None

  for fname in track(fnames):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if imsize is None:
      imsize = gray.shape[::-1]
    else:
      assert imsize == gray.shape[::-1]

    ret, corners = cv.findChessboardCorners(gray,
                                            patternSize=pattern_size,
                                            corners=None)
    if ret:
      obj_points.append(objp)

      corners2 = cv.cornerSubPix(gray,
                                 corners=corners,
                                 winSize=(11, 11),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
      img_points.append(corners)

      cv.drawChessboardCorners(img,
                               patternSize=pattern_size,
                               corners=corners2,
                               patternWasFound=ret)
      # cv.imshow(fname, img)
      # cv.waitKey(500)
      images.append(img)
      if show:
        imt.show_images_mpl([img])
      if save_dir is not None:
        path = save_dir.joinpath(
            Path(fname).with_suffix('.jpg').name).as_posix()
        imt.imsave(fname=path, arr=img)

  # cv.destroyAllWindows()

  if not obj_points:
    print('chessboard 추출 실패')
    return

  ret, mtx, dist_coeff, rvecs, tvecs = cv.calibrateCamera(
      objectPoints=obj_points,
      imagePoints=img_points,
      imageSize=imsize,
      cameraMatrix=None,
      distCoeffs=None)
  # imt.show_images_mpl(images)

  res_dict = {
      'image_size': list(imsize),
      'ret': ret,
      'matrix': mtx.tolist(),
      'dist_coeff': dist_coeff.tolist(),
      'rvecs': np.array(rvecs).tolist(),
      'tvecs': np.array(tvecs).tolist(),
  }
  with open(save_dir.joinpath('parameters.yaml'), 'w') as f:
    yaml.dump(res_dict, stream=f)

  np.savez(file=save_dir.joinpath('parameters'),
           imsize=imsize,
           matrix=mtx,
           dist_coeff=dist_coeff,
           rvecs=np.array(rvecs),
           tvecs=np.array(tvecs))

  if show:
    for fname in fnames:
      image = cv.imread(fname)

      new_mtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix=mtx,
                                                  distCoeffs=dist_coeff,
                                                  imageSize=imsize,
                                                  alpha=1,
                                                  newImgSize=imsize)

      dst = cv.undistort(image,
                         cameraMatrix=mtx,
                         distCoeffs=dist_coeff,
                         dst=None,
                         newCameraMatrix=new_mtx)

      imt.show_images_mpl([dst])


if __name__ == '__main__':
  # image_dir = Path(__file__).parents[2].joinpath('data/opencv_sample')
  # assert image_dir.exists()
  #
  # images = [x.as_posix() for x in image_dir.rglob('left*.jpg')]
  # assert len(images) == 13

  image_dir = utils.DATA_DIR.joinpath('FLIR_E95_Calibration')
  assert image_dir.exists()

  images = [x.as_posix() for x in image_dir.rglob('*.png')]

  main(images, save_dir=image_dir, pattern_size=(5, 3))
