from pathlib import Path

import utils

import cv2 as cv
import numpy as np
import skimage.exposure
import skimage.io

import flir


def main(path, save_dir: Path, fname: str):
  image = flir.FlirExtractor().extract_ir(path=path)

  mask = (image < -30.0)

  image_mask = image.copy()
  # image_mask = skimage.exposure.rescale_intensity(image=image,
  #                                                 out_range='uint8')
  image_mask[mask] = np.min(image)
  # image_mask[mask] = 0

  image_histeq = skimage.exposure.equalize_hist(image=image_mask)
  image_uint = skimage.exposure.rescale_intensity(image=image_histeq,
                                                  out_range='uint8')
  image_bilateral = cv.bilateralFilter(image_uint,
                                       d=-1,
                                       sigmaColor=20,
                                       sigmaSpace=10)

  # imt.show_images_mpl([image, image_histeq, image_bilateral])

  save_path = save_dir.joinpath(fname).as_posix()

  skimage.io.imsave(fname=save_path + '_1_original.png', arr=image)
  skimage.io.imsave(fname=save_path + '_2_mask.png', arr=image_mask)
  skimage.io.imsave(fname=save_path + '_3_histeq.png', arr=image_uint)
  skimage.io.imsave(fname=save_path + '_4_bilateral.png', arr=image_bilateral)


if __name__ == "__main__":
  path = Path(r'D:\repo\ThermalImage\Panorama\KICT\2020-11-10\original'
              r'\WaterResourceBldgPanTiltTripod\IR_2020-11-10_0142.jpg')
  save_dir = utils.PRJ_DIR.joinpath('report', 'preprocess')
  if not save_dir.exists():
    save_dir.mkdir()

  main(path=path, save_dir=save_dir, fname=path.stem)
