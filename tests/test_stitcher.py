from pathlib import Path

import utils

import cv2 as cv
import numpy as np
import skimage.exposure
import skimage.io

import tools.ivimages as ivi
import tools.stitcher


def test_stitcher():
  img_dir = Path(r'D:\test\panorama\20210204\지빠귀')
  loader = ivi.ImageLoader(img_dir=img_dir, img_ext='npy')
  arrays = [loader.read(x).astype('float32') for x in loader.files]

  def _prep(image):
    mask = (image > -30.0).astype(np.uint8)
    # mask = None
    image = skimage.exposure.equalize_hist(image)
    image = skimage.exposure.rescale_intensity(image=image, out_range='uint8')
    image = cv.bilateralFilter(image, d=-1, sigmaColor=20, sigmaSpace=10)

    return image, mask

  stitching_images = tools.stitcher.StitchingImages(arrays=arrays,
                                                    preprocess=_prep)

  stitcher = tools.stitcher.Stitcher(
      mode='pano',
      # features_finder=cv.ORB_create(),
      # compose_scale=0.75,
      # work_scale=1.0,
  )
  stitcher.warper_type = 'plane'

  image, mask, graph, indices = stitcher.stitch(images=stitching_images,
                                                masks=None,
                                                image_names=None)

  # print(graph)
  # imt.show_images_mpl([image, mask])

  assert isinstance(image, np.ndarray)
  assert isinstance(mask, np.ndarray)

  image_uint = skimage.exposure.rescale_intensity(image, out_range='uint16')
  # image_uint[mask] = 0
  skimage.io.imsave(fname=img_dir.joinpath('res.jpg').as_posix(),
                    arr=image_uint)


if __name__ == "__main__":
  # pytest.main(['-v', '-k', 'test_stitcher'])
  test_stitcher()
