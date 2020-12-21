from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.exposure
import skimage.io
from rich.progress import track

import utils
import tools.cmap
import tools.imagetools as imt
import tools.ivimages as ivi

if __name__ == '__main__':
  sns.set_context('paper')

  data_dir = Path(r'D:\repo\ThermalImage\Panorama\KICT\2020-11-10'
                  r'\extracted\MainBldgFrontPanTiltTripod\ir\original')
  res_dir = dirs.ROOT_DIR.joinpath('result/preprocessing_all')
  if not res_dir.exists():
    res_dir.mkdir()

  cmap_path = dirs.ROOT_DIR.joinpath('data/iron_colormap_rgb.txt')
  cmap = tools.cmap.Colormap.from_uint8_text(cmap_path)

  loader = ivi.ImageLoader(img_dir=data_dir, img_ext='npy')

  for file in track(loader.files):
    # if '018' not in file.as_posix():
    #   continue

    stem = file.stem

    arr = loader.read(file)
    image = imt.array_to_image(arr)
    equalized = skimage.exposure.equalize_hist(arr)
    equalized_image = imt.array_to_image(equalized)
    bilateral_image = cv.bilateralFilter(equalized_image,
                                         d=-1,
                                         sigmaColor=10,
                                         sigmaSpace=10)

    # iron cmap
    # image = imt.array_to_image(cmap(imt.normalize_array(arr)))
    # eq_image = imt.array_to_image(cmap(imt.normalize_array(equalized)))

    skimage.io.imsave(
        fname=res_dir.joinpath(stem + '_original.png').as_posix(),
        arr=image,
    )
    skimage.io.imsave(
        fname=res_dir.joinpath(stem + '_equalized.png').as_posix(),
        arr=equalized_image,
    )
    skimage.io.imsave(
        fname=res_dir.joinpath(stem + '_bilateral.png').as_posix(),
        arr=bilateral_image,
    )

    fig, axes = imt.show_images_mpl(
        images=[image, equalized_image, bilateral_image], show=False)
    fig.savefig(res_dir.joinpath(stem + '_all.png'), dpi=200)
    plt.close(fig)
