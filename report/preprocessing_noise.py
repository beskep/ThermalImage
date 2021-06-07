from pathlib import Path

import utils

import seaborn as sns
import skimage.io
from rich.progress import track
from skimage import exposure
from skimage.restoration import denoise_bilateral

import tools.cmap
import tools.imagetools as imt
import tools.ivimages as ivi

if __name__ == '__main__':
  sns.set_context('paper')

  data_dir = Path(r'D:\repo\ThermalImage\Panorama\KICT\2020-11-10'
                  r'\extracted\MainBldgBackLoc1PanTiltTripod\ir\original')
  res_dir = utils.ROOT_DIR.joinpath('result/preprocessing_noise')
  if not res_dir.exists():
    res_dir.mkdir()

  cmap_path = utils.ROOT_DIR.joinpath('data/cmap/iron_colormap_rgb.txt')
  cmap = tools.cmap.Colormap.from_uint8_text(cmap_path)

  loader = ivi.ImageLoader(img_dir=data_dir, img_ext='npy')

  for file in track(loader.files):
    # 대상 정함
    if '018' not in file.as_posix():
      continue

    stem = file.stem

    arr = loader.read(file)

    arr_mask = arr.copy()
    arr_mask[arr < -20.0] = -20

    equalized = exposure.equalize_hist(arr)
    equalized_mask = exposure.equalize_hist(arr_mask)

    bilateral = denoise_bilateral(equalized, sigma_color=0.05, sigma_spatial=10)
    bilateral_mask = denoise_bilateral(equalized_mask,
                                       sigma_color=0.05,
                                       sigma_spatial=10)

    # iron cmap
    # image = imt.array_to_image(cmap(imt.normalize_array(arr)))
    # eq_image = imt.array_to_image(cmap(imt.normalize_array(equalized)))

    skimage.io.imsave(
        fname=res_dir.joinpath(stem + '_original.png').as_posix(),
        arr=exposure.rescale_intensity(arr, out_range='uint8'),
    )
    skimage.io.imsave(
        fname=res_dir.joinpath(stem + '_equalized.png').as_posix(),
        arr=exposure.rescale_intensity(equalized, out_range='uint8'),
    )
    skimage.io.imsave(
        fname=res_dir.joinpath(stem + '_equalized_mask.png').as_posix(),
        arr=exposure.rescale_intensity(equalized_mask, out_range='uint8'),
    )
    skimage.io.imsave(
        fname=res_dir.joinpath(stem + '_bilateral.png').as_posix(),
        arr=exposure.rescale_intensity(bilateral, out_range='uint8'),
    )
    skimage.io.imsave(
        fname=res_dir.joinpath(stem + '_bilateral_mask.png').as_posix(),
        arr=exposure.rescale_intensity(bilateral_mask, out_range='uint8'),
    )
