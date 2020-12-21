from pathlib import Path

import seaborn as sns
import skimage.exposure
import skimage.io
from rich.progress import track

import utils
import tools.imagetools as imt
import tools.ivimages as ivi
import tools.cmap

if __name__ == '__main__':
  sns.set_context('paper')

  data_dir = Path(r'D:\repo\ThermalImage\Panorama\KICT\2020-11-10'
                  r'\extracted\MainBldgBackLoc1PanTiltTripod\ir\original')
  res_dir = dirs.ROOT_DIR.joinpath('result/preprocessing_noise')
  if not res_dir.exists():
    res_dir.mkdir()

  cmap_path = dirs.ROOT_DIR.joinpath('data/iron_colormap_rgb.txt')
  cmap = tools.cmap.Colormap.from_uint8_text(cmap_path)

  loader = ivi.ImageLoader(img_dir=data_dir, img_ext='npy')

  for file in track(loader.files):
    # 대상 정함
    if '018' not in file.as_posix():
      continue

    stem = file.stem

    arr = loader.read(file)
    equalized = skimage.exposure.equalize_hist(arr)

    # gray
    image = imt.array_to_image(arr)
    eq_image = imt.array_to_image(equalized)

    # iron cmap
    # image = imt.array_to_image(cmap(imt.normalize_array(arr)))
    # eq_image = imt.array_to_image(cmap(imt.normalize_array(equalized)))

    skimage.io.imsave(
        fname=res_dir.joinpath(stem + '_original.png').as_posix(),
        arr=image,
    )
    skimage.io.imsave(
        fname=res_dir.joinpath(stem + '_equalized.png').as_posix(),
        arr=eq_image,
    )
