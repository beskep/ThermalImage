from pathlib import Path

import utils

import seaborn as sns
from matplotlib import colors
from rich.progress import track

from tools import imagetools as imt
from tools import ivimages as ivi


def write_thermal_image(image: ivi.IVImages,
                        path: Path,
                        cmap: colors.Colormap = None):
  img = image.ir_image

  if cmap is not None:
    img = cmap(img.astype('float') / 255.0) * 255
    img = img.astype('uint8')

  imt.imsave(path, img)


def write_thermal_images(loader: ivi.IVImagesLoader, save_dir: Path, cmap=None):
  if not save_dir.exists():
    save_dir.mkdir(parents=True)

  for iv in track(loader.ir.iv_list):
    image = loader.read_by_iv(iv)
    path = save_dir.joinpath(iv).with_suffix('.png')

    write_thermal_image(image=image, path=path, cmap=cmap)


def main(data_dir: Path, save_dir: Path, cmap=None):
  loader = ivi.IVImagesLoader(vis_dir=None, ir_dir=data_dir, ir_ext='xlsx')

  write_thermal_images(loader=loader, save_dir=save_dir, cmap=None)
  if cmap is not None:
    write_thermal_images(loader=loader,
                         save_dir=save_dir.with_name(save_dir.name + '_cmap'),
                         cmap=cmap)


if __name__ == '__main__':
  data_dir = Path(
      r'D:\repo\ThermalImage\AnomalyDetection\gongrung1&2_2020-11-26\excel')
  save_dir = Path(
      r'D:\repo\ThermalImage\AnomalyDetection\gongrung1&2_2020-11-26\ir')

  # data_dir = Path(
  #     r'D:\repo\ThermalImage\OriginalDefectDetection\20190109KwangWoon')
  # save_dir = Path(
  #     r'D:\repo\ThermalImage\OriginalDefectDetection\20190109KwangWoonImage')

  if not save_dir.exists():
    save_dir.mkdir()

  # cmap = sns.color_palette('rocket', as_cmap=True)
  cmap = None

  # for ddir in data_dir.iterdir():
  #     if not ddir.is_dir():
  #         continue

  #     sdir = save_dir.joinpath(ddir.name)
  #     main(data_dir=ddir, save_dir=sdir)

  main(data_dir=data_dir, save_dir=save_dir, cmap=cmap)
