"""
2020.11.24
흑백 이미지에 iron 컬러맵 적용
건기연 결과 보고용
"""
from pathlib import Path

import utils

import skimage.io
from rich.progress import track

import tools.cmap
import tools.imagetools as imt
import tools.ivimages as ivi


def colorize(files, save_dir: Path, cmap):
  dir_name = save_dir.name

  for file in track(files, description=dir_name):
    gray_array = skimage.io.imread(fname=file, as_gray=True)
    color_array = cmap(gray_array)
    color_image = imt.array_to_image(color_array)

    # imt.show_images_mpl([gray_image, color_image])
    skimage.io.imsave(fname=save_dir.joinpath(file.name), arr=color_image)


def colorize_seq(data_dirs, save_dirs, cmap, pattern):
  for data_dir, save_dir in zip(data_dirs, save_dirs):
    if not save_dir.exists():
      save_dir.mkdir()

    loader = ivi.ImageLoader(img_dir=data_dir, img_ext='.png', pattern=pattern)
    files = loader.files.copy()
    colorize(files=files, save_dir=save_dir, cmap=cmap)


if __name__ == "__main__":
  basedir = utils.PRJ_DIR.joinpath('result/stitch')

  # 1
  # data_dirs = [
  #     basedir.joinpath('KICT-2020-11-10_' + x) for x in ['allpano', 'cam-cali']
  # ]
  # save_dirs = [x.with_name(x.name + '_colorized') for x in data_dirs]
  # pattern = '*ir*.png'

  # 3
  data_dirs = [basedir.joinpath('KICT-2020-11-10_' + x) for x in ['basic']]
  save_dirs = [x.with_name(x.name + '_colorized') for x in data_dirs]
  pattern = '*ir*.png'

  # 실행부
  cmap_path = utils.PRJ_DIR.joinpath('data/cmap/iron_colormap_rgb.txt')
  cmap = tools.cmap.Colormap.from_uint8_text(cmap_path)

  colorize_seq(data_dirs=data_dirs,
               save_dirs=save_dirs,
               cmap=cmap,
               pattern=pattern)
