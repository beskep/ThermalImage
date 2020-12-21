import utils

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import tools.exif

if __name__ == '__main__':
  path = r'D:\repo\ThermalImage\Panorama\건기연 촬영 (2020.11.10)\IR_2020-11-10_0001.jpg'
  tag = '-Palette'

  data_dir = utils.PRJ_DIR.joinpath('data')
  res_dir = utils.PRJ_DIR.joinpath('result')

  palette = tools.exif.get_exif_binary(image_path=path, tag=tag)
  palette_ycrcb = np.array(list(palette)).reshape([1, -1, 3]).astype('uint8')

  np.savetxt(fname=data_dir.joinpath('iron_colormap_ycrcb.txt').as_posix(),
             X=palette_ycrcb.reshape([-1, 3]),
             fmt='%d')

  palette_rgb = cv.cvtColor(palette_ycrcb, code=cv.COLOR_YCrCb2RGB)
  np.savetxt(fname=data_dir.joinpath('iron_colormap_rgb.txt').as_posix(),
             X=palette_rgb.reshape([-1, 3]),
             fmt='%d')

  # visualization
  palette_rgb_wide = np.vstack([palette_rgb] * 10)

  fig, ax = plt.subplots(1, 1, figsize=(8, 1))
  ax.imshow(palette_rgb_wide)
  ax.set_yticks([])
  xticks = ax.get_xticks()
  xticks = xticks[xticks >= 0]
  xticks = xticks[xticks < 225]
  ax.set_xticks(list(xticks) + [223])

  fig.tight_layout()
  fig.savefig(res_dir.joinpath('iron_colormap.png').as_posix(), dpi=200)
