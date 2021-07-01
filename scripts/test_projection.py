import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform

import tools.imagetools as imt
from tools import projection

if __name__ == '__main__':
  path = (r'D:\test\panorama\MainBldgBackLoc1PanTiltTripodResult'
          r'\panorama_colormap.png')
  image = skimage.io.imread(path)

  image_alpha = image[:, :, -1]
  image = image[:, :, :3]
  image[np.logical_not(image_alpha)] = 0

  # imt.show_images_mpl([image])

  image = skimage.transform.rescale(image=image[:, :, :3],
                                    scale=0.1,
                                    multichannel=True)

  prj = projection.ImageProjection(image=image,
                                   angles=(0.0, 0.0, 0.0),
                                   viewing_angle=(42 * np.pi / 180.0))

  max_angle = 60.0 * np.pi / 180.0

  roll = 0.0
  # uds = np.linspace(-max_angle, max_angle, num=5)
  lrs = np.linspace(-max_angle, max_angle, num=5)

  uds = np.linspace(0, max_angle, num=3)
  # uds = [40.0 * np.pi / 180.0]
  # lrs = [10.0 * np.pi / 180.0]

  fig, axes = plt.subplots(nrows=len(uds), ncols=len(lrs), squeeze=False)

  for row, ud in enumerate(uds):
    for col, lr in enumerate(lrs):
      # print(f'row: {row} | col: {col}')

      prj.angles = (roll, ud, lr)

      projected = prj.project()
      axes[row, col].imshow(projected)

      axes[row, col].set_xticks([])
      axes[row, col].set_yticks([])

  fig.tight_layout()
  plt.show()
