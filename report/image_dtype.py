import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import seaborn as sns
import skimage.transform

import utils
from tools.imagetools import normalize_array

if __name__ == "__main__":
  sns.set_style('whitegrid')
  sns.set_context('paper', font_scale=0.9)

  # data
  flights_long = sns.load_dataset('flights')
  flights = flights_long.pivot('month', 'year', 'passengers').to_numpy()
  half_size = [int(x / 2.0) for x in flights.shape[:2]]
  flights_half = skimage.transform.resize(
      image=flights,
      output_shape=tuple(half_size),
  )

  flights_norm = normalize_array(flights_half.astype('float'))
  pseudo_temp = flights_norm * 10 - 2
  pseudo_temp[3, 5] -= 1.5
  flights_image = (255 * normalize_array(pseudo_temp)).astype('uint8')

  # colormap
  color = 0.9
  cdict = {
      x: ((0, color, color), (1, color, color))
      for x in ['red', 'green', 'blue']
  }
  custom_cmap = LinearSegmentedColormap('whitecmap', cdict)

  iron = np.loadtxt(
      dirs.ROOT_DIR.joinpath('data/iron_colormap_rgb.txt').as_posix())

  iron_cmap = ListedColormap(iron / 255)

  # plot
  fig, axes = plt.subplots(1, 4, figsize=(9, 2))
  for ax, arr, cmap, fmt in zip(
      axes,
      [pseudo_temp, flights_image, flights_image, flights_image],
      [custom_cmap, custom_cmap, 'gray', iron_cmap],
      ['.1f', 'd', 'd', 'd'],
  ):
    sns.heatmap(arr,
                annot=True,
                fmt=fmt,
                cbar=False,
                cmap=cmap,
                linewidth=0.5,
                ax=ax)

  for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

  # plt.show()
  fig.savefig(dirs.ROOT_DIR.joinpath('report/images.png'), dpi=200)
