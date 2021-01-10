import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage.exposure
import skimage.io
from matplotlib.lines import Line2D

import utils
import tools.imagetools as imt
import tools.ivimages as ivi


def image_and_hist_plot(images, xlabels, hist_kwargs, cumul=False):
  fig, axes = plt.subplots(len(images), 2, figsize=(8, 4.5))
  for row, (img, xlabel, kw) in enumerate(zip(images, xlabels, hist_kwargs)):
    axes[row, 0].imshow(img, cmap='gray')
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])

    sns.histplot(
        img.flatten(),
        ax=axes[row, 1],
        stat='probability',
        # bins=bins,
        # binwidth=1.0,
        color=color,
        cumulative=cumul,
        **kw)
    axes[row, 1].set_xlabel(xlabel)

    if row == 0:
      axes[row, 1].set_xlabel('Temperature (ºC)')
    else:
      axes[row, 1].set_xlabel('Equalized temperature')

  fig.tight_layout()

  return fig, axes


if __name__ == '__main__':
  sns.set_context('paper')

  data_dir = utils.PRJ_DIR.joinpath('data/preprocessing_example')
  res_dir = utils.PRJ_DIR.joinpath('result/preprocessing_histogram')
  if not res_dir.exists():
    res_dir.mkdir()

  loader = ivi.ImageLoader(img_dir=data_dir, img_ext='npy')

  bins = 'sturges'
  bins = 'auto'
  # color = sns.color_palette('Dark2', n_colors=3)[0]
  color = sns.mpl_palette('Dark2')[0]

  for file in loader.files:
    fname = file.with_suffix('').name
    path = res_dir.joinpath(fname).as_posix()

    image = loader.read(file)
    image_eq = skimage.exposure.equalize_hist(image)

    image_int = imt.array_to_image(image)
    image_eq_int = imt.array_to_image(image_eq)

    # fig, axes = image_and_hist_plot(
    #     images=[image, image_eq],
    #     xlabels=['Temperature (ºC)', 'Equalized temperature'],
    #     hist_kwargs=[dict(binwidth=1.0), dict(bins='auto')])

    fig, axes = plt.subplots(2, 2, figsize=(8, 4.5))
    for row, img in enumerate([image, image_eq]):
      axes[row, 0].imshow(img, cmap='gray')
      axes[row, 0].set_xticks([])
      axes[row, 0].set_yticks([])

      if row == 0:
        kwargs = {'binwidth': 1.0}
      else:
        kwargs = {'bins': 'auto'}

      sns.histplot(
          img.flatten(),
          ax=axes[row, 1],
          stat='probability',
          # bins=bins,
          # binwidth=1.0,
          color=color,
          **kwargs)

      if row == 0:
        axes[row, 1].set_xlabel('Temperature (ºC)')
      else:
        axes[row, 1].set_xlabel('Equalized temperature')

    fig.tight_layout()

    fig.savefig(path + '_hist.png', dpi=300)
    # plt.close(fig)

    legend_colors = [color, (0.2, 0.2, 0.2)]
    legend_lines = [
        Line2D([0], [0], color=x, linewidth=2) for x in legend_colors
    ]
    legend_labels = ['Probability', 'Cumulative dist.']

    for row, img in enumerate([image, image_eq]):
      ax2 = axes[row, 1].twinx()
      hist = np.histogram(img.flatten(), bins=100)
      cum_sum = np.cumsum(hist[0])

      line_x = np.repeat(hist[1], 2)[1:-1]
      line_y = np.repeat(cum_sum / np.max(cum_sum), 2)
      assert line_x.shape == line_y.shape

      ax2.plot(line_x, line_y, c=(0.2, 0.2, 0.2))
      ax2.set_ylabel('Cumulative distribution')

      ax2.legend(legend_lines, legend_labels)

    fig.tight_layout()
    # plt.show()

    fig.savefig(path + '_hist_cumul.png', dpi=300)
    plt.close(fig)

    skimage.io.imsave(fname=path + '_orig.png', arr=image_int)
    skimage.io.imsave(fname=path + '_eq.png', arr=image_eq_int)

    fig, axes = imt.show_images_mpl([image, image_eq], show=False)
    fig.savefig(res_dir.joinpath(fname).with_suffix('.png').as_posix(), dpi=200)
    plt.close(fig)

    minmax = (np.min(image), np.max(image))
    with open(path + '_minmax.txt', 'w') as f:
      f.write(str(minmax))
