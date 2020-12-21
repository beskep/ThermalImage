"""
2020.11.10 건기연에서 촬영해온 데이터 
열화상 stitching 테스트
"""
import itertools
import sys
from pathlib import Path

import utils

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure
from rich.console import Console
from rich.progress import track

from tools import imagetools as imt
from tools import ivimages as ivi
from tools import stitcher
from tools.stitching_option import StitchingOption

console = Console()


class Stitcher(stitcher.Stitcher):

  def __init__(self, mode):
    super(Stitcher, self).__init__(mode=mode)

    self.features_finder = cv.ORB_create()
    self.seam_finder = 'gc_color'
    self.flag_wave_correction = False


def stitch(images, save_dir: Path, fname, mode, mask=None):
  # stitcher = stitcher_dict[mode]
  save_path = save_dir.joinpath('stitched_{}'.format(fname))
  if any(save_path.with_suffix(x).exists() for x in ['.png', '.txt']):
    return None, None

  stitcher = Stitcher(mode)

  if images[0].ndim == 2:
    for idx in range(len(images)):
      images[idx] = np.repeat(images[idx][:, :, np.newaxis], 3, axis=2)

  try:
    stitched_image, stitched_mask = stitcher.stitch(images=images, masks=mask)
  except (ValueError, cv.error):
    stitched_image = None
    stitched_mask = None

  if stitched_image is not None:
    if stitched_image.dtype != np.uint8:
      stitched_image = imt.array_to_image(stitched_image.astype('float'))

    fig, axes = imt.show_images_mpl(images=(images +
                                            [stitched_image, stitched_mask]),
                                    show=False,
                                    cmap='inferno')
    save_path = save_dir.joinpath('compare_{}.jpg'.format(fname))
    fig.savefig(save_path.as_posix(), dpi=200)
    plt.close(fig)

    save_path = save_dir.joinpath('stitched_{}.png'.format(fname))
    imt.imsave(fname=save_path.as_posix(),
               arr=stitched_image,
               check_contrast=False)
  else:
    save_path = save_dir.joinpath('stitched_{}.txt'.format(fname))
    with open(save_path, 'w') as f:
      f.write('')

  return stitched_image, stitched_mask


def full_iv(x):
  return 'IR_2020-11-10_{:04d}'.format(x)


def main(
    option: StitchingOption,
    loader: ivi.IVImagesLoader,
    preprocess,
    case,
    direction,
    spec,
    mode,
    modes,
):
  elements = list(option.elements(case=case, direction=direction))
  if spec == 'ir':
    ldr = loader.ir
  elif spec == 'vis':
    ldr = loader.vis
  else:
    raise ValueError

  if direction == 'all':
    assert len(elements) == 1
    fname = '{}_{}_{}_{}'.format(case, spec, direction, mode)
    if save_dir.joinpath('stitched_{}.png'.format(fname)).exists():
      return

    images = [ldr.read_by_iv(full_iv(x)) for x in elements[0]]
    if preprocess is not None:
      images = [preprocess(x)[0] for x in images]

    stitch(images=images, save_dir=save_dir, fname=fname, mode=mode)

  else:
    fname = '{}_{}_{}_m1{}'.format(case, spec, direction, mode)
    fnames = [
        save_dir.joinpath('stitched_{}.png'.format(fname + '_m2' + x))
        for x in modes
    ]
    if any(x.exists() for x in fnames):
      return

    subimages = []
    submasks = []
    for idx, subset in enumerate(elements):
      images = [ldr.read_by_iv(full_iv(x)) for x in subset]
      subimage, submask = stitch(images=images,
                                 save_dir=save_dir,
                                 fname=(fname + '_{}'.format(idx)),
                                 mode=mode)
      if subimage is not None:
        subimages.append(subimage)
        submasks.append(submask)

    if subimages:
      for mode2 in modes:
        stitch(images=subimages,
               save_dir=save_dir,
               fname=fname + '_m2' + mode2,
               mode=mode2,
               mask=submasks)
    else:
      console.print('\n', fname, 'failed', style='bold red')


if __name__ == '__main__':
  if len(sys.argv) > 1:
    option_path = Path(sys.argv[1])
    if not option_path.exists():
      option_path = dirs.ROOT_DIR.joinpath('data', option_path)
  else:
    option_path = dirs.ROOT_DIR.joinpath('data/KICT-2020-11-10.yaml')

  assert option_path.exists()
  option = StitchingOption(open(option_path, 'r'))

  cases = option.cases()
  spectrum = ['ir', 'vis']
  modes = ['pano', 'scan']

  # save_dir = dirs.ROOT_DIR.joinpath('result/stitch/KICT-2020-11-10')
  save_dir = Path(option.options['save_dir'])
  if not save_dir.exists():
    save_dir.mkdir(parents=True)

  def preprocess_ir(array, trange=(-40, None)):
    if trange is None:
      mask = None
    else:
      mask = np.ones_like(array, dtype=np.uint8)
      if trange[0] is not None:
        mask[array < trange[0]] = 0
      if trange[1] is not None:
        mask[array > trange[1]] = 0

    array = imt.normalize_array(array.astype('float'))
    array = skimage.exposure.equalize_hist(array, mask=mask)
    image = imt.array_to_image(array)
    # image = cv.equalizeHist(image, dst=None)
    image = cv.bilateralFilter(image, d=-1, sigmaColor=20, sigmaSpace=10)
    image[mask == 0] = 0

    return image, mask

  def preprocess_vis(image):
    image = imt.normalize_rgb_image_hist(image)
    image = cv.bilateralFilter(image, d=-1, sigmaColor=20, sigmaSpace=10)

    return image, None

  for case in cases:
    console.print('case:', case)

    loader = option.ivimages_loader(case)
    # directions = option.directions(case)
    directions = ['all']

    prod = list(itertools.product(directions, spectrum, modes))
    it = track(prod)
    # it = prod
    for direction, spec, mode in it:
      console.print(
          '\ncase: {} | {} | {} | {}'.format(case, direction, spec, mode),)

      if spec == 'ir':
        preprocess = preprocess_ir
      else:
        preprocess = preprocess_vis

      try:
        main(option=option,
             loader=loader,
             preprocess=preprocess,
             case=case,
             direction=direction,
             spec=spec,
             mode=mode,
             modes=modes)
      except Exception as e:
        console.print('\nFucked up \n{}'.format(e), style='bold red')
        raise e
