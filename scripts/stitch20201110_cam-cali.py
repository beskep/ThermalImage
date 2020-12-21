"""
2020.11.10 건기연에서 촬영해온 데이터 
열화상 stitching 테스트
camera calibration 적용 테스트
"""

import sys
from itertools import product
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
from tools.camera_calibration import CameraCalibration
from tools.stitching_option import StitchingOption

console = Console(width=120)


class Stitcher(stitcher.Stitcher):

  def __init__(self, mode, try_cuda=False, warper=None):
    super(Stitcher, self).__init__(mode=mode, try_cuda=try_cuda)

    self.features_finder = cv.ORB_create()
    self.seam_finder = 'gc_color'
    self.flag_wave_correction = False

    if warper is not None:
      self.warper_type = warper


def stitch(images, save_dir: Path, fname, stitch_kwargs: dict, masks=None):
  stitcher = Stitcher(**stitch_kwargs)

  if images[0].ndim == 2:
    for idx in range(len(images)):
      images[idx] = np.repeat(images[idx][:, :, np.newaxis], 3, axis=2)

  try:
    stitched_image, stitched_mask = stitcher.stitch(images=images, masks=masks)
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

  return stitched_image, stitched_mask


def full_iv(x):
  return 'IR_2020-11-10_{:04d}'.format(x)


def stitch_seq(option: StitchingOption,
               loader: ivi.IVImagesLoader,
               preprocess,
               case,
               spec,
               mode,
               warpers,
               resize=1.0):
  elements = list(option.elements(case=case, direction='all'))
  if spec == 'ir':
    ldr = loader.ir
  elif spec == 'vis':
    ldr = loader.vis
  else:
    raise ValueError
  assert len(elements) == 1

  raw_images = [ldr.read_by_iv(full_iv(x)) for x in elements[0]]
  prep = [preprocess(x) for x in raw_images]
  images = [x[0] for x in prep]
  masks = [x[1] for x in prep]

  if resize and resize < 1.0:

    def resize_fn(img):
      if img is None:
        res = None
      else:
        res = cv.resize(img,
                        dsize=None,
                        fx=resize,
                        fy=resize,
                        interpolation=cv.INTER_AREA)
      return res

    images = [resize_fn(x) for x in images]
    masks = [resize_fn(x) for x in masks]

  it = track(warpers,
             description='{} | {:>3}'.format(case, spec),
             console=console)
  for warper in it:
    fname = '{}_{}_{}_{}'.format(case, spec, mode, warper)
    paths = [
        save_dir.joinpath('stitched_{}'.format(fname)).with_suffix(x)
        for x in ['.png', '.txt']
    ]
    if any(x.exists() for x in paths):
      continue

    stitch_kwargs = {'mode': mode, 'warper': warper}

    try:
      simage, smask = stitch(images=images,
                             save_dir=save_dir,
                             fname=fname,
                             stitch_kwargs=stitch_kwargs,
                             masks=masks)
    except (ValueError, cv.error):
      simage = None

    if simage is None:
      path = save_dir.joinpath('stitched_{}.txt'.format(fname))
      with open(path, 'w') as f:
        f.write('fucked')


if __name__ == '__main__':
  if len(sys.argv) > 1:
    option_path = Path(sys.argv[1])
    if not option_path.exists():
      option_path = dirs.ROOT_DIR.joinpath('data', option_path)
  else:
    option_path = dirs.ROOT_DIR.joinpath('data/KICT-2020-11-10_pano.yaml')

  assert option_path.exists(), option_path
  option = StitchingOption(open(option_path, 'r'))

  cases = option.cases()
  spectrum = ['ir']
  modes = ['pano']
  mode = modes[0]
  # warpers = stitcher.AVAILABLE_WARPER[:]
  warpers = (
      'compressedPlaneA1.5B1',
      # 'compressedPlaneA2B1',
      # 'paniniA1.5B1',
      # 'paniniA2B1',
      'plane',
      'spherical',
  )

  cali_path = dirs.ROOT_DIR.joinpath('data/FLIR_E95_Calibration/parameters.npz')
  cali = CameraCalibration(cali_path)

  def preprocess_ir(array, trange=(-40, None), calibration=True):
    if trange is not None:
      mask = np.ones_like(array, dtype=np.uint8)
      if trange[0] is not None:
        mask[array < trange[0]] = 0
      if trange[1] is not None:
        mask[array > trange[1]] = 0
    mask = None

    array = imt.normalize_array(array)
    array = skimage.exposure.equalize_hist(array, mask=mask)
    image = imt.array_to_image(array)
    # image = cv.equalizeHist(image, dst=None)
    image = cv.bilateralFilter(image, d=-1, sigmaColor=20, sigmaSpace=10)

    if mask is not None:
      image[mask == 0] = 0

    if calibration:
      image = cali.calibrate(image)

    return image, None

  save_dir = dirs.ROOT_DIR.joinpath('result/stitch/KICT-2020-11-10-cam-cali')
  if not save_dir.exists():
    save_dir.mkdir(parents=True)

  for case, spec in product(cases, spectrum):
    # console.print('case:', case)
    loader = option.ivimages_loader(case)
    directions = option.directions(case)
    assert len(directions) == 1
    assert directions[0] == 'all'

    try:
      stitch_seq(option=option,
                 loader=loader,
                 preprocess=lambda x: preprocess_ir(
                     x, trange=(-40, None), calibration=True),
                 case=case,
                 spec=spec,
                 mode=mode,
                 warpers=warpers,
                 resize=0.5)
    except Exception as e:
      console.print('\nFucked up \n{}'.format(e), style='bold red')
      raise e
