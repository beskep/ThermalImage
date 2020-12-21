import sys
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from rich.progress import track

import utils

from tools import imagetools as imt
from tools import ivimages as ivi
from tools.stitcher import Stitcher


class StitchTester:

  def __init__(self, stitcher: Stitcher, loader: ivi.ImageLoader):
    self.stitcer = stitcher
    self.loader = loader
    self.ivimage_dict = dict()

  def load(self, iv):
    if iv in self.ivimage_dict:
      image = self.ivimage_dict[iv]
    else:
      image = self.loader.read_by_iv(iv)
      self.ivimage_dict[iv] = image
    return image

  def stitch(self, ivs, save_dir: Path, case_name=None):
    images = [self.load(x) for x in ivs]
    for idx in range(len(images)):
      if images[idx].ndim == 2:
        images[idx] = np.repeat(images[idx][:, :, np.newaxis], 3, axis=2)

    if case_name is None:
      case_name = '_'.join([str(x) for x in ivs])

    stitched_image, stitched_mask = self.stitcer.stitch(images=images)

    fig, axes = imt.show_images_mpl(images + [stitched_image, stitched_mask],
                                    show=False,
                                    cmap='inferno')
    path = save_dir.joinpath('compare_' + case_name).with_suffix('.png')
    fig.savefig(path.as_posix())
    plt.close(fig)

    path = save_dir.joinpath('stitched_' + case_name).with_suffix('.png')
    imt.imsave(fname=path.as_posix(), arr=stitched_image, check_contrast=False)


if __name__ == '__main__':
  cases = [
      # 세로
      ['WaterResourceBuilding', ['0006', '0007']],
      ['WaterResourceBuilding', ['0007', '0008']],
      ['WaterResourceBuilding', ['0009', '0010']],
      ['WaterResourceBuilding', ['0010', '0011']],
      ['WaterResourceBuilding', ['0012', '0013']],
      ['WaterResourceBuilding', ['0013', '0014']],
      ['WaterResourceBuilding', ['0015', '0016']],
      ['WaterResourceBuilding', ['0016', '0017']],
      # 가로
      # 1행
      ['WaterResourceBuilding', ['0006', '0009']],
      ['WaterResourceBuilding', ['0009', '0012']],
      ['WaterResourceBuilding', ['0012', '0015']],
      # 2행
      ['WaterResourceBuilding', ['0007', '0010']],
      ['WaterResourceBuilding', ['0010', '0013']],
      ['WaterResourceBuilding', ['0013', '0016']],
      # 3행
      ['WaterResourceBuilding', ['0008', '0011']],
      ['WaterResourceBuilding', ['0011', '0014']],
      ['WaterResourceBuilding', ['0014', '0017']],
      # 4개
      ['WaterResourceBuilding', [6, 9, 7, 10]],
      ['WaterResourceBuilding', [7, 10, 8, 11]],
      ['WaterResourceBuilding', [9, 12, 10, 13]],
      ['WaterResourceBuilding', [10, 13, 11, 14]],
      ['WaterResourceBuilding', [12, 15, 13, 16]],
      ['WaterResourceBuilding', [13, 16, 14, 17]],
      # 6개 (가로)
      ['WaterResourceBuilding', [6, 9, 12, 7, 10, 13]],
      ['WaterResourceBuilding', [7, 10, 13, 8, 11, 14]],
      ['WaterResourceBuilding', [9, 12, 15, 10, 13, 16]],
      ['WaterResourceBuilding', [10, 13, 16, 11, 14, 17]],
      # 6개 (세로)
      ['WaterResourceBuilding', [6, 7, 8, 9, 10, 11]],
      ['WaterResourceBuilding', [9, 10, 11, 12, 13, 14]],
      ['WaterResourceBuilding', [12, 13, 14, 15, 16, 17]],
      # 9개
      ['WaterResourceBuilding', [6, 9, 12, 7, 10, 13, 8, 11, 14]],
      ['WaterResourceBuilding', [9, 12, 15, 10, 13, 16, 11, 14, 17]],
      # 전체
      ['WaterResourceBuilding', list(range(6, 18))],
  ]

  image_dir = Path(r'D:\repo\ThermalImage\Panorama\KICT\WaterResourceBuilding')
  save_dir = dirs.ROOT_DIR.joinpath('result/stitch/WaterResourceBuilding')

  for mode in ['pano', 'scan']:
    stitcher = Stitcher(mode=mode)
    stitcher.features_finder = cv.ORB_create()
    # stitcher.blend_type = 'no'
    stitcher.seam_finder = 'gc_color'
    stitcher.flag_wave_correction = False

    ir_loader = ivi.ImageLoader(
        img_dir=image_dir.joinpath('ir/preprocess'),
        img_ext='png',
    )
    vis_loader = ivi.ImageLoader(
        img_dir=image_dir.joinpath('vis/preprocess'),
        img_ext='png',
    )

    ir_tester = StitchTester(stitcher=stitcher, loader=ir_loader)
    vis_tester = StitchTester(stitcher=stitcher, loader=vis_loader)

    ir_save_dir = save_dir.joinpath('{}/ir'.format(mode))
    vis_save_dir = save_dir.joinpath('{}/vis'.format(mode))
    for d in [ir_save_dir, vis_save_dir]:
      if not d.exists():
        d.mkdir(parents=True)

    case_num = {}
    for case, ivs in track(cases[::-1]):
      ivs = [int(x) for x in ivs]
      full_ivs = ['IR_2020-10-19_{:04d}'.format(x) for x in ivs]

      iv_count = len(ivs)
      case_name = 'imgs{}_'.format(iv_count)
      if iv_count <= 6:
        case_name += '_'.join(['{:02d}'.format(x) for x in ivs])
      else:
        if iv_count in case_num:
          ic = case_num[iv_count]
        else:
          ic = 0
          case_num[iv_count] = 0
        case_name += 'case_{}'.format(ic)

      try:
        ir_tester.stitch(ivs=full_ivs,
                         case_name=case_name,
                         save_dir=ir_save_dir)
      except ValueError as e:
        print(case_name, 'ir')
        print(e)

      try:
        vis_tester.stitch(ivs=full_ivs,
                          case_name=case_name,
                          save_dir=vis_save_dir)
      except ValueError as e:
        print(case_name, 'vis')
        print(e)
