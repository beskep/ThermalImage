"""
2020.10.26.
건기연 FLIR 데이터 변환, 전처리
"""
from itertools import product
from pathlib import Path

import utils

import cv2 as cv
import matplotlib.pyplot as plt
import yaml
from rich.progress import track
from skimage.exposure import rescale_intensity
from skimage.io import imsave

import flir
from tools import exif
from tools import imagetools as imt

cmap = plt.get_cmap('inferno')


def extract_flir_images(src_dir,
                        dst_dir,
                        extract_image=True,
                        extract_exif=True,
                        ir_preprocess=None,
                        vis_preprocess=None,
                        ir_suffix='.npy',
                        vis_suffix='.png'):
  src_dir = Path(src_dir)
  dst_dir = Path(dst_dir)
  for path in (src_dir, dst_dir):
    if not path.exists():
      raise FileNotFoundError(path)

  for d1, d2 in product(['ir', 'vis'], ['', 'preprocess']):
    d = dst_dir.joinpath(d1, d2)
    if not d.exists():
      d.mkdir(parents=True)
  if extract_exif:
    d = dst_dir.joinpath('EXIF')
    if not d.exists():
      d.mkdir(parents=True)

  extractor = flir.FlirExtractor()

  files = list(src_dir.rglob('*.jpg'))
  for file in track(files):
    # 파일 이름
    fname = file.with_suffix('').name

    if extract_image or ir_preprocess or vis_preprocess:
      # 이미지 추출
      extractor.process_image(file)
      ir, vis = extractor.extract_data()
      ir_image = imt.array_to_image(ir)

      # 원본 저장
      ir_path = dst_dir.joinpath('ir/{}'.format(fname))
      vis_path = dst_dir.joinpath('vis/{}'.format(fname))
      imsave(fname=ir_path.with_suffix('.png'), arr=ir_image)
      extractor.write_data(ir_path=ir_path.with_suffix(ir_suffix),
                           vis_path=vis_path.with_suffix(vis_suffix))
    else:
      ir, vis, ir_image = None, None, None

    if extract_exif:
      # EXIF 태그 저장
      tags = exif.get_exif_tags(file)
      tags_path = dst_dir.joinpath('EXIF', fname).with_suffix('.yaml')
      with open(tags_path, 'w') as f:
        yaml.dump(data=tags, stream=f)

    # 전처리 거친 이미지 저장
    if ir_preprocess is not None:
      irp = ir_image.copy()
      for fn in ir_preprocess:
        irp = fn(irp)

      irp = rescale_intensity(
          cmap(rescale_intensity(irp, out_range=(0.0, 1.0))),
          out_range='uint8',
      )
      irp_path = dst_dir.joinpath(
          'ir/preprocess/{}'.format(fname)).with_suffix('.png')
      imsave(fname=irp_path, arr=irp, check_contrast=False)

    if vis_preprocess is not None:
      visp = vis.copy()
      for fn in vis_preprocess:
        visp = fn(visp)

      visp_path = dst_dir.joinpath(
          'vis/preprocess/{}'.format(fname)).with_suffix('.png')
      imsave(fname=visp_path, arr=visp)


if __name__ == '__main__':
  src_dir = Path(r'D:\repo\ThermalImage\Panorama\KICT\2020-11-10\original')
  dst_dir = Path(r'D:\repo\ThermalImage\Panorama\KICT\2020-11-10\extracted')

  src_dir = Path(r'D:\Python\ThermalPanorama\Registration\data\KICT\FLIRE95')
  dst_dir = Path(r'D:\Python\ThermalPanorama\Registration\data\KICT\FLIRE95')

  subdirs = src_dir.glob('**/*')

  # ir_preprocess = [
  #     lambda x: cv.equalizeHist(x, dst=None),
  #     # lambda x: cv.bilateralFilter(x, d=-1, sigmaColor=20, sigmaSpace=10)
  # ]
  # vis_preprocess = [
  #     imt.normalize_rgb_image_hist,
  #     # lambda x: cv.bilateralFilter(x, d=-1, sigmaColor=20, sigmaSpace=10)
  # ]
  ir_preprocess = None
  vis_preprocess = None

  if not dst_dir.exists():
    dst_dir.mkdir(parents=True)

  extract_flir_images(src_dir=src_dir,
                      dst_dir=dst_dir,
                      extract_image=True,
                      extract_exif=True,
                      ir_preprocess=ir_preprocess,
                      vis_preprocess=vis_preprocess)
