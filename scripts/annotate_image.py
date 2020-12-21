"""
2020.10.26.
건기연 FLIR 데이터 변환, 전처리
"""
from pathlib import Path

import utils

import cv2 as cv
import numpy as np
import seaborn as sns
from matplotlib.colors import Colormap
from tqdm import tqdm

import flir
from tools import imagetools as imt
from tools import ivimages as ivi


def annotate_image(src_dir, dst_dir, cmap: Colormap, text_fn=None):
  src_dir = Path(src_dir)
  dst_dir = Path(dst_dir)
  for path in (src_dir, dst_dir):
    if not path.exists():
      raise FileNotFoundError(path)

  # extractor = flir.FlirExtractor()
  loader = ivi.ImageLoader(img_dir=src_dir, img_ext='npy')

  # images = list(src_dir.rglob('*.jpg'))
  images = loader.files
  for image in tqdm(images):
    fname = image.with_suffix('').name
    text = fname if text_fn is None else text_fn(fname)

    # extractor.process_image(image)
    # ir, vis = extractor.extract_data()
    ir = loader.read(image)

    irmin = np.min(ir)
    irmax = np.max(ir)
    ir_norm = (ir - irmin) / (irmax - irmin)

    ir_color = cmap(ir_norm) * 255
    ir_image = ir_color.astype('uint8')

    # loc = (int(ir_image.shape[1] / 2), int(ir_image.shape[0] / 2))
    loc = (20, ir_image.shape[0] - 20)

    ir_text = cv.putText(ir_image,
                         text=text,
                         org=loc,
                         fontFace=cv.FONT_HERSHEY_DUPLEX,
                         fontScale=2,
                         color=(0, 0, 0, 255),
                         thickness=2)

    # imt.show_images_mpl([ir_text])
    imt.imsave(fname=dst_dir.joinpath(fname).with_suffix('.png'), arr=ir_text)


if __name__ == '__main__':
  src_dir = Path(r'D:\repo\ThermalImage\Panorama\KICT\2020-11-10\extracted')
  dst_dir = Path(r'D:\repo\ThermalImage\Panorama\KICT\2020-11-10\annotated')
  subdirs = src_dir.glob('**/')
  cmap = sns.color_palette('rocket', as_cmap=True)

  for src in src_dir.iterdir():
    assert src.is_dir()

    print(src)
    dst = dst_dir.joinpath(src.name)
    if not dst.exists():
      dst.mkdir(parents=True)

    annotate_image(src_dir=src,
                   dst_dir=dst,
                   cmap=cmap,
                   text_fn=lambda x: x.replace('2020-11-10_', ''))
