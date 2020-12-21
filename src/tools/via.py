"""
VGG Image Annotator (VIA)의 프로젝트 저장 결과로부터 annotation 해석
"""
import json
import os
from collections import defaultdict
from pathlib import Path
from warnings import warn

import numpy as np
import skimage.draw
import skimage.io


def draw_mask(shape, rows, cols):
  mask = np.zeros(shape=shape, dtype=np.int)
  rr, cc = skimage.draw.polygon(r=rows, c=cols)
  mask[rr, cc] = 1

  return mask


class VIAProject:

  def __init__(self, path, attribute_name, attributes_ids):
    path = os.path.normpath(path)
    if not os.path.exists(path):
      raise FileNotFoundError(path)

    self._path = path
    self._json = json.load(open(path, 'r', encoding='utf-8-sig'))
    self._image_metadata = self._json['_via_img_metadata']
    self._fname_dict = {
        os.path.normpath(self._image_metadata[x]['filename']): x
        for x in self._image_metadata
    }

    self._attr_name = attribute_name
    self._attr_ids = attributes_ids

  @property
  def files(self):
    return list(self._fname_dict.keys())

  def fname_key(self, fname: str):
    return self._fname_dict[fname]

  def meta_data(self, key):
    return self._image_metadata[key]

  def regions(self, fname: str):
    meta_data = self.meta_data(self.fname_key(fname))
    regions = meta_data['regions']

    for region in regions:
      region_shape = region['shape_attributes']
      assert region_shape['name'] == 'polygon'

      ra = region['region_attributes'][self._attr_name]
      if not any(x in ra for x in self._attr_ids):
        warn('file {}: attribute 지정 안됨'.format(fname))
        continue

      for x in self._attr_ids:
        if x in ra and ra[x]:
          region_class = x
          break
      else:
        raise ValueError

      yield region_class, region_shape

  def write_masks(self, fname, save_dir, shape):
    save_dir = Path(save_dir).resolve()
    fname_stem = Path(fname).stem
    class_count = defaultdict(int)

    for rclass, rshape in self.regions(fname):
      class_count[rclass] += 1
      mask = draw_mask(shape=shape,
                       rows=rshape['all_points_y'],
                       cols=rshape['all_points_x'])
      mask = mask.astype('uint8') * 255

      path = save_dir.joinpath('{}_{}_{}'.format(
          fname_stem, rclass, class_count[rclass])).with_suffix('.png')
      skimage.io.imsave(fname=path, arr=mask, check_contrast=False)
