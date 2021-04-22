import io
import logging

import utils

import numpy as np
from kivy.core.image import Image as KvCoreImage
from kivy.uix.image import Image as KvImage
from PIL import Image as PilImage


def kivy_core_texture(image: np.ndarray):
  image_pil = PilImage.fromarray(image)
  image_io = io.BytesIO()
  image_pil.save(image_io, format='png')
  image_io.seek(0)
  image_data = io.BytesIO(image_io.read())
  texture = KvCoreImage(image_data, ext='png').texture

  return texture


class ImageWidget(KvImage):

  def __init__(self, **kwargs):
    self.texture = None
    super().__init__(**kwargs)

  def update_image(self, image: np.ndarray, reload=True):
    texture = kivy_core_texture(image)
    self.texture = texture

    if reload:
      self.reload()
