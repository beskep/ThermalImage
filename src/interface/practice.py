import io
import logging

import utils

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
from kivy.core.image import Image as KvCoreImage
from kivy.lang import Builder
from kivy.uix.image import Image as KvImage
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.slider import MDSlider
from PIL import Image as PilImage

from interface.widget.image_widget import ImageWidget
from interface.widget.mpl_widget import figure_widget
from tools import projection

KV = """
MDBoxLayout:
  orientation: 'vertical'

  MDBoxLayout:
    id: kv_layout
  MDSlider:
    min: 0
    max: 100
    value: 50
    on_value_normalized: app.update_plot(self.value_normalized)
"""


def kivy_core_texture(image: np.ndarray):
  image_pil = PilImage.fromarray(image)
  image_io = io.BytesIO()
  image_pil.save(image_io, format='png')
  image_io.seek(0)
  image_data = io.BytesIO(image_io.read())
  texture = KvCoreImage(image_data, ext='png').texture

  return texture


class Test(MDApp):

  def __init__(self, path, **kwargs):
    super().__init__(**kwargs)

    self._path = path
    self._kv_layout: MDBoxLayout = None
    self._kv_image: KvImage = None

    image = skimage.io.imread(path)
    image_alpha = image[:, :, -1]
    image = image[:, :, :3]
    image[np.logical_not(image_alpha)] = 0

    image = skimage.transform.rescale(image=image[:, :, :3],
                                      scale=0.05,
                                      multichannel=True)

    self.prj = projection.ImageProjection(image=image,
                                          angles=(0.0, 0.0, 0.0),
                                          viewing_angle=(42 * np.pi / 180.0))

  def build(self):
    return Builder.load_string(KV)

  def on_start(self):
    super().on_start()

    # matplotlib
    self._kv_layout = self.root.ids.kv_layout

    img = self.prj.project()

    # kivy image
    # self._kv_image = KvImage(source='')
    self._kv_image = ImageWidget(source='')
    self._kv_layout.add_widget(self._kv_image)

    img_uint = np.round(255 * img).astype('uint8')
    # self._kv_image.texture = kivy_core_texture(img_uint)
    # self._kv_image.reload()
    self._kv_image.update_image(img_uint)

  def update_plot(self, value):
    lr = value * 2.0 - 1.0
    self.prj.angles = (0, 0, lr)

    img = self.prj.project()

    # kivy image
    img_uint = np.round(255 * img).astype('uint8')
    # self._kv_image.texture = kivy_core_texture(img_uint)
    # self._kv_image.reload()
    self._kv_image.update_image(img_uint)


if __name__ == '__main__':
  path = (r'D:\repo\ThermalImage\Panorama\KICT\2020-11-10\result'
          r'\MainBldgBackLoc1PanTiltHandheld_plane\panorama_colormap.png')
  Test(path).run()
