import threading
from pathlib import Path
from typing import List

import utils

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage.io as skio
import yaml
from kivy.clock import mainthread
from kivy.metrics import dp
from kivy.uix.widget import WidgetException
from loguru import logger
from skimage.exposure import rescale_intensity
from skimage.transform import resize as skresize

import flir
import interface.widget.text_field
import tools.cmap
import tools.imagetools as imt
import tools.ivimages as ivi
import tools.preprocess as prep
from interface import kvtools
from interface.widget.mpl_widget import figure_widget
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.imagelist import SmartTileWithLabel
from kivymd.uix.selectioncontrol import MDCheckbox
from kivymd.uix.snackbar import Snackbar
from tools.stitcher import Stitcher, StitchingImages

_FONT_STYLES = ('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'Subtitle1', 'Subtitle2',
                'Body1', 'Body2', 'Button', 'Caption', 'Overline')


def get_cmap():
  config = utils.get_config()
  cmap_path = utils.PRJ_DIR.joinpath(config['cmap_path'])
  cmap = tools.cmap.Colormap.from_uint8_text(cmap_path)

  return cmap


def threadwrap(fn):

  def wrapper(*args, **kwargs):
    th = threading.Thread(target=fn, args=args, kwargs=kwargs)
    th.daemon = True
    th.start()

    return th

  return wrapper


def pbar(fn):

  def fn_(self, *args, **kwargs):
    fn(self, *args, **kwargs)
    self.progressbar_state(start=False)

  def wrapper(self, *args, **kwargs):
    self.progressbar_state(start=True)

    th = threading.Thread(target=fn_, args=((self,) + args), kwargs=kwargs)
    th.daemon = True
    th.start()

    return th

  return wrapper


class CheckOnlyBox(MDCheckbox):

  def on_touch_down(self, touch):
    if self.state == 'normal':
      super().on_touch_down(touch)


class PanoramaWidget(MDBoxLayout):
  pass


class PanoramaApp(MDApp):
  _options_dict = {
      'file': ['image', 'flir_ir', 'flir_vis'],
      'contrast': ['none', 'equalization', 'normalization'],
      'noise': ['none', 'bilateral', 'gaussian'],
      'perspective': ['panorama', 'scan'],
      'warp': ['plane', 'compressed_plane', 'panini', 'spherical'],
      'numerical': ['compose_scale', 'work_scale', 'mask_threshold']
  }

  def __init__(self, **kwargs):
    self.root = None
    self.built = None

    super().__init__(**kwargs)

    for fs in _FONT_STYLES:
      self.theme_cls.font_styles[fs][0] = 'NotoSansKR'

    # self.theme_cls.theme_style = 'Dark'
    self.theme_cls.primary_palette = 'Red'
    self.theme_cls.primary_hue = '400'

    self.title = 'Facade Thermal Image Panorama'
    # todo: title 아이콘 교체

    self.file_manager = MDFileManager()
    self.file_manager.selector = 'multi'

    self.file_manager.exit_manager = lambda x: self.file_manager.close()
    self.file_manager.select_path = self.select_path

    self._snackbar = Snackbar()
    self._snackbar.duration = 2.5
    self._snackbar.font_size = dp(16)

    self._selected_files: List[Path] = []
    self._save_dir = None
    self._result = None

    try:
      cmap = get_cmap()
    except (IOError, LookupError):
      logger.exception('FLIR 컬러맵 로드에 실패했습니다. magma 컬러맵을 사용합니다.')
      cmap = sns.color_palette(palette='magma', as_cmap=True)

    self._cmap = cmap

  def build(self):
    return PanoramaWidget()

  def on_start(self):
    gap = dp(10)
    self._snackbar.snackbar_x = gap
    self._snackbar.snackbar_y = gap
    self._snackbar.size_hint_x = (self.root.width - 2 * gap) / self.root.width

    file_rail = self.root.ids.file
    file_rail.trigger_action()

  def manual_build(self):
    """run() 실행 전 build

    Raises
    ------
    WidgetException
        widget 생성 실패
    """
    root = self.build()

    if not root:
      msg = '{} failed to build'.format(__class__.__name__)
      raise WidgetException(msg)

    self.root = root
    self.built = True

  @mainthread
  def progressbar_state(self, start: bool):
    self.root.ids.pb.type = 'indeterminate'
    if start:
      self.root.ids.pb.start()
    else:
      self.root.ids.pb.stop()

  @mainthread
  def progressbar_value(self, value):
    self.root.ids.pb.type = 'determinate'
    self.root.ids.pb.value = value

  @mainthread
  def show_snackbar(self, message, duration=None):
    self._snackbar.text = message
    logger.debug('Snackbar message: {}', message)

    if duration:
      self._snackbar.duration = duration

    try:
      self._snackbar.open()
    except WidgetException:
      pass

  def open_file_manager(self, mode: str):
    if mode not in ('load', 'load_all', 'save'):
      raise ValueError

    if mode.startswith('load'):
      # ext = ['.jpg', '.jpeg', '.png', '.tiff', '.xlsx', '.csv', '.npy']
      ext = self.get_ext_option()
      assert ext is not None
      ext = ext + [x.upper() for x in ext]
    else:
      ext = []

    if mode == 'load':
      selector = 'multi'
    else:
      selector = 'folder'

    self.file_manager.mode = mode
    self.file_manager.ext = ext
    self.file_manager.selector = selector
    self.file_manager.show('\\')

    if not self.file_manager._window_manager_open:
      self.file_manager._window_manager.open()
      self.file_manager._window_manager_open = True

  def select_path(self, path):
    if self.file_manager.mode.startswith('load'):
      if self.file_manager.mode == 'load':
        assert isinstance(path, list)

        self._selected_files = [Path(x) for x in path]
      else:
        # mode == 'load_all'
        path = Path(path)
        assert path.is_dir()

        ext = self.get_ext_option()
        self._selected_files = [
            x for x in path.glob('*')
            if x.is_file() and (x.suffix.lower() in ext)
        ]

      self.show_selected_images(self._selected_files)
    else:
      path = Path(path)
      assert path.is_dir()
      self._save_dir = path
      self.save_image()

    self.file_manager.close()

  def clear_selected_images(self):
    self._selected_files = None
    self.show_selected_images(files=[])

  def show_selected_images(self, files: list):
    view: MDGridLayout = self.root.ids.file_screen.ids.images_view
    view.clear_widgets()

    files_count = len(files)
    if files_count:
      if files_count < 4:
        ncols = 3
      elif files_count == 4:
        ncols = 2
      else:
        nrows = max(round((files_count * 9.0 / 16.0)**0.5), 1)
        ncols = int(np.ceil(files_count / nrows))

      view.cols = ncols

      for file in files:
        tile = SmartTileWithLabel()
        tile.source = file.as_posix()
        tile.text = file.name

        view.add_widget(tile)

  def get_options(self):
    option_screen = self.root.ids.option_screen
    options = dict()

    for key, values in self._options_dict.items():
      panel = getattr(option_screen.ids, key)

      if key == 'numerical':
        for value in values:
          scale = getattr(panel.ids, value).value()
          options[value] = scale
      else:
        for value in values:
          if getattr(panel.ids, value).state == 'down':
            options[key] = value
            break

    # note: csv 등을 load한 경우, self.execute()에서 옵션 변경함
    options['is_numeric'] = options['file'] == 'flir_ir'

    for key, value in options.items():
      if value == 'none':
        options[key] = None

    return options

  def get_ext_option(self):
    # fixme: 막 만듬
    file_screen = self.root.ids.file_screen
    ext = {
        'ext_image': ['.jpg', '.jpeg', '.png', '.tiff'],
        'ext_array': ['.xlsx', '.csv'],
        'ext_npy': ['.npy'],
    }
    res = None
    for key, value in ext.items():
      if getattr(file_screen.ids, key).state == 'down':
        res = value
        break

    return res

  @staticmethod
  def _read_fn(option: str):
    if option not in ['image', 'flir_ir', 'flir_vis']:
      raise ValueError

    if option == 'image':
      fn = ivi.ImageLoader.read
    else:
      extractor = flir.FlirExtractor()

      if option == 'flir_ir':
        fn = extractor.extract_ir
      else:
        fn = extractor.extract_vis

    return fn

  def read_files(self, option: str):
    files_count = len(self._selected_files)
    if not files_count:
      raise ValueError

    _read = self._read_fn(option)
    flag_flir = option.startswith('flir')

    images = []
    for idx, file in enumerate(self._selected_files):
      try:
        image = _read(file)
      except flir.FlirExifNotFoundError:
        image = None
        msg = '파일 형식 오류 (FLIR Exif 정보 없음): {}'.format(file.name)
      except Exception:
        image = None
        logger.exception('파일 불러오기 실패: {}', file.name)

      else:
        msg = None

        # fixme: 매번 FLIR 해석하기 귀찮아서 임시로 저장하게 함
        if option == 'flir_ir':
          npy_path = file.with_suffix('.npy')
          np.save(file=npy_path.as_posix(), arr=np.round(image, 3))

      if image is not None:
        images.append(image)

        if flag_flir:
          msg = '파일 로드 ({}/{}): {}'.format(idx + 1, files_count, file.name)
          self.show_snackbar(msg, duration=1.0)
      else:
        logger.error(msg)
        self.show_snackbar(msg)
        images = None
        break

    return images

  @pbar
  def execute(self):
    if len(self._selected_files) <= 1:
      if len(self._selected_files) == 0:
        msg = '선택된 파일이 없습니다'
      else:
        msg = '파일이 하나만 선택되었습니다'

      logger.debug(msg)
      self.show_snackbar(msg)
      return

    options = self.get_options()

    # 영상 읽기
    images = self.read_files(option=options['file'])
    if images is None:
      return

    self.show_snackbar('파일 로드 완료', duration=1.0)

    # bug: compose aspect가 제대로 작동 안함
    # 화질은 낮아지는데 결과물 해상도가 그대로인거 보면 다시 upscale하는지도...

    # 파노라마 생성
    panorama, mask, graph, indices = self.make_panorama(images=images,
                                                        options=options)
    if panorama is None:
      self.show_snackbar('파노라마 생성 실패', duration=4)
      return
    else:
      self.show_snackbar('파노라마 생성')

    # 열화상 정보만 있는 구역 (bounding box) crop
    x1, x2, y1, y2 = imt.mask_bbox(mask=mask, morphology_open=True)
    if (x1, y1) != (0, 0) or mask.shape != (y2, x2):
      logger.debug('Image cropped: [{:d}:{:d}, {:d}:{:d}]', x1, x2, y1, y2)
      panorama = panorama[y1:y2, x1:x2]
      mask = mask[y1:y2, x1:x2]

    self._result = {
        'panorama': panorama,
        'mask': mask,
        'graph': graph,
        'indices': indices,
        'images': images,
        'options': options
    }

    # 결과 표시
    self.show_image(image=panorama, mask=mask)

    self.show_snackbar('파노라마 생성 완료')

  def make_panorama(self, images, options):
    # 대상 영상, 전처리 설정
    stitching_images = StitchingImages(arrays=images, preprocess=None)

    if stitching_images.ndim == 2:
      options['is_numeric'] = True

    preprocess = prep.PanoramaPreprocess(
        is_numeric=options['is_numeric'],
        mask_threshold=options['mask_threshold'],
        constrast=options['contrast'],
        noise=options['noise'])

    stitching_images.set_preprocess(preprocess)

    # stitcher 설정
    if options['perspective'] == 'scan':
      warp = 'affine'
    else:
      warp = options['warp']
      if warp == 'panini':
        warp = 'paniniA1.5B1'
      elif warp == 'compressed_plane':
        warp = 'compressedPlaneA1.5B1'

    stitcher = Stitcher(mode=options['perspective'],
                        compose_scale=options['compose_scale'],
                        work_scale=options['work_scale'])
    stitcher.warper_type = warp

    # 파노라마 생성
    try:
      panorama, mask, graph, indices = stitcher.stitch(images=stitching_images,
                                                       masks=None)
    except Exception:
      panorama, mask, graph, indices = None, None, None, None
      logger.exception('파노라마 생성 실패')

    # if panorama is None and 'A1.5B1' in warp:
    #   # warper 바꾸고 재시도
    #   warp = warp.replace('A1.5', 'A2')
    #   stitcher.warper_type = warp

    #   try:
    #     panorama, mask, graph, indices = stitcher.stitch(
    #         images=stitching_images, masks=None)
    #   except Exception:
    #     panorama, mask, graph, indices = None, None, None, None
    #     logger.exception('파노라마 생성 실패')

    if (panorama is not None and options['is_numeric'] and
        options['mask_threshold']):
      # 경계면에 thresholding 오류 보정
      error_mask = panorama < options['mask_threshold']
      fill_value = panorama[np.logical_not(mask)].ravel()[0]
      panorama[error_mask] = fill_value

    return panorama, mask, graph, indices

  def colorize(self, image):
    rescaled_iamge = rescale_intensity(image=image, out_range=(0.0, 1.0))
    color_image = self._cmap(rescaled_iamge)

    return color_image

  @mainthread
  def show_image(self, image, mask=None):
    width = 2000  # TODO option
    if image.shape[1] > width:
      output_shape = (int(image.shape[0] * width / image.shape[1]), width)
      image = skresize(image=image,
                       output_shape=output_shape,
                       anti_aliasing=False)
      mask = skresize(image=mask,
                      output_shape=output_shape,
                      anti_aliasing=False)

    fig, ax = plt.subplots(1, 1)

    if image.ndim == 3:
      image = rescale_intensity(image)
      if mask is not None:
        image[np.logical_not(mask)] = np.nan
      ax.imshow(image)
    else:
      if mask is not None:
        image[np.logical_not(mask)] = np.nan
      sns.heatmap(data=image, cmap=self._cmap, ax=ax, cbar=True)

    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    widget = figure_widget(fig)

    layout: MDBoxLayout = self.root.ids.run_screen.ids.view
    layout.clear_widgets()
    layout.add_widget(widget)

  def panorama_info(self, panorama, options):
    selected = [self._selected_files[x] for x in self._result['indices']]
    unselected = [x for x in self._selected_files if x not in selected]

    panorama_info = {
        'image': {
            'min':
                float(panorama.min()),
            'max':
                float(panorama.max()),
            'selected_files': [x.name for x in selected],
            'unselected_files':
                ([x.name for x in unselected] if unselected else None)
        },
        'option': options
    }

    return panorama_info

  def save_image(self):
    save_dir: Path = self._save_dir
    assert save_dir and save_dir.exists()
    if self._result is None:
      msg = '파노라마가 생성되지 않았습니다.'
      logger.debug(msg)
      self.show_snackbar(msg)
      return

    panorama: np.ndarray = self._result['panorama']
    mask = self._result['mask']
    options = self._result['options']

    # 파노라마 저장 (열화상의 경우엔 흑백으로 저장)
    image_uint = panorama.copy()
    image_uint = rescale_intensity(panorama, out_range='uint16')
    image_uint[mask] = 0
    skio.imsave(fname=save_dir.joinpath('panorama.png').as_posix(),
                arr=image_uint)

    # 파노라마 생성 정보
    panorama_info = self.panorama_info(panorama=panorama, options=options)
    with open(save_dir.joinpath('panorama information.txt'), 'w') as f:
      yaml.dump(panorama_info, f, indent=4)

    # 마스크
    mask_uint = rescale_intensity(mask, out_range='uint8')
    skio.imsave(fname=save_dir.joinpath('panorama_mask.png').as_posix(),
                arr=mask_uint)

    if options['is_numeric']:
      # 컬러맵 입힌 이미지
      image_color = self.colorize(panorama)
      assert image_color.ndim == 3
      assert image_color.shape[2] == 4
      image_color = rescale_intensity(image_color, out_range='uint8')
      image_color[:, :, 3] = mask_uint

      skio.imsave(fname=save_dir.joinpath('panorama_colormap.png').as_posix(),
                  arr=image_color)

      # 행렬
      panorama[np.logical_not(mask)] = np.nan
      np.savetxt(fname=save_dir.joinpath('panorama.csv'),
                 X=panorama,
                 fmt='%.2f',
                 delimiter=',')

    if options['file'].startswith('flir'):
      # 추출한 이미지 저장
      for image, file in zip(self._result['images'], self._selected_files):
        path = save_dir.joinpath(file.stem)

        if options['is_numeric']:
          np.savetxt(fname=path.with_suffix('.csv'),
                     X=image,
                     fmt='%.2f',
                     delimiter=',')
        else:
          skio.imsave(fname=path.with_suffix('.png').as_posix(), arr=image)


def main():
  font_dir = utils.DATA_DIR.joinpath('fonts')
  font_regular = font_dir.joinpath('NotoSansCJKkr-Medium.otf')
  font_bold = font_dir.joinpath('NotoSansCJKkr-Bold.otf')

  kvtools.register_font(name='NotoSansKR', fn_regular=font_regular.as_posix())
  kvtools.register_font(name='NotoSansKRBold', fn_regular=font_bold.as_posix())
  kvtools.set_window_size(size=(1280, 720))
  kvtools.config()

  kv_dir = utils.SRC_DIR.joinpath('interface/kv')
  kv_files = kv_dir.glob('*.kv')
  for file in kv_files:
    kvtools.load_kv(file)

  app = PanoramaApp()
  app.manual_build()

  app.run()


if __name__ == "__main__":
  main()
