import os
from pathlib import Path

import numpy as np
import pytest

import utils
from flir.extract import FlirExtractor

extractor = FlirExtractor()
image_dir = dirs.ROOT_DIR.joinpath('data')
images = ['FLIR/IR_2020-10-19_0017.jpg']


def test_extractor_init():
  assert os.path.exists(extractor.extractor.exiftool_path)


@pytest.mark.parametrize('image', images)
def test_process_image(image):
  path = image_dir.joinpath(image)
  extractor.process_image(path)
  assert extractor.extractor.thermal_image_np is not None
  assert extractor.extractor.rgb_image_np is not None


@pytest.mark.parametrize('image', images)
def test_extract_data(image):
  path = image_dir.joinpath(image)
  ir_array, vis_array = extractor.extract_data(path)

  assert isinstance(ir_array, np.ndarray)
  assert isinstance(vis_array, np.ndarray)
  assert vis_array.dtype == np.uint8


@pytest.mark.parametrize('image', images)
def test_write_data(image):
  path = image_dir.joinpath(image)
  res_dir = Path(__file__).joinpath('../../result/test').resolve()
  if not res_dir.exists():
    res_dir.mkdir(parents=True)

  ir_path = res_dir.joinpath('flir_ir.npy')
  vis_path = res_dir.joinpath('flir_vis.png')

  extractor.write_data(ir_path=ir_path, vis_path=vis_path, image_path=path)

  assert ir_path.exists()
  assert vis_path.exists()


def test_extract_ir_non_flir_file():
  path = dirs.ROOT_DIR.joinpath('data/opencv_sample/left01.jpg')

  extractor.extract_ir(path)


if __name__ == '__main__':
  # pytest.main()
  test_extract_ir_non_flir_file()
