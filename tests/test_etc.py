from pathlib import Path
from pprint import pprint

import pytest
import yaml

import utils
from tools import stitching_option


def test_yaml():
  file = dirs.ROOT_DIR.joinpath('data', 'KICT-2020-11-10.yaml')
  assert file.exists()

  with open(file, 'r') as f:
    loc = yaml.load(f, Loader=yaml.FullLoader)

  assert loc

  pprint(loc)

  assert 'MainBldgBackLoc1PanTiltTripod' in loc['cases']


option_text = """
ImageLoader:
    img_dir: '.'
    img_ext: 'test_ext'
    pattern:
IVImagesLoader:
    vis_dir: .
    ir_dir: .
    vis_ext: vis_ext
    ir_ext: ir_ext
    vis_pattern: vis_pattern
cases:
    Test1:
        all:
            - range: [21, 38]
    Test2:
        all:
            - range: [2, 20]
        vertical:
            - [7, 8, 19]
            - [9, 18]
        horizontal:
            - range: [2, 8]
            - range: [8, 15]
            - range: [15, 20]
    """


def test_stitching_option_lodaer():
  option = stitching_option.StitchingOption(option_text)
  cur_loc = Path('.').resolve()

  image_loader = option.image_loader()
  assert image_loader.dir == cur_loc
  assert image_loader.ext == 'test_ext'
  assert image_loader.pattern is None

  ivimages_loader = option.ivimages_loader()
  assert ivimages_loader.vis.dir == cur_loc
  assert ivimages_loader.ir.dir == cur_loc
  assert ivimages_loader.vis.pattern == 'vis_pattern'
  assert ivimages_loader.ir.pattern is None

  empty_option_text = """
    cases:
        Test1:
    """
  empty_option = stitching_option.StitchingOption(empty_option_text)
  assert empty_option.image_loader() is None
  assert empty_option.ivimages_loader() is None


def test_stitching_option_cases():

  option = stitching_option.StitchingOption(option_text)
  assert all(x in option.cases() for x in ['Test1', 'Test2'])

  directions = {'all', 'vertical', 'horizontal'}
  assert set(option.case_options['Test2'].keys()) == directions

  test2_horizontal = [
      list(range(2, 8)),
      list(range(8, 15)),
      list(range(15, 20))
  ]

  for opt1, opt2 in zip(option.elements('Test2', 'horizontal'),
                        test2_horizontal):
    assert opt1 == opt2

  assert list(option.elements('Test2', 'horizontal')) == test2_horizontal


if __name__ == "__main__":
  pytest.main(['-v', '-k', 'test_stitching_option'])
