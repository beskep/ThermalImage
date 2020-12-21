from pathlib import Path
from typing import Union

import yaml

import tools.ivimages as ivi


class StitchingOption:

  def __init__(self, stream):
    self._options = yaml.load(stream, Loader=yaml.FullLoader)

  @property
  def options(self):
    return self._options.copy()

  @property
  def case_options(self):
    return self.options['cases'].copy()

  def cases(self):
    return list(self.case_options.keys())

  def directions(self, case):
    return [x for x in self.case_options[case].keys() if x != 'subdir']

  @classmethod
  def elements_generator(cls, elements: list):
    for element in elements:
      if isinstance(element, dict):
        if 'range' not in element:
          raise SyntaxError
        yield list(range(*element['range']))

      elif isinstance(element, list):
        yield element

      else:
        raise SyntaxError

  def elements(self, case, direction):
    return self.elements_generator(self.case_options[case][direction])

  def walk(self):
    cases = self.cases()
    for case in cases:
      directions = self.directions(case)

      for direction in directions:
        yield self.elements(case, direction)

  def _get_loader(self, cls, case=None):
    clsdict = {
        'ImageLoader': ivi.ImageLoader,
        'IVImagesLoader': ivi.IVImagesLoader
    }
    if cls not in clsdict:
      raise ValueError

    if cls not in self.options:
      loader = None
    else:
      kwargs: dict = self.options[cls].copy()
      if case is not None:
        if 'loader' in self.case_options[case]:
          options: dict = self.case_options[case]['loader'].copy()
          subdir = options.pop('subdir', case)

          for key, value in options.items():
            kwargs[key] = value
        else:
          subdir = case

        for k, v in kwargs.items():
          if k.endswith('dir'):
            kwargs[k] = Path(v).joinpath(subdir).as_posix()

      loader = clsdict[cls](**kwargs)

    return loader

  def image_loader(self, case=None) -> Union[None, ivi.ImageLoader]:
    return self._get_loader('ImageLoader', case)

  def ivimages_loader(self, case=None) -> Union[None, ivi.IVImagesLoader]:
    return self._get_loader('IVImagesLoader', case)
