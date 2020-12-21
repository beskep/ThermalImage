"""보고서 코드"""

import io
import subprocess

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


def fn1(image_path):
  binary_palette = subprocess.check_output(
      ['exiftool.exe', '-Palette', '-b', image_path])
  palette_array = np.array(list(binary_palette))
  ycrcb_palette = palette_array.reshape([1, -1, 3]).astype('uint8')
  rgb_palette = cv.cvtColor(ycrcb_palette, code=cv.COLOR_YCrCb2RGB)
  np.savetxt(fname='iron_colormap.txt', X=ycrcb_palette, fmt='%d')  # 저장
  plt.imshow(rgb_palette)
  plt.show()  # 시각화


#####
EXIFTOOL_PATH = None


def run_exif_tool(*args):
  """ExifTool을 통한 영상 태그 추출 함수"""
  args = (EXIFTOOL_PATH,) + args
  res = subprocess.check_output(args)
  return res


def get_exif_tags(image_path: str, *args):
  """문자 형태의 일반 태그를 추출하는 함수"""
  tag_byte = run_exif_tool(image_path, '-j', *args)
  tag = yaml.load(tag_byte.decode(), Loader=yaml.FullLoader)
  return tag


def get_exif_binary(image_path: str, tag):
  """binary 형태의 태그를 추출하는 함수"""
  res = run_exif_tool(tag, '-b', image_path)
  return res


def get_thermal_image(image_path):
  """적외선 raw data 추출"""
  image_byte = get_exif_binary(image_path, '-RawThermalImage')
  image_stream = io.BytesIO(image_byte)
  image_array = np.array(Image.open(image_stream))
  return image_array


def calculate_temperature(array, E, OD, RTemp, ATemp, IRWTemp, IRT, RH, PR1, PB,
                          PF, PO, PR2):
  """열화상 온도 계산 함수

  Parameters
  ----------
  array : ndarray
      Raw thermal image (infrared intensity)
  E : float
      Emission
  OD : flaot
      Object distance
  RTemp : float
      Apparent reflected temperature
  ATemp : flaot
      Atmospheric temperature
  IRWTemp : flaot
      Infrared window temperature
  IRT : flaot
      Infrared window transmission
  RH : flaot
      [description]
  PR1 : flaot
      camera calibration parameter
  PB : flaot
      camera calibration parameter
  PF : flaot
      camera calibration parameter
  PO : flaot
      camera calibration parameter
  PR2 : flaot
      camera calibration parameter

  Returns
  -------
  ndarray
      Thermal image
  """
  # 상수 항목
  ATA1 = 0.006569
  ATA2 = 0.01262
  ATB1 = -0.002276
  ATB2 = -0.00667
  ATX = 1.9

  # 렌즈 투과 보정
  window_emission = 1 - IRT
  window_reflection = 0

  # 공기 투과 보정
  h2o = ((RH / 100) * np.exp(1.5587 + 0.06939 * ATemp - 0.00027816 * ATemp**2 +
                             0.00000068455 * ATemp**3))
  tau1 = (ATX * np.exp(-np.sqrt(OD / 2) * (ATA1 + ATB1 * np.sqrt(h2o))) +
          (1 - ATX) * np.exp(-np.sqrt(OD / 2) * (ATA2 + ATB2 * np.sqrt(h2o))))
  tau2 = (ATX * np.exp(-np.sqrt(OD / 2) * (ATA1 + ATB1 * np.sqrt(h2o))) +
          (1 - ATX) * np.exp(-np.sqrt(OD / 2) * (ATA2 + ATB2 * np.sqrt(h2o))))

  # 주변 환경 복사
  raw_refl1 = PR1 / (PR2 * (np.exp(PB / (RTemp + 273.15)) - PF)) - PO
  raw_refl1_attn = (1 - E) / E * raw_refl1

  raw_atm1 = PR1 / (PR2 * (np.exp(PB / (ATemp + 273.15)) - PF)) - PO
  raw_atm1_attn = (1 - tau1) / E / tau1 * raw_atm1

  raw_wind = PR1 / (PR2 * (np.exp(PB / (IRWTemp + 273.15)) - PF)) - PO
  raw_wind_attn = window_emission / E / tau1 / IRT * raw_wind

  raw_refl2 = PR1 / (PR2 * (np.exp(PB / (RTemp + 273.15)) - PF)) - PO
  raw_refl2_attn = window_reflection / E / tau1 / IRT * raw_refl2

  raw_atm2 = PR1 / (PR2 * (np.exp(PB / (ATemp + 273.15)) - PF)) - PO
  raw_atm2_attn = (1 - tau2) / E / tau1 / IRT / tau2 * raw_atm2

  raw_obj = (array / E / tau1 / IRT / tau2 - raw_atm1_attn - raw_atm2_attn -
             raw_wind_attn - raw_refl1_attn - raw_refl2_attn)

  temperature = PB / np.log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15

  return temperature
