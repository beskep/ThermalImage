from pathlib import Path

import utils

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.progress import track
from skimage import exposure, feature
from skimage.exposure.exposure import rescale_intensity
from skimage.io import imsave
from skimage.restoration import denoise_bilateral

import tools.cmap
import tools.ivimages as ivi

if __name__ == '__main__':
  sns.set_context('paper')

  data_dir = Path(r'D:\repo\ThermalImage\Panorama\KICT\2020-11-10'
                  r'\extracted\MainBldgBackLoc1PanTiltTripod\ir\original')
  res_dir: Path = utils.ROOT_DIR.joinpath('result/preprocessing_noise_features')
  if not res_dir.exists():
    res_dir.mkdir()

  cmap_path = utils.ROOT_DIR.joinpath('data/cmap/iron_colormap_rgb.txt')
  cmap = tools.cmap.Colormap.from_uint8_text(cmap_path)

  loader = ivi.ImageLoader(img_dir=data_dir, img_ext='npy')

  # detector = feature.ORB(n_keypoints=500)

  df = {'file': [], 'original_features_count': [], 'features_count': []}
  files = loader.files.copy()

  for file in track(files[-2:]):
    # # 대상 정함
    # if '018' not in file.as_posix():
    #   continue

    stem = file.stem
    arr = loader.read(file)

    arr_mask = arr.copy()
    arr_mask[arr < -20.0] = -20

    # equalized = exposure.equalize_hist(arr)
    equalized_mask = exposure.equalize_hist(arr_mask)

    # bilateral = denoise_bilateral(equalized, sigma_color=0.05, sigma_spatial=10)
    bilateral_mask = denoise_bilateral(equalized_mask,
                                       sigma_color=0.05,
                                       sigma_spatial=10)

    imsave(fname=res_dir.joinpath(f'{stem}.png'),
           arr=rescale_intensity(arr, out_range='uint8'))
    imsave(fname=res_dir.joinpath(f'{stem}_preprocess.png'),
           arr=rescale_intensity(bilateral_mask, out_range='uint8'))

    # detector.detect_and_extract(arr)
    # keypoints_original = detector.keypoints
    # # features_arr = orb.descriptors

    # detector.detect_and_extract(bilateral_mask)
    # keypoints = detector.keypoints

    # df['file'].append(stem)
    # df['original_features_count'] = keypoints_original.shape[0]
    # df['features_count'] = keypoints.shape[0]

    # fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    # axes[0].imshow(arr)
    # axes[0].scatter(keypoints_original[:, 1],
    #                 keypoints_original[:, 0],
    #                 marker='D')
    # axes[1].imshow(bilateral_mask)
    # axes[1].scatter(keypoints[:, 1], keypoints[:, 0], marker='D')

    # fig.savefig(res_dir.joinpath(stem).with_suffix('.png'))
    # plt.close(fig)

  # df = pd.DataFrame(df)
  # df.to_csv(res_dir.joinpath('df.csv'))
