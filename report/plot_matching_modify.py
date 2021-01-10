import utils

import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

import tools.preprocess

img1_path = (
    r'D:\repo\ThermalImage\Panorama\KICT\2020-11-10\extracted'
    r'\WaterResourceBldgPanTiltTripod\ir\original\IR_2020-11-10_0142.npy')
img2_path = (
    r'D:\repo\ThermalImage\Panorama\KICT\2020-11-10\extracted'
    r'\WaterResourceBldgPanTiltTripod\ir\original\IR_2020-11-10_0143.npy')

save_dir = utils.PRJ_DIR.joinpath('report', 'match')
if not save_dir.exists():
  save_dir.mkdir()

img_orig = np.load(img1_path)
img_warped = np.load(img2_path)

# trnsf = AffineTransform
trnsf = ProjectiveTransform

# img_orig_gray = rescale_intensity(img_orig)
# img_warped_gray = rescale_intensity(img_warped)

prep = tools.preprocess.PanoramaPreprocess(is_numeric=True,
                                           mask_threshold=(-20.0))
img_orig_gray, _ = prep(img_orig)
img_warped_gray, _ = prep(img_warped)

orb = ORB(n_keypoints=200, fast_threshold=0.05)

orb.detect_and_extract(img_orig)
coords_orig_subpix = orb.keypoints.copy().astype('int')

orb.detect_and_extract(img_warped)
coords_warped_subpix = orb.keypoints.copy().astype('int')
coords_warped = coords_warped_subpix

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.imshow(img_orig_gray, cmap='gray')
ax.plot(coords_orig_subpix[:, 1],
        coords_orig_subpix[:, 0],
        '+r',
        markersize=15,
        linestyle='None')
fig.savefig(save_dir.joinpath('features.jpg'), dpi=300)
# plt.show()
plt.close(fig)


def gaussian_weights(window_ext, sigma=1):
  y, x = np.mgrid[-window_ext:window_ext + 1, -window_ext:window_ext + 1]
  g = np.zeros(y.shape, dtype=np.double)
  g[:] = np.exp(-0.5 * (x**2 / sigma**2 + y**2 / sigma**2))
  g /= 2 * np.pi * sigma * sigma

  return g


def match_corner(coord, window_ext=5):
  r, c = np.round(coord).astype(np.intp)
  weights = gaussian_weights(window_ext, 3)
  if img_orig.ndim == 3:
    window_orig = img_orig[(r - window_ext):(r + window_ext + 1),
                           (c - window_ext):(c + window_ext + 1), :]
    # weight pixels depending on distance to center pixel
    weights = np.dstack((weights, weights, weights))
  else:
    window_orig = img_orig[(r - window_ext):(r + window_ext + 1),
                           (c - window_ext):(c + window_ext + 1)]

  # compute sum of squared differences to all corners in warped image
  SSDs = []
  for cr, cc in coords_warped:
    if img_warped.ndim == 3:
      window_warped = img_warped[(cr - window_ext):(cr + window_ext + 1),
                                 (cc - window_ext):(cc + window_ext + 1), :]
    else:
      window_warped = img_warped[(cr - window_ext):(cr + window_ext + 1),
                                 (cc - window_ext):(cc + window_ext + 1)]
    SSD = np.sum(weights * (window_orig - window_warped)**2)
    SSDs.append(SSD)

  # use corner with minimum SSD as correspondence
  min_idx = np.argmin(SSDs)
  return coords_warped_subpix[min_idx]


# find correspondences using simple weighted sum of squared differences
src = []
dst = []
for coord in coords_orig_subpix:
  src.append(coord)
  dst.append(match_corner(coord))
src = np.array(src)
dst = np.array(dst)

# estimate affine transform model using all coordinates
model = trnsf()
model.estimate(src, dst)

# robustly estimate affine transform model with RANSAC
model_robust, inliers = ransac((src, dst),
                               trnsf,
                               min_samples=3,
                               residual_threshold=2,
                               max_trials=100)

outliers = (inliers == False)

inlier_idxs = np.nonzero(inliers)[0]
outlier_idxs = np.nonzero(outliers)[0]

inlier_idxs_colstack = np.column_stack((inlier_idxs, inlier_idxs))
outlier_idxs_colstack = np.column_stack((outlier_idxs, outlier_idxs))
indices = np.vstack((inlier_idxs_colstack, outlier_idxs_colstack))

# all
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plot_matches(ax=ax,
             image1=img_orig_gray,
             image2=img_warped_gray,
             keypoints1=src,
             keypoints2=dst,
             matches=indices,
             keypoints_color='k',
             matches_color=None,
             only_matches=False,
             alignment='horizontal')
fig.savefig(save_dir.joinpath('all.jpg'), dpi=300)
plt.close(fig)

# match
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plot_matches(ax=ax,
             image1=img_orig_gray,
             image2=img_warped_gray,
             keypoints1=src,
             keypoints2=dst,
             matches=inlier_idxs_colstack,
             keypoints_color='k',
             matches_color='b',
             only_matches=False,
             alignment='horizontal')
fig.savefig(save_dir.joinpath('match.jpg'), dpi=300)
plt.close(fig)

# outlier
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plot_matches(ax=ax,
             image1=img_orig_gray,
             image2=img_warped_gray,
             keypoints1=src,
             keypoints2=dst,
             matches=outlier_idxs_colstack,
             keypoints_color='k',
             matches_color='r',
             only_matches=False,
             alignment='horizontal')
fig.savefig(save_dir.joinpath('outlier.jpg'), dpi=300)
plt.close(fig)
