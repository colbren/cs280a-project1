# Builds on colorize_skel.py / coarse_to_fine.py and adds the automatic
# post-processing steps:
#   1. auto_crop
#   2. auto_contrast
#   3. white_balance
#   4. decorrelation_stretch
#   5. gradient_magnitude

import argparse

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from coarse_to_fine import imread_float, align_full

# 1. Automatic cropping.
#
# Two things need to be cropped off, and neither is a fixed margin:
#
#  - align() shifts each channel with np.roll, which is a circular
#    shift. It shifts a channel by n pixels wraps n pixels of content
#    from the opposite edge into view. Each channel gets a different
#    shift, so this wrap band is a hard requirement to crop, not
#    something to detect.
#  - beyond that, the scanned plate itself usually has its own border
#    (black, white, or a solid color) right up to a sharp transition into
#    the actual photograph. We look for that transition directly: within
#    a further band past the mandatory wrap margin, find the row/column
#    with the largest intensity jump, and only trust it as a real border
#    edge if it stands out clearly from the surrounding texture (a tall
#    peak relative to the band's median gradient) -- otherwise assume
#    there's no extra border there and stop at the mandatory margin.

def auto_crop(im, shifts=(), extra_border_frac=0.06, min_peak_ratio=2.0):
    """`shifts` is an iterable of (dx, dy) pairs -- one per aligned
    channel -- used to compute the mandatory wrap-around margin."""
    h, w = im.shape[:2]
    gray = im.mean(axis=2) if im.ndim == 3 else im

    abs_dx = [abs(dx) for dx, dy in shifts] or [0]
    abs_dy = [abs(dy) for dx, dy in shifts] or [0]
    margin_v, margin_h = max(abs_dy), max(abs_dx)

    row_grad = np.abs(np.diff(gray, axis=0)).mean(axis=1)  # length h-1
    col_grad = np.abs(np.diff(gray, axis=1)).mean(axis=0)  # length w-1

    def find_edge(grad, n, base_margin, from_end):
        band_len = max(1, int(n * extra_border_frac))
        g = grad[::-1] if from_end else grad
        band = g[base_margin:base_margin + band_len]
        if len(band) == 0:
            idx = base_margin
        else:
            peak, med = int(np.argmax(band)), np.median(band)
            stands_out = band[peak] > min_peak_ratio * (med + 1e-6)
            idx = base_margin + peak + 1 if stands_out else base_margin
        return (n - idx) if from_end else idx

    top = find_edge(row_grad, h, margin_v, from_end=False)
    bottom = find_edge(row_grad, h, margin_v, from_end=True)
    left = find_edge(col_grad, w, margin_h, from_end=False)
    right = find_edge(col_grad, w, margin_h, from_end=True)

    if bottom <= top or right <= left:
        return im  # degenerate case, bail out and keep the original

    return im[top:bottom, left:right]

# 2. Automatic contrasting.
#
# Linear rescale so the darkest value maps to 0 and the brightest to 1.
# We use a small percentile clip (instead of the literal min/max) so that
# a handful of outlier/noise pixels (dust, scratches) can't single-handedly
# compress the whole range -- this is the "more drastic/non-linear" variant
# the assignment suggests. The same lo/hi is used for every channel so
# color balance isn't disturbed by the contrast step itself.

def auto_contrast(im, low_pct=0.5, high_pct=99.5):
    lo = np.percentile(im, low_pct)
    hi = np.percentile(im, high_pct)
    if hi <= lo:
        return im
    return np.clip((im - lo) / (hi - lo), 0, 1)

# 3. Automatic white balance.
# Two simple illuminant estimators (Szeliski 2.3.2):

#  - gray_world:  assumes the *average* scene color should be neutral gray,
#                 so each channel is scaled to bring its mean in line with
#                 the average of all channel means.
def white_balance_gray_world(im):
    means = im.reshape(-1, 3).mean(axis=0)
    scale = means.mean() / (means + 1e-8)
    return np.clip(im * scale, 0, 1)

#  - white_patch: assumes the *brightest* scene point should be neutral
#                 white, so each channel is scaled to bring a high
#                 percentile of it up to the max of those percentiles.
#                 (A percentile instead of the literal max avoids letting
#                 one specular pixel decide the whole image's scale.)
def white_balance_white_patch(im, percentile=99):
    highs = np.percentile(im.reshape(-1, 3), percentile, axis=0)
    scale = highs.max() / (highs + 1e-8)
    return np.clip(im * scale, 0, 1)

def white_balance(im, method='gray_world'):
    if method == 'none':
        return im
    if method == 'gray_world':
        return white_balance_gray_world(im)
    if method == 'white_patch':
        return white_balance_white_patch(im)
    raise ValueError(f'unknown white balance method: {method}')

# 4. Better color mapping: decorrelation stretch.
#
# The R/G/B channels are three different glass-plate exposures, so their
# colors don't behave like a normal camera as they tend to be strongly correlated
# (mostly agreeing on brightness) with only a little real color information
# which is what makes images look washed out/monochrome-ish even after
# white balancing. Decorrelation stretch (Gillespie standard in remote 
# sensing/geology image enhancement) fixes this without any ground truth:
#   1. treat the 3 channels as 3 correlated random variables and compute
#      their 3x3 covariance matrix,
#   2. rotate into the covariance matrix's eigenbasis (PCA) -- this is the
#      basis in which the channels are decorrelated,
#   3. rescale each eigenbasis axis part-way toward a common (target)
#      standard deviation, so no single axis dominates,
#   4. rotate back into RGB.
# The net effect is a single 3x3 linear color transform, found from the
# image's own statistics, that amplifies the (small) color variation that
# was being drowned out by the (large) shared brightness variation.

def decorrelation_stretch(im, strength=0.15, target_std=None):
    pixels = im.reshape(-1, 3).astype(np.float64)
    mean = pixels.mean(axis=0)
    centered = pixels - mean

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-8, None)
    std = np.sqrt(eigvals)

    if target_std is None:
        target_std = std.mean()
    new_std = std * (1 - strength) + target_std * strength

    decorrelated = (centered @ eigvecs) / std
    stretched = decorrelated * new_std
    out = stretched @ eigvecs.T + mean

    return np.clip(out.reshape(im.shape), 0, 1)


def colorize(imname, window=8, refine_window=2, min_size=64, metric='ncc',
             features='raw', crop=True, white_balance_method='gray_world',
             color_matrix=False, contrast=True):
    """Full automatic pipeline: align -> crop -> white balance ->
    color mapping -> contrast. Returns (im_out, dy_g, dx_g, dy_r, dx_r)."""
    im = imread_float(imname)

    height = int(np.floor(im.shape[0] / 3.0))
    b = im[:height]
    g = im[height:2 * height]
    r = im[2 * height:3 * height]

    feature_fn = gradient_magnitude if features == 'gradient' else None

    dy_g, dx_g, ag = align_full(g, b, window=window, refine_window=refine_window,
                                 min_size=min_size, metric=metric,
                                 feature_fn=feature_fn)
    dy_r, dx_r, ar = align_full(r, b, window=window, refine_window=refine_window,
                                 min_size=min_size, metric=metric,
                                 feature_fn=feature_fn)

    im_out = np.dstack([ar, ag, b]).astype(np.float64)

    if crop:
        im_out = auto_crop(im_out, shifts=[(dx_g, dy_g), (dx_r, dy_r)])
    if white_balance_method != 'none':
        im_out = white_balance(im_out, white_balance_method)
    if color_matrix:
        im_out = decorrelation_stretch(im_out)
    if contrast:
        im_out = auto_contrast(im_out)

    return im_out, dy_g, dx_g, dy_r, dx_r

# 5. Better features for alignment: gradients instead of raw intensity.
#
# Prokudin-Gorskii's three plates were exposed through different filters at
# different times, so a channel's *brightness* isn't reliable to match. Edges
# and gradients are more filter/exposure invariant, since they mostly
# come from scene structure, not filter color. 
#
# We approximate the gradient with simple central differences (np.gradient) 
# and use the magnitude as the feature that gets matched instead of intensity.

def gradient_magnitude(im):
    gy, gx = np.gradient(im)
    return np.sqrt(gx ** 2 + gy ** 2)

def main():
    parser = argparse.ArgumentParser(description='Bells & whistles colorization pipeline.')
    parser.add_argument('--imname', type=str, default='./emir.tif',
                         help='path to the glass plate image')
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--refine-window', type=int, default=2)
    parser.add_argument('--min-size', type=int, default=64)
    parser.add_argument('--metric', type=str, default='ncc', choices=['ncc', 'ssd'])
    parser.add_argument('--features', type=str, default='gradient',
                         choices=['raw', 'gradient'],
                         help='align on raw pixel intensity or gradient magnitude')
    parser.add_argument('--no-crop', action='store_false', dest='crop')
    parser.add_argument('--white-balance', type=str, default='gray_world',
                         choices=['none', 'gray_world', 'white_patch'])
    parser.add_argument('--color-matrix', action='store_true',
                         help='apply decorrelation stretch color remapping')
    parser.add_argument('--no-contrast', action='store_false', dest='contrast')
    parser.add_argument('--out', type=str, default='./out_bw.jpg')
    args = parser.parse_args()

    im_out, dy_g, dx_g, dy_r, dx_r = colorize(
        args.imname, window=args.window, refine_window=args.refine_window,
        min_size=args.min_size, metric=args.metric, features=args.features,
        crop=args.crop, white_balance_method=args.white_balance,
        color_matrix=args.color_matrix, contrast=args.contrast)

    print(f'G displacement (x, y): ({dx_g}, {dy_g})')
    print(f'R displacement (x, y): ({dx_r}, {dy_r})')

    plt.figure(figsize=(8, 8))
    plt.imshow(im_out)
    plt.title('Colorized (bells & whistles)')
    plt.axis('off')
    plt.show()

    out_uint8 = np.clip(im_out * 255.0, 0, 255).astype(np.uint8)
    out_bgr = cv.cvtColor(out_uint8, cv.COLOR_RGB2BGR)
    cv.imwrite(args.out, out_bgr)

if __name__ == '__main__':
    main()
