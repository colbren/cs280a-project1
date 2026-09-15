import argparse
import time

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from colorize_skel import align, crop_border

def imread_float(imname):
    """Read an image and convert to float32 in [0, 1], regardless of
    whether the source is an 8-bit JPG or a 16-bit TIFF."""
    im = cv.imread(imname, cv.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f'Could not read image: {imname}')
    if im.ndim == 3:
        # shouldn't normally happen for these stacked-grayscale scans,
        # but fall back to grayscale just in case
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    if im.dtype == np.uint8:
        max_val = 255.0
    elif im.dtype == np.uint16:
        max_val = 65535.0
    else:
        max_val = float(im.max()) if im.max() > 0 else 1.0

    return im.astype(np.float32) / max_val


def downsample2(im):
    """Downsample a 2D image by a factor of 2 using 2x2 block averaging
    (a simple box filter + subsample, done by hand -- no library resize).
    """
    h, w = im.shape
    h2, w2 = h - (h % 2), w - (w % 2)   # trim to even dims
    im = im[:h2, :w2]
    # reshape into 2x2 blocks and average over them -- fully vectorized,
    # no per-pixel Python loop
    return im.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))


def align_pyramid(im, ref, window=8, refine_window=2, min_size=64,
                   metric='ncc', border_frac=0.1, feature_fn=None):
    """Coarse-to-fine search for the (dy, dx) displacement that best
    aligns `im` onto `ref`.

    Recursively downsamples both images by 2 until they're small enough
    to brute-force (min_size), then works back down the pyramid. At each
    finer level the previous level's estimate is doubled and refined
    with a small local search.

    `feature_fn`, if given, is passed through to align() at every level
    so scoring can be done on which is useful when channels differ in
    brightness rather than just being shifted copies of each other.

    Returns (dy, dx) at the resolution of the images passed in.
    """
    h, w = im.shape

    # base case: coarsest level, do the full exhaustive search
    if min(h, w) <= min_size:
        dy, dx, _ = align(im, ref, window=window, metric=metric,
                           border_frac=border_frac, feature_fn=feature_fn)
        return dy, dx

    # recurse on the downsampled level first
    dy_coarse, dx_coarse = align_pyramid(
        downsample2(im), downsample2(ref),
        window=window, refine_window=refine_window,
        min_size=min_size, metric=metric, border_frac=border_frac,
        feature_fn=feature_fn)

    # scale the coarse estimate up to this level, apply it, then refine
    # with a small window to correct for rounding error
    dy_est, dx_est = dy_coarse * 2, dx_coarse * 2
    im_shifted = np.roll(im, (dy_est, dx_est), axis=(0, 1))
    dy_ref, dx_ref, _ = align(im_shifted, ref, window=refine_window,
                               metric=metric, border_frac=border_frac,
                               feature_fn=feature_fn)

    return dy_est + dy_ref, dx_est + dx_ref


def align_full(im, ref, window=8, refine_window=2, min_size=64,
               metric='ncc', border_frac=0.1, feature_fn=None):
    """Convenience wrapper: runs align_pyramid and also returns the
    aligned image, matching the (dy, dx, aligned) signature of align()."""
    dy, dx = align_pyramid(im, ref, window=window, refine_window=refine_window,
                            min_size=min_size, metric=metric,
                            border_frac=border_frac, feature_fn=feature_fn)
    aligned = np.roll(im, (dy, dx), axis=(0, 1))
    return dy, dx, aligned


def main():
    parser = argparse.ArgumentParser(description='Coarse-to-fine pyramid channel alignment.')
    parser.add_argument('--imname', type=str, default='./church.tif',
                         help='path to the glass plate image')
    parser.add_argument('--window', type=int, default=8,
                         help='exhaustive search window at the coarsest pyramid level')
    parser.add_argument('--refine-window', type=int, default=2,
                         help='local refinement window at each finer level')
    parser.add_argument('--min-size', type=int, default=64,
                         help='stop downsampling once the shorter side is <= this')
    parser.add_argument('--metric', type=str, default='ncc', choices=['ncc', 'ssd'],
                         help='matching metric used to score candidate shifts')
    parser.add_argument('--out', type=str, default='./out_fname.jpg',
                         help='path to save the output color image (JPG)')
    args = parser.parse_args()

    t0 = time.time()

    im = imread_float(args.imname)

    # compute the height of each part (just 1/3 of total)
    height = int(np.floor(im.shape[0] / 3.0))

    # separate color channels (order in the scan is B, G, R top to bottom)
    b = im[:height]
    g = im[height:2 * height]
    r = im[2 * height:3 * height]

    dy_g, dx_g, ag = align_full(g, b, window=args.window,
                                 refine_window=args.refine_window,
                                 min_size=args.min_size, metric=args.metric)
    dy_r, dx_r, ar = align_full(r, b, window=args.window,
                                 refine_window=args.refine_window,
                                 min_size=args.min_size, metric=args.metric)

    # displacement vectors, reported as (x, y)
    print(f'G displacement (x, y): ({dx_g}, {dy_g})')
    print(f'R displacement (x, y): ({dx_r}, {dy_r})')
    print(f'elapsed: {time.time() - t0:.1f}s')

    im_out = np.dstack([ar, ag, b])

    plt.figure(figsize=(8, 8))
    plt.imshow(im_out)
    plt.title('Colorized')
    plt.axis('off')
    plt.show()

    out_uint8 = np.clip(im_out * 255.0, 0, 255).astype(np.uint8)
    out_bgr = cv.cvtColor(out_uint8, cv.COLOR_RGB2BGR)

    # save as JPG (not TIFF) to keep disk usage down
    cv.imwrite(args.out, out_bgr)


if __name__ == '__main__':
    main()
