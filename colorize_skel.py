import argparse

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def crop_border(im, frac=0.1):
    """Crop off a border fraction on each side so the black plate edges
    don't dominate the alignment score."""
    h, w = im.shape
    dy, dx = int(h * frac), int(w * frac)
    return im[dy:h - dy, dx:w - dx]

def ncc_score(im1, im2):
    """Normalized cross-correlation between two mean-subtracted images.
    Higher is better where 1.0 = perfectly correlated."""
    a = im1 - np.mean(im1)
    b = im2 - np.mean(im2)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return np.sum(a * b) / (na * nb)

def ssd_score(im1, im2):
    """Sum of squared differences (L2 norm squared). Lower is better."""
    return np.sum((im1 - im2) ** 2)

def align(im, ref, window=15, metric='ncc', border_frac=0.1, feature_fn=None):
    """Exhaustively search displacements (dy, dx) in [-window, window]^2
    that best align `im` onto `ref` using np.roll (circular shift).

    `feature_fn`, if given, is applied to the cropped images before
    scoring.
    The returned shift is still applied to the original-space `im`.

    Returns (dy, dx, aligned_im).
    """
    im_c = crop_border(im, border_frac)
    ref_c = crop_border(ref, border_frac)

    if feature_fn is not None:
        im_c = feature_fn(im_c)
        ref_c = feature_fn(ref_c)

    best_score = None
    best_shift = (0, 0)

    for dy in range(-window, window + 1):
        for dx in range(-window, window + 1):
            shifted = np.roll(im_c, (dy, dx), axis=(0, 1))

            if metric == 'ncc':
                score = ncc_score(shifted, ref_c)
                better = (best_score is None) or (score > best_score)
            else:  # 'ssd' / L2
                score = ssd_score(shifted, ref_c)
                better = (best_score is None) or (score < best_score)

            if better:
                best_score = score
                best_shift = (dy, dx)

    dy, dx = best_shift
    aligned = np.roll(im, (dy, dx), axis=(0, 1))
    return dy, dx, aligned

def main():
    parser = argparse.ArgumentParser(description='Single-scale channel alignment.')
    parser.add_argument('--imname', type=str, default='./cathedral.jpg',
                         help='path to the glass plate image')
    parser.add_argument('--window', type=int, default=15,
                         help='search window: displacements in [-window, window]')
    parser.add_argument('--metric', type=str, default='ncc', choices=['ncc', 'ssd'],
                         help='matching metric used to score candidate shifts')
    parser.add_argument('--out', type=str, default='./out_fname.jpg',
                         help='path to save the output color image')
    args = parser.parse_args()

    # read in the image as grayscale (the glass plate scan is stacked grayscale)
    im = cv.imread(args.imname, cv.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f'Could not read image: {args.imname}')

    # convert to float in [0,1]
    im = im.astype(np.float32) / 255.0

    # compute the height of each part (just 1/3 of total)
    height = int(np.floor(im.shape[0] / 3.0))

    # separate color channels
    b = im[:height]
    g = im[height:2 * height]
    r = im[2 * height:3 * height]

    # align G and R to B, searching over the user-specified window
    dy_g, dx_g, ag = align(g, b, window=args.window, metric=args.metric)
    dy_r, dx_r, ar = align(r, b, window=args.window, metric=args.metric)

    # displacement vectors, reported as (x, y)
    print(f'G displacement (x, y): ({dx_g}, {dy_g})')
    print(f'R displacement (x, y): ({dx_r}, {dy_r})')

    # create a color image
    im_out = np.dstack([ar, ag, b])

    # display the image using matplotlib (expects RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(im_out)
    plt.title('Colorized')
    plt.axis('off')
    plt.show()

    # prepare for OpenCV saving (expects BGR uint8)
    out_uint8 = np.clip(im_out * 255.0, 0, 255).astype(np.uint8)
    out_bgr = cv.cvtColor(out_uint8, cv.COLOR_RGB2BGR)

    # save the image
    cv.imwrite(args.out, out_bgr)

if __name__ == '__main__':
    main()
