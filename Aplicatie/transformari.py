import numpy as np
import math
from PIL import Image

def transformata_fourier(img):
    gray = img.convert("L")
    pixels = np.array(gray, dtype=np.float64)
    fft_result = np.fft.fft2(pixels)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude = np.log(1 + np.abs(fft_shifted))
    max_mag = magnitude.max()
    if max_mag > 0:
        magnitude = (magnitude / max_mag * 255).astype(np.uint8)
    else:
        magnitude = magnitude.astype(np.uint8)
    return Image.fromarray(magnitude).convert("RGB")


def floyd_steinberg(img, palette=None):
    if palette is None:
        palette = [(0, 0, 0), (255, 255, 255)]
    src = img.convert("RGB")
    w, h = src.size
    pixels = np.array(src, dtype=np.float32)

    def nearest_color(r, g, b):
        best = None
        best_dist = float('inf')
        for cr, cg, cb in palette:
            dist = math.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
            if dist < best_dist:
                best_dist = dist
                best = (cr, cg, cb)
        return best

    result = Image.new("RGB", (w, h))
    res_px = result.load()
    for j in range(h):
        for i in range(w):
            old_r = pixels[j, i, 0]
            old_g = pixels[j, i, 1]
            old_b = pixels[j, i, 2]
            nr, ng, nb = nearest_color(old_r, old_g, old_b)
            res_px[i, j] = (nr, ng, nb)
            err_r = old_r - nr
            err_g = old_g - ng
            err_b = old_b - nb
            if i + 1 < w:
                pixels[j,   i+1, 0] += err_r * 7 / 16
                pixels[j,   i+1, 1] += err_g * 7 / 16
                pixels[j,   i+1, 2] += err_b * 7 / 16
            if j + 1 < h and i - 1 >= 0:
                pixels[j+1, i-1, 0] += err_r * 3 / 16
                pixels[j+1, i-1, 1] += err_g * 3 / 16
                pixels[j+1, i-1, 2] += err_b * 3 / 16
            if j + 1 < h:
                pixels[j+1, i,   0] += err_r * 5 / 16
                pixels[j+1, i,   1] += err_g * 5 / 16
                pixels[j+1, i,   2] += err_b * 5 / 16
            if j + 1 < h and i + 1 < w:
                pixels[j+1, i+1, 0] += err_r * 1 / 16
                pixels[j+1, i+1, 1] += err_g * 1 / 16
                pixels[j+1, i+1, 2] += err_b * 1 / 16
    return result