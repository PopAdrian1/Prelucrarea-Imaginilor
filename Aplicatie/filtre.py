"""

Conține:
  • Filtre pixel-cu-pixel (grayscale, binarizare, CMYK, negative, YUV, YCbCr, HSV, RGB back)
  • Filtre pe imagine întreagă bazate pe kernel 3×3 (mediere, median, minim, maxim, accentuare)
  • Filtre noi: Laplacian, eliminare zgomot Gaussian, SNR (2 variante)
  • Filtre detecție contur: Vertical, Horizontal, Sobel V/H, Scharr V/H
  • Laplacianul Gaussianului (LoG)
"""

import math
from PIL import Image


#  FILTRE GRAYSCALE — Lab 1 & 2

def filter_gray_1(r, g, b, v):
    """
    Conversie RGB -> Grayscale folosind media aritmetică a celor trei canale.
    Formula: Gray = (R + G + B) / 3
    Se aplică un factor de amplificare 'gray_gain' din parametri.
    """
    val = int(((r + g + b) / 3) * v["gray_gain"])
    return val, val, val


def filter_gray_2(r, g, b, v):
    """
    Conversie RGB -> Grayscale folosind formula luminanței perceptuale (ITU-R BT.601).
    Formula: Gray = 0.299*R + 0.587*G + 0.114*B
    Aceasta corespunde sensibilității ochiului uman la fiecare culoare.
    Se aplică factorul 'gray_gain' din parametri.
    """
    val = int((0.299 * r + 0.587 * g + 0.114 * b) * v["gray_gain"])
    return val, val, val


def filter_gray_3(r, g, b, v):
    """
    Conversie RGB -> Grayscale prin desaturare (media min-max).
    Formula: Gray = (max(R,G,B) + min(R,G,B)) / 2
    Echivalent cu L din modelul HSL.
    Se aplică factorul 'gray_gain' din parametri.
    """
    val = int(((max(r, g, b) + min(r, g, b)) // 2) * v["gray_gain"])
    return val, val, val


def binarizare(r, g, b, v):
    """
    Binarizarea imaginii pe baza unui prag ('thresh').
    Dacă luminanța pixelului > prag => alb (255), altfel negru (0).
    Luminanța se calculează cu formula ITU BT.601.
    """
    lum = int(0.299 * r + 0.587 * g + 0.114 * b)
    val = 255 if lum > v["thresh"] else 0
    return val, val, val



#  FILTRE SPAȚIU DE CULOARE — Lab 2


def filtru_cmyk(r, g, b, v):
    """
    Conversie RGB -> CMYK (model substractiv de culoare).
    Pașii:
      1. Normalizăm R, G, B la [0,1]
      2. K = 1 - max(R', G', B')
      3. C = (1 - R' - K) / (1 - K)  (similar M, Y)
    Fiecare canal este scalat cu parametrii 'c', 'm', 'y_c' (yellow), 'k'.
    Rezultatul este o imagine cu culorile CMYK convertite înapoi la RGB pentru afișare.
    """
    rf, gf, bf_n = r / 255, g / 255, b / 255
    # Componenta K (negru) din modelul CMYK
    k_f = (1 - max(rf, gf, bf_n)) * v["k"]

    nr = ng = nb = 0
    if k_f < 1:
        # Calculăm C, M, Y normalizat și aplicăm scalele din parametri
        nr = int(((1 - rf - k_f) / (1 - k_f)) * 255 * v["c"])
        ng = int(((1 - gf - k_f) / (1 - k_f)) * 255 * v["m"])
        nb = int(((1 - bf_n - k_f) / (1 - k_f)) * 255 * v["y_c"])

    return nr, ng, nb


def filtru_negativ(r, g, b, v):
    """
    Filtrul negativ inversează fiecare canal de culoare.
    Formula: R' = 255 - R (similar G, B)
    Parametrii 'r', 'g', 'b' scalează independent fiecare canal.
    """
    nr = int((255 - r) * v["r"])
    ng = int((255 - g) * v["g"])
    nb = int((255 - b) * v["b"])
    return nr, ng, nb



#  SPAȚIU LUMINANȚĂ/CROMINANȚĂ — Lab 3

def filtru_yuv(r, g, b, v):
    """
    Conversie RGB -> YUV (spațiu luminanță + crominanță).
    Y = 0.3*R + 0.6*G + 0.1*B  (luminanță)
    U = 0.74*(R-Y) + 0.27*(B-Y)  (diferență cromatică albastru)
    V = 0.48*(R-Y) + 0.41*(B-Y)  (diferență cromatică roșu)
    Fiecare componentă este scalată cu parametrul corespunzător.
    """
    y_v = int((0.3 * r + 0.6 * g + 0.1 * b) * v["y_luma"])
    nr = y_v
    ng = int((0.74 * (r - y_v) + 0.27 * (b - y_v)) * v["u"]) + 128
    nb = int((0.48 * (r - y_v) + 0.41 * (b - y_v)) * v["v"]) + 128
    return nr, ng, nb


def filtru_ycbcr(r, g, b, v):
    """
    Conversie RGB -> YCbCr (standard JFIF/JPEG, ITU-R BT.601).
    Y   =  0.299*R + 0.587*G + 0.114*B          (luminanță)
    Cb  = -0.1687*R - 0.3313*G + 0.5*B  + 128  (crominanță albastru)
    Cr  =  0.5*R - 0.4187*G - 0.0813*B  + 128  (crominanță roșu)
    Parametrii 'y_luma', 'cb', 'cr' scalează componentele.
    """
    y_v = int((0.299 * r + 0.587 * g + 0.114 * b) * v["y_luma"])
    cb = int((-0.1687 * r - 0.3313 * g + 0.5 * b) * v["cb"]) + 128
    cr = int((0.5 * r - 0.4187 * g - 0.0813 * b) * v["cr"]) + 128
    return y_v, cb, cr


def rgb_back(r, g, b, v):
    """
    Conversie inversă YCbCr -> RGB.
    Se calculează mai întâi YCbCr cu parametrii curenți, apoi se inversează:
    R = Y + 1.402*(Cr-128)
    G = Y - 0.34414*(Cb-128) - 0.71414*(Cr-128)
    B = Y + 1.772*(Cb-128)
    Parametrii 'r', 'g', 'b' scalează canalele rezultate.
    """
    y_v, cb, cr = filtru_ycbcr(r, g, b, v)
    nr = int((y_v + 1.402 * (cr - 128)) * v["r"])
    ng = int((y_v - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)) * v["g"])
    nb = int((y_v + 1.772 * (cb - 128)) * v["b"])
    return nr, ng, nb


#  SPAȚIUL HSV — Lab 4


def filtru_hsv(r, g, b, v):
    """
    Conversie RGB -> HSV (Hue, Saturation, Value).
    Pași:
      1. Normalizăm R, G, B la [0,1]
      2. M = max(r,g,b);  m = min(r,g,b);  C = M - m
      3. V = M
      4. S = C/V dacă V != 0, altfel S = 0
      5. H calculat în funcție de care componentă e maximă, apoi normalizat la [0,360]
    Valorile H, S, V sunt scalate la [0,255] pentru reprezentare ca imagine 8-bit.
    Parametrii 'h', 's', 'v_hsv' scalează componentele.
    """
    rf, gf, bf_n = r / 255, g / 255, b / 255
    M, mn = max(rf, gf, bf_n), min(rf, gf, bf_n)
    C = M - mn
    V = M  # Value = max

    # Saturație
    S = (C / V) if V else 0

    # Nuanță (Hue)
    if C:
        if M == rf:
            H = 60 * (((gf - bf_n) / C) % 6)   # roșu dominant
        elif M == gf:
            H = 60 * ((bf_n - rf) / C + 2)      # verde dominant
        else:
            H = 60 * ((rf - gf) / C + 4)        # albastru dominant
    else:
        H = 0  # pixel acromatic (gri)

    # Normalizare la [0,255]
    nr = int((H / 360) * 255 * v["h"])
    ng = int(S * 255 * v["s"])
    nb = int(V * 255 * v["v_hsv"])
    return nr, ng, nb


#  FILTRE KERNEL 3×3 — Lab 5 

def apply_kernel_3x3(img, kernel):
    """
    Aplică un kernel 3×3 pe imagine prin convoluție.
    Marginile imaginii (1 pixel) rămân negre deoarece vecinătatea nu e completă.
    Parametri:
        img    — obiect PIL.Image
        kernel — matrice 3×3 (lista de liste de float)
    Returnează o nouă imagine PIL.Image cu filtrul aplicat.
    """
    src = img.convert("RGB")
    w, h = src.size
    dst = Image.new("RGB", (w, h), (0, 0, 0))
    src_px = src.load()
    dst_px = dst.load()

    for i in range(1, w - 1):
        for j in range(1, h - 1):
            sumr = sumg = sumb = 0.0
            # Parcurgem vecinătatea 3×3
            for k in range(-1, 2):
                for l in range(-1, 2):
                    r, g, b = src_px[i + k, j + l]
                    coef = kernel[k + 1][l + 1]
                    sumr += coef * r
                    sumg += coef * g
                    sumb += coef * b
            # Clampare la [0,255]
            dst_px[i, j] = (
                max(0, min(255, int(sumr))),
                max(0, min(255, int(sumg))),
                max(0, min(255, int(sumb)))
            )
    return dst


def filtru_mediere(img):
    """
    Filtrul de mediere (blur) 3×3.
    Fiecare pixel devine media aritmetică a celor 9 pixeli din vecinătatea sa.
    Kernelul: toate coeficienții = 1/9.
    Efect: netezire (eliminare zgomot), pierdere detalii fine.
    """
    kernel = [[1 / 9] * 3 for _ in range(3)]
    return apply_kernel_3x3(img, kernel)


def filtru_median(img):
    """
    Filtrul median 3×3.
    Selectează valoarea mediană (elementul de mijloc) din cei 9 pixeli vecini,
    sortați crescător cu Bubble Sort (ca în exemplul Java din laborator).
    Efect: elimină zgomotul de tip 'salt and pepper' fără a blur-ui marginile.
    Returnează o imagine grayscale (R=G=B=median).
    """
    src = img.convert("RGB")
    w, h = src.size
    dst = Image.new("RGB", (w, h), (0, 0, 0))
    src_px = src.load()
    dst_px = dst.load()

    for i in range(1, w - 1):
        for j in range(1, h - 1):
            sir = []
            # Colectăm cei 9 pixeli din vecinătatea 3×3 (canal R)
            for m in range(-1, 2):
                for n in range(-1, 2):
                    r, g, b = src_px[i + m, j + n]
                    sir.append(r)
            # Sortare Bubble Sort (ca în Java)
            changed = True
            while changed:
                changed = False
                for idx in range(len(sir) - 1):
                    if sir[idx] > sir[idx + 1]:
                        sir[idx], sir[idx + 1] = sir[idx + 1], sir[idx]
                        changed = True
            # Elementul median = indexul 4 (mijlocul din 9)
            med = sir[4]
            dst_px[i, j] = (med, med, med)
    return dst


def filtru_minim(img):
    """
    Filtrul de minim 3×3 (eroziune morfologică).
    Fiecare pixel devine valoarea minimă din cei 9 vecini.
    Efect: subțierea obiectelor albe, eliminarea zgomotului alb.
    Returnează imagine grayscale.
    """
    src = img.convert("RGB")
    w, h = src.size
    dst = Image.new("RGB", (w, h), (0, 0, 0))
    src_px = src.load()
    dst_px = dst.load()

    for i in range(1, w - 1):
        for j in range(1, h - 1):
            vals = []
            for m in range(-1, 2):
                for n in range(-1, 2):
                    r, g, b = src_px[i + m, j + n]
                    vals.append(r)
            mn = min(vals)
            dst_px[i, j] = (mn, mn, mn)
    return dst


def filtru_maxim(img):
    """
    Filtrul de maxim 3×3 (dilatare morfologică).
    Fiecare pixel devine valoarea maximă din cei 9 vecini.
    Efect: îngroșarea obiectelor albe, eliminarea zgomotului negru.
    Returnează imagine grayscale.
    """
    src = img.convert("RGB")
    w, h = src.size
    dst = Image.new("RGB", (w, h), (0, 0, 0))
    src_px = src.load()
    dst_px = dst.load()

    for i in range(1, w - 1):
        for j in range(1, h - 1):
            vals = []
            for m in range(-1, 2):
                for n in range(-1, 2):
                    r, g, b = src_px[i + m, j + n]
                    vals.append(r)
            mx = max(vals)
            dst_px[i, j] = (mx, mx, mx)
    return dst


def filtru_accentuare(img, alpha=0.6):
    """
    Filtrul de accentuare (sharpening) — Lab 5.
    Kernelul folosit:
        [  0   -1/4   0  ]
        [-1/4   1   -1/4 ]
        [  0   -1/4   0  ]
    Se calculează suma ponderată (high-pass), apoi se adaugă la pixelul original
    scalat cu 'alpha': pixel_nou = pixel_original + alpha * suma_kernel
    Efect: accentuarea marginilor și detaliilor.
    """
    kernel = [
        [0,     -1 / 4,  0   ],
        [-1 / 4, 1,     -1 / 4],
        [0,     -1 / 4,  0   ]
    ]
    src = img.convert("RGB")
    w, h = src.size
    dst = Image.new("RGB", (w, h), (0, 0, 0))
    src_px = src.load()
    dst_px = dst.load()

    for i in range(1, w - 1):
        for j in range(1, h - 1):
            sumr = sumg = sumb = 0.0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    r, g, b = src_px[i + k, j + l]
                    coef = kernel[k + 1][l + 1]
                    sumr += coef * r
                    sumg += coef * g
                    sumb += coef * b
            ro, go, bo = src_px[i, j]
            # Adăugăm componenta de accentuare la pixelul original
            nr = max(0, min(255, int(ro + alpha * sumr)))
            ng = max(0, min(255, int(go + alpha * sumg)))
            nb = max(0, min(255, int(bo + alpha * sumb)))
            dst_px[i, j] = (nr, ng, nb)
    return dst


#  FILTRUL LAPLACIAN — Lab 6

def filtru_laplacian(img):
    """
    Filtrul Laplacian pentru detectarea marginilor.
    Kernelul:
        [-1 -1 -1]
        [-1  8 -1]
        [-1 -1 -1]
    Produce margini de un singur pixel. Pixelii luminoși lângă cei întunecați
    devin mai luminoși și invers.
    Imaginea este mai întâi convertită la grayscale pentru calcul,
    iar rezultatul este salvat ca RGB (R=G=B=valoare).

    Pseudocod:
      pentru fiecare pixel (x,y) din interior:
        sum = 0
        pentru fiecare vecin (dx,dy) in [-1,0,1]:
          sum += kernel[dx+1][dy+1] * pixel_gri(x+dx, y+dy)
        clampam sum la [0,255]
        rezultat(x,y) = (sum, sum, sum)
    """
    # Kernel Laplacian 3×3
    kernel = [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ]
    # Convertim la grayscale pentru calcul
    gray = img.convert("L").convert("RGB")
    w, h = gray.size
    dst = Image.new("RGB", (w, h), (0, 0, 0))
    src_px = gray.load()
    dst_px = dst.load()

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            total = 0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    # Luăm doar canalul R (imagine grayscale => R=G=B)
                    total += kernel[m + 1][n + 1] * src_px[x + m, y + n][0]
            # Clampăm valoarea la [0, 255]
            val = max(0, min(255, abs(total)))
            dst_px[x, y] = (val, val, val)
    return dst


# ══════════════════════════════════════════════════════════════════════════════
#  ELIMINARE ZGOMOT GAUSSIAN — Lab 6
# ══════════════════════════════════════════════════════════════════════════════

def filtru_gaussian_noise_removal(img):
    """
    Elimină zgomotul Gaussian printr-un filtru de mediere 3×3 cu tratarea marginilor.
    (Spre deosebire de filtru_mediere, acesta nu lasă marginile negre —
     folosește 'mirror padding': pixelii de margine se clampează la bord.)

    Pseudocod:
      pentru fiecare pixel (x,y):
        suma_R = suma_G = suma_B = 0
        pentru fiecare offset (i,j) in [-1,0,1]:
          nx = clamp(x+i, 0, w-1)   # reflectare pe margini
          ny = clamp(y+j, 0, h-1)
          suma_R += R(nx,ny); suma_G += G(nx,ny); suma_B += B(nx,ny)
        pixel_nou = (suma_R/9, suma_G/9, suma_B/9)
    """
    src = img.convert("RGB")
    w, h = src.size
    dst = Image.new("RGB", (w, h), (0, 0, 0))
    src_px = src.load()
    dst_px = dst.load()

    kernel_size = 3
    half = kernel_size // 2

    for y in range(h):
        for x in range(w):
            sum_r = sum_g = sum_b = 0
            for i in range(-half, half + 1):
                for j in range(-half, half + 1):
                    # Mirror padding: clampăm coordonatele la marginile imaginii
                    nx = max(0, min(w - 1, x + i))
                    ny = max(0, min(h - 1, y + j))
                    r, g, b = src_px[nx, ny]
                    sum_r += r
                    sum_g += g
                    sum_b += b
            count = kernel_size * kernel_size
            dst_px[x, y] = (sum_r // count, sum_g // count, sum_b // count)
    return dst



# ══════════════════════════════════════════════════════════════════════════════
#  FILTRE DETECȚIE CONTUR — Lab 7
# ══════════════════════════════════════════════════════════════════════════════

# Definim kernelele de detecție a contururilor conform laboratorului
FILTER_VERTICAL  = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
FILTER_HORIZONTAL= [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
FILTER_SOBEL_V   = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]   # Sobel pe X (detectează margini verticale)
FILTER_SOBEL_H   = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]   # Sobel pe Y (detectează margini orizontale)
FILTER_SCHARR_V  = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]  # Scharr pe X (mai precis decât Sobel)
FILTER_SCHARR_H  = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]  # Scharr pe Y


def _apply_edge_kernel(img, kernel):
    """
    Funcție auxiliară: aplică un kernel de detecție contur pe imaginea RGB.
    Se aplică separat pe canalele R, G, B, apoi se face suma.
    Rezultatul este normalizat la [0, 255].
    """
    src = img.convert("RGB")
    w, h = src.size
    dst = Image.new("RGB", (w, h), (0, 0, 0))
    src_px = src.load()
    dst_px = dst.load()

    for j in range(1, h - 1):
        for i in range(1, w - 1):
            sum_r = sum_g = sum_b = 0.0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    r, g, b = src_px[i + m, j + n]
                    coef = kernel[m + 1][n + 1]
                    sum_r += coef * r
                    sum_g += coef * g
                    sum_b += coef * b
            # Suma canalelor (ca în Java original)
            total = abs(sum_r) + abs(sum_g) + abs(sum_b)
            val = max(0, min(255, int(total / 3)))
            dst_px[i, j] = (val, val, val)
    return dst


def filtru_contur_vertical(img):
    """Detectează marginile verticale cu filtrul simplu."""
    return _apply_edge_kernel(img, FILTER_VERTICAL)


def filtru_contur_orizontal(img):
    """Detectează marginile orizontale cu filtrul simplu."""
    return _apply_edge_kernel(img, FILTER_HORIZONTAL)


def filtru_sobel_v(img):
    """
    Detectează marginile verticale cu operatorul Sobel pe direcția X.
    Kernelul:  [-1  0  1]
               [-2  0  2]
               [-1  0  1]
    Adaugă mai multă greutate la mijloc față de filtrul vertical simplu.
    """
    return _apply_edge_kernel(img, FILTER_SOBEL_V)


def filtru_sobel_h(img):
    """
    Detectează marginile orizontale cu operatorul Sobel pe direcția Y.
    Kernelul:  [-1 -2 -1]
               [ 0  0  0]
               [ 1  2  1]
    """
    return _apply_edge_kernel(img, FILTER_SOBEL_H)


def filtru_scharr_v(img):
    """
    Detectează marginile verticale cu operatorul Scharr (mai precis decât Sobel).
    Kernelul:  [-3   0   3]
               [-10  0  10]
               [-3   0   3]
    """
    return _apply_edge_kernel(img, FILTER_SCHARR_V)


def filtru_scharr_h(img):
    """
    Detectează marginile orizontale cu operatorul Scharr.
    Kernelul:  [-3  -10  -3]
               [ 0    0   0]
               [ 3   10   3]
    """
    return _apply_edge_kernel(img, FILTER_SCHARR_H)


#  LAPLACIANUL GAUSSIANULUI (LoG) — Lab 8

def _gaussian_kernel(size, sigma):
    """
    Generează un kernel Gaussian de dimensiunea 'size' x 'size' cu deviația 'sigma'.
    Formula: G(i,j) = exp(-(i^2 + j^2) / (2*sigma^2))
    Kernelul este normalizat astfel încât suma coeficienților = 1.
    """
    half = size // 2
    kernel = []
    total = 0.0
    for i in range(-half, half + 1):
        row = []
        for j in range(-half, half + 1):
            val = math.exp(-(i * i + j * j) / (2 * sigma * sigma))
            row.append(val)
            total += val
        kernel.append(row)
    # Normalizare
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= total
    return kernel


def _apply_gaussian_filter(img, size=3, sigma=1.4):
    """
    Aplică un filtru Gaussian de dimensiune 'size' x 'size' cu sigma dat.
    Returnează o nouă imagine PIL (grayscale convertit la RGB).
    """
    gray = img.convert("L")
    w, h = gray.size
    src_px = gray.load()
    dst = Image.new("L", (w, h), 0)
    dst_px = dst.load()
    kernel = _gaussian_kernel(size, sigma)
    half = size // 2

    for y in range(half, h - half):
        for x in range(half, w - half):
            total = 0.0
            for i in range(-half, half + 1):
                for j in range(-half, half + 1):
                    total += kernel[i + half][j + half] * src_px[x + j, y + i]
            dst_px[x, y] = max(0, min(255, int(round(total))))
    return dst.convert("RGB")


def filtru_log(img):
    """
    Laplacianul Gaussianului (LoG) — detectare margini cu netezire.
    Pași:
      1. Aplică filtrul Gaussian 3×3 cu sigma=1.4 pentru a reduce zgomotul
      2. Aplică filtrul Laplacian pe imaginea netezită
    Efect: detectează marginile cu mai puțin zgomot față de Laplacianul simplu.
    """
    smoothed = _apply_gaussian_filter(img, size=3, sigma=1.4)
    return filtru_laplacian(smoothed)



#  DICȚIONARE DE MAPARE — folosite de app.py

# Filtre pixel-cu-pixel (funcție(r,g,b,v) -> (nr,ng,nb))
filters_map = {
    "gray(1)":   filter_gray_1,       # Grayscale medie aritmetică
    "gray(2)":   filter_gray_2,       # Grayscale luminanță perceptuală
    "gray(3)":   filter_gray_3,       # Grayscale desaturare
    "binarize":  binarizare,          # Binarizare cu prag
    "cmyk":      filtru_cmyk,         # Spațiu CMYK
    "negative":  filtru_negativ,      # Negativ
    "yuv":       filtru_yuv,          # Spațiu YUV
    "ycbcr":     filtru_ycbcr,        # Spațiu YCbCr
    "rgb_back":  rgb_back,            # Conversie inversă YCbCr->RGB
    "hsv":       filtru_hsv,          # Spațiu HSV
}

# Filtre pe imagine întreagă (funcție(img, v) -> img)
filters_map_img = {
    "mediere":           lambda img, v: filtru_mediere(img),
    "median":            lambda img, v: filtru_median(img),
    "minim":             lambda img, v: filtru_minim(img),
    "maxim":             lambda img, v: filtru_maxim(img),
    "accentuare":        lambda img, v: filtru_accentuare(img, v["alpha"]),
    "laplacian":         lambda img, v: filtru_laplacian(img),
    "gaussian_denoise":  lambda img, v: filtru_gaussian_noise_removal(img),
    "log":               lambda img, v: filtru_log(img),
    "contur_v":          lambda img, v: filtru_contur_vertical(img),
    "contur_h":          lambda img, v: filtru_contur_orizontal(img),
    "sobel_v":           lambda img, v: filtru_sobel_v(img),
    "sobel_h":           lambda img, v: filtru_sobel_h(img),
    "scharr_v":          lambda img, v: filtru_scharr_v(img),
    "scharr_h":          lambda img, v: filtru_scharr_h(img),
}