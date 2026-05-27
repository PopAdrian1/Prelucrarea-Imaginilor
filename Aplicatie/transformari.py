"""
transformari.py — Transformări și algoritmi avansați de procesare imagini
=========================================================================
Conține:
  • Transformata Fourier 2D (numpy permis)
  • Floyd-Steinberg dithering
  • Metoda Canny de detectare a marginilor (fără OpenCV)
  • Compresie/Decompresie LZW pe imagini
  • Codificare/Decodificare Huffman (bonus)
  • Codificare/Decodificare RLE/RLC (bonus)
"""

import numpy as np
import math
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMATA FOURIER 2D — Lab 5
# ══════════════════════════════════════════════════════════════════════════════

def transformata_fourier(img):
    """
    Aplică Transformata Fourier Discretă 2D (FFT2) pe o imagine grayscale.
    Pași:
      1. Convertim imaginea la grayscale
      2. Aplicăm np.fft.fft2 (FFT pe 2 dimensiuni)
      3. Mutăm componenta DC (frecvența 0) în centrul imaginii cu fftshift
      4. Calculăm magnitudinea în scala logaritmică: log(1 + |F|)
      5. Normalizăm la [0,255] pentru afișare

    Scala logaritmică este necesară deoarece magnitudinea variază pe mai mulți
    ordini de mărime — fără log, frecvențele joase (centrul) ar domina complet.

    Returnează imagine RGB (grayscale vizualizat) cu spectrul de frecvențe.
    """
    gray = img.convert("L")
    # Convertim la array numpy float64 pentru calcul
    pixels = np.array(gray, dtype=np.float64)

    # FFT 2D și centrare
    fft_result = np.fft.fft2(pixels)
    fft_shifted = np.fft.fftshift(fft_result)

    # Magnitudine în scala log (evităm log(0) cu +1)
    magnitude = np.log(1 + np.abs(fft_shifted))

    # Normalizare la [0, 255]
    max_mag = magnitude.max()
    if max_mag > 0:
        magnitude = (magnitude / max_mag * 255).astype(np.uint8)
    else:
        magnitude = magnitude.astype(np.uint8)

    return Image.fromarray(magnitude).convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
#  FLOYD-STEINBERG DITHERING — Lab 5 (Extra)
# ══════════════════════════════════════════════════════════════════════════════

def floyd_steinberg(img, palette=None):
    """
    Algoritm Floyd-Steinberg de dithering cu difuzarea erorii de cuantificare.
    Reduce numărul de culori din imagine la paleta dată, distribuind eroarea
    la pixelii vecini conform matricei:
              X   7/16
         3/16 5/16 1/16

    Parametri:
        img     — imagine PIL.Image sursă
        palette — listă de tupluri (R,G,B); implicit alb-negru [(0,0,0),(255,255,255)]

    Pseudocod (conform laboratorului):
      pentru fiecare pixel (i,j) de la stânga-sus la dreapta-jos:
        pixel_vechi = img[i,j]
        pixel_nou   = culoarea cea mai apropiată din paletă
        img[i,j]    = pixel_nou
        eroare      = pixel_vechi - pixel_nou
        img[i, j+1]   += eroare * 7/16   (dreapta)
        img[i+1, j-1] += eroare * 3/16   (jos-stânga)
        img[i+1, j]   += eroare * 5/16   (jos)
        img[i+1, j+1] += eroare * 1/16   (jos-dreapta)
    """
    if palette is None:
        palette = [(0, 0, 0), (255, 255, 255)]

    src = img.convert("RGB")
    w, h = src.size
    # Lucrăm pe array float32 pentru a acumula erorile (pot fi fracții)
    pixels = np.array(src, dtype=np.float32)

    def nearest_color(r, g, b):
        """Găsește culoarea cea mai apropiată din paletă (distanță euclidiană)."""
        best = None
        best_dist = float('inf')
        for cr, cg, cb in palette:
            dist = math.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
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

            # Cuantizare: găsim culoarea cea mai apropiată
            nr, ng, nb = nearest_color(old_r, old_g, old_b)
            res_px[i, j] = (nr, ng, nb)

            # Calculăm eroarea de cuantificare pe fiecare canal
            err_r = old_r - nr
            err_g = old_g - ng
            err_b = old_b - nb

            # Difuzăm eroarea la pixelii vecini (conform matricei Floyd-Steinberg)
            if i + 1 < w:
                pixels[j,     i + 1, 0] += err_r * 7 / 16
                pixels[j,     i + 1, 1] += err_g * 7 / 16
                pixels[j,     i + 1, 2] += err_b * 7 / 16
            if j + 1 < h and i - 1 >= 0:
                pixels[j + 1, i - 1, 0] += err_r * 3 / 16
                pixels[j + 1, i - 1, 1] += err_g * 3 / 16
                pixels[j + 1, i - 1, 2] += err_b * 3 / 16
            if j + 1 < h:
                pixels[j + 1, i,     0] += err_r * 5 / 16
                pixels[j + 1, i,     1] += err_g * 5 / 16
                pixels[j + 1, i,     2] += err_b * 5 / 16
            if j + 1 < h and i + 1 < w:
                pixels[j + 1, i + 1, 0] += err_r * 1 / 16
                pixels[j + 1, i + 1, 1] += err_g * 1 / 16
                pixels[j + 1, i + 1, 2] += err_b * 1 / 16

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  METODA CANNY — Lab 7
# ══════════════════════════════════════════════════════════════════════════════

def _gaussian_blur_canny(img, size=3):
    """
    Blur Gaussian simplu 3×3 (medie) pentru preprocesare Canny.
    Folosit intern — nu are padding pe margini (pixelii de 1px sunt lăsați 0).
    """
    src = img.convert("L")
    w, h = src.size
    dst = Image.new("L", (w, h), 0)
    src_px = src.load()
    dst_px = dst.load()

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            total = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    total += src_px[x + dx, y + dy]
            dst_px[x, y] = total // 9
    return dst


def _sobel_gradients(img):
    """
    Calculează gradienții Sobel Gx și Gy pe o imagine grayscale.
    Kerneluri:
      Gx: [-1  0  1]    Gy: [-1 -2 -1]
          [-2  0  2]        [ 0  0  0]
          [-1  0  1]        [ 1  2  1]
    Returnează (magnitude, directie) ca liste 2D de float.
    """
    src_px = img.load()
    w, h = img.size
    # Inițializăm matricele de gradient
    magnitude = [[0.0] * w for _ in range(h)]
    direction = [[0.0] * w for _ in range(h)]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # Gradient pe X (detectează margini verticale)
            gx = (src_px[x + 1, y - 1] + 2 * src_px[x + 1, y] + src_px[x + 1, y + 1]
                  - src_px[x - 1, y - 1] - 2 * src_px[x - 1, y] - src_px[x - 1, y + 1])
            # Gradient pe Y (detectează margini orizontale)
            gy = (src_px[x - 1, y + 1] + 2 * src_px[x, y + 1] + src_px[x + 1, y + 1]
                  - src_px[x - 1, y - 1] - 2 * src_px[x, y - 1] - src_px[x + 1, y - 1])

            magnitude[y][x] = math.sqrt(gx * gx + gy * gy)
            direction[y][x] = math.atan2(gy, gx)

    return magnitude, direction


def _non_max_suppression(magnitude, direction, w, h):
    """
    Subțierea marginilor (Non-Maximum Suppression).
    Compară magnitudinea fiecărui pixel cu vecinii săi în direcția gradientului.
    Dacă pixelul nu este maxim local → se elimină (0).
    Se obțin margini de grosime 1 pixel.
    """
    suppressed = [[0.0] * w for _ in range(h)]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            mag = magnitude[y][x]
            # Convertim direcția la grade și o normalizăm la [0, 180)
            angle = math.degrees(direction[y][x]) % 180

            # Alegem pixelii vecini în funcție de direcția gradientului
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                # Direcție aproape orizontală → comparăm cu stânga/dreapta
                p1 = magnitude[y][x - 1]
                p2 = magnitude[y][x + 1]
            elif 22.5 <= angle < 67.5:
                # Diagonală jos-stânga → sus-dreapta
                p1 = magnitude[y + 1][x - 1]
                p2 = magnitude[y - 1][x + 1]
            elif 67.5 <= angle < 112.5:
                # Direcție aproape verticală → comparăm cu sus/jos
                p1 = magnitude[y - 1][x]
                p2 = magnitude[y + 1][x]
            else:
                # Diagonală sus-stânga → jos-dreapta
                p1 = magnitude[y - 1][x - 1]
                p2 = magnitude[y + 1][x + 1]

            # Păstrăm pixelul doar dacă e maxim local
            if mag >= p1 and mag >= p2:
                suppressed[y][x] = mag
    return suppressed


def _hysteresis_thresholding(suppressed, w, h, low=50, high=150):
    """
    Prag cu histerezis (Hysteresis Thresholding).
    Clasifică pixelii în:
      - SIGUR margine (> high)
      - POSIBIL margine (low < val <= high) — inclus dacă e conectat la o margine sigură
      - FUNDAL (< low)
    Returnează o imagine PIL.Image binară (255 = margine, 0 = fundal).
    """
    result = Image.new("L", (w, h), 0)
    res_px = result.load()

    # Prima trecere: marcăm pixelii siguri și posibili
    strong = []
    weak = []
    for y in range(h):
        for x in range(w):
            val = suppressed[y][x]
            if val > high:
                res_px[x, y] = 255
                strong.append((x, y))
            elif val > low:
                weak.append((x, y))

    # A doua trecere: includem pixelii posibili care sunt conectați la cei siguri
    for (x, y) in weak:
        connected = False
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if res_px[nx, ny] == 255:
                        connected = True
                        break
            if connected:
                break
        if connected:
            res_px[x, y] = 255

    return result.convert("RGB")


def canny_edge_detection(img, low=50, high=150, iterations=1):
    """
    Detectarea marginilor prin metoda Canny (fără OpenCV).
    Pași:
      1. Conversie la grayscale
      2. Blur Gaussian 3×3 (reducere zgomot)
      3. Calculul gradienților Sobel (Gx, Gy) → magnitudine și direcție
      4. Non-Maximum Suppression (subțierea marginilor)
      5. Hysteresis Thresholding (eliminarea marginilor slabe izolate)

    Parametri:
        img        — imagine PIL sursă
        low        — pragul inferior pentru histerezis (default 50)
        high       — pragul superior pentru histerezis (default 150)
        iterations — de câte ori se repetă procesul (default 1)

    Returnează imagine PIL.Image RGB cu marginile detectate.
    """
    result = img
    for _ in range(iterations):
        # Pasul 1 & 2: grayscale + blur
        blurred = _gaussian_blur_canny(result)
        w, h = blurred.size

        # Pasul 3: gradienți Sobel
        magnitude, direction = _sobel_gradients(blurred)

        # Pasul 4: subțiere margini
        suppressed = _non_max_suppression(magnitude, direction, w, h)

        # Pasul 5: histerezis
        result = _hysteresis_thresholding(suppressed, w, h, low, high)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  COMPRESIE LZW — Lab 8
# ══════════════════════════════════════════════════════════════════════════════

def lzw_compress(data):
    """
    Compresie LZW a unei secvențe de bytes (sau lista de int).
    Inițializăm dicționarul cu toate simbolurile posibile (0-255).
    Algoritmul identifică șiruri repetitive și le înlocuiește cu coduri.

    Pseudocod (conform laboratorului):
      Inițializează dicționar cu simboluri individuale (0..255)
      P = primul simbol din intrare
      WHILE nu s-a terminat intrarea:
        C = următorul simbol
        DACA P+C există în dicționar: P = P+C
        ALTFEL: outputează codul lui P
                adaugă P+C în dicționar
                P = C
      outputează codul lui P

    Returnează lista de coduri (int).
    """
    # Inițializăm dicționarul cu simbolurile de bază (0-255)
    dict_size = 256
    dictionary = {bytes([i]): i for i in range(dict_size)}

    # Convertim la bytes dacă e necesar
    if isinstance(data, (list, bytearray)):
        data = bytes(data)

    result = []
    current = bytes([data[0]])  # P = primul simbol

    for byte in data[1:]:
        combined = current + bytes([byte])  # P + C
        if combined in dictionary:
            current = combined              # Extindem secvența curentă
        else:
            result.append(dictionary[current])   # Outputăm codul lui P
            dictionary[combined] = dict_size      # Adăugăm P+C în dicționar
            dict_size += 1
            current = bytes([byte])         # P = C

    result.append(dictionary[current])  # Outputăm ultimul cod
    return result


def lzw_decompress(codes):
    """
    Decompresie LZW dintr-o listă de coduri întregi.
    Reconstruiește secvența originală de bytes.

    Pseudocod (conform laboratorului):
      Inițializează dicționar invers (cod -> șir)
      OLD = primul cod; outputează traducerea lui OLD
      WHILE nu s-a terminat intrarea:
        NEW = următorul cod
        DACA NEW nu e în dicționar:
          S = traducerea lui OLD + primul caracter al traducerii lui OLD
        ALTFEL:
          S = traducerea lui NEW
        outputează S
        C = primul caracter din S
        adaugă (OLD + C) în dicționar
        OLD = NEW

    Returnează bytes cu datele decomprimate.
    """
    # Inițializăm dicționarul invers
    dict_size = 256
    dictionary = {i: bytes([i]) for i in range(dict_size)}

    result = bytearray()
    old_code = codes[0]
    result.extend(dictionary[old_code])

    for code in codes[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            # Cazul special: codul tocmai adăugat (nu e încă în dicționar)
            entry = dictionary[old_code] + bytes([dictionary[old_code][0]])
        else:
            raise ValueError(f"Cod LZW invalid: {code}")

        result.extend(entry)
        # Adăugăm noua intrare în dicționar
        dictionary[dict_size] = dictionary[old_code] + bytes([entry[0]])
        dict_size += 1
        old_code = code

    return bytes(result)


def compress_image_lzw(img):
    """
    Comprimă o imagine PIL în format LZW.
    Imaginea este convertită la grayscale, pixelii sunt extrași ca secvență
    de bytes, apoi se aplică compresia LZW.
    Returnează (coduri_lzw, width, height) pentru a putea decomprima ulterior.
    """
    gray = img.convert("L")
    w, h = gray.size
    # Extragem pixelii ca secvență plată de bytes
    pixel_data = list(gray.getdata())
    codes = lzw_compress(bytes(pixel_data))
    return codes, w, h


def decompress_image_lzw(codes, w, h):
    """
    Decomprimă datele LZW înapoi într-o imagine PIL.
    Parametri:
        codes — lista de coduri întregi (din compress_image_lzw)
        w, h  — dimensiunile imaginii originale
    Returnează imagine PIL.Image grayscale.
    """
    pixel_data = lzw_decompress(codes)
    # Reconstruim imaginea din datele decomprimate
    img = Image.new("L", (w, h))
    img.putdata(list(pixel_data[:w * h]))
    return img.convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
#  CODIFICARE HUFFMAN — Lab 8 (Bonus)
# ══════════════════════════════════════════════════════════════════════════════

class _HuffmanNode:
    """Nod pentru arborele Huffman."""
    def __init__(self, symbol, freq):
        self.symbol = symbol   # Simbolul (valoare pixel 0-255 sau None pentru nod intern)
        self.freq   = freq     # Frecvența de apariție
        self.left   = None     # Copil stânga
        self.right  = None     # Copil dreapta

    def __lt__(self, other):
        return self.freq < other.freq


def huffman_encode(img):
    """
    Codificare Huffman a unei imagini grayscale.
    Pași:
      1. Calculăm frecvența fiecărei valori de pixel (0-255)
      2. Construim arborele Huffman (min-heap pe frecvențe)
      3. Generăm codurile binare pentru fiecare simbol
      4. Codificăm imaginea ca șir de biți

    Returnează (coduri_dict, bit_string, width, height)
    unde coduri_dict = {valoare_pixel: cod_binar_string}
    """
    gray = img.convert("L")
    w, h = gray.size
    pixels = list(gray.getdata())

    # Calculăm frecvențele
    freq = {}
    for p in pixels:
        freq[p] = freq.get(p, 0) + 1

    # Construim heap simplu (sortare repetată — fără heapq pentru claritate)
    nodes = [_HuffmanNode(sym, f) for sym, f in freq.items()]
    nodes.sort(key=lambda n: n.freq)

    # Construim arborele Huffman
    while len(nodes) > 1:
        left = nodes.pop(0)
        right = nodes.pop(0)
        parent = _HuffmanNode(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        nodes.append(parent)
        nodes.sort(key=lambda n: n.freq)

    root = nodes[0]

    # Generăm codurile binare prin parcurgere DFS
    codes = {}
    def generate_codes(node, code=""):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = code if code else "0"
            return
        generate_codes(node.left,  code + "0")
        generate_codes(node.right, code + "1")

    generate_codes(root)

    # Codificăm imaginea
    bit_string = "".join(codes[p] for p in pixels)
    return codes, bit_string, w, h


def huffman_decode(codes, bit_string, w, h):
    """
    Decodificare Huffman: reconstruiește imaginea din șirul de biți și dicționarul de coduri.
    Returnează imagine PIL.Image grayscale.
    """
    # Inversăm dicționarul: cod_binar -> valoare_pixel
    reverse_codes = {v: k for k, v in codes.items()}

    pixels = []
    current = ""
    for bit in bit_string:
        current += bit
        if current in reverse_codes:
            pixels.append(reverse_codes[current])
            current = ""

    img = Image.new("L", (w, h))
    img.putdata(pixels[:w * h])
    return img.convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
#  CODIFICARE RLE/RLC — Lab 8 (Bonus)
# ══════════════════════════════════════════════════════════════════════════════

def rle_encode(img):
    """
    Codificare RLE (Run-Length Encoding) pe o imagine grayscale.
    Comprimă secvențe de pixeli identici ca (valoare, număr_repetări).
    Eficientă pentru imagini cu zone uniforme (ex: imagini binare).

    Pseudocod:
      pentru fiecare pixel din secvența plată:
        dacă pixel == pixel_anterior: incrementăm contorul
        altfel: outputăm (pixel_anterior, contor); resetăm contorul
      outputăm ultima pereche

    Returnează (perechi_rle, width, height)
    unde perechi_rle = [(valoare, count), ...]
    """
    gray = img.convert("L")
    w, h = gray.size
    pixels = list(gray.getdata())

    encoded = []
    if not pixels:
        return encoded, w, h

    count = 1
    current = pixels[0]
    for p in pixels[1:]:
        if p == current:
            count += 1
        else:
            encoded.append((current, count))
            current = p
            count = 1
    encoded.append((current, count))  # Ultima secvență
    return encoded, w, h


def rle_decode(encoded, w, h):
    """
    Decodificare RLE: reconstruiește imaginea din perechile (valoare, count).
    Returnează imagine PIL.Image grayscale.
    """
    pixels = []
    for value, count in encoded:
        pixels.extend([value] * count)

    img = Image.new("L", (w, h))
    img.putdata(pixels[:w * h])
    return img.convert("RGB")