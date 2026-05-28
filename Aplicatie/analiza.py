"""
Conține:
  • Calculul centrului de greutate (momentul de ordinul 1)
  • Momentele centrale de ordinul 2 (mu20, mu02, mu11)
  • Matricea de covarianță normalizată
  • Proiecțiile orizontală și verticală
  • Calcul SNR (o singură imagine și două imagini comparate)
  • Raport complet de analiză (toate metricile reunite)
  • Etichetare componente
  • Generare culori etichete, randare, extragere mască obiect
  • Calcul orientare Sobel pe mască obiect

"""

import math
from collections import deque
from PIL import Image, ImageDraw


#  DETECȚIE OBIECT — extragere pixeli non-albi

def detecteaza_pixeli_obiect(img, prag_alb=240):
    """
    Detectează pixelii care aparțin obiectului (pixeli non-albi).

    Parcurgem fiecare pixel al imaginii. Dacă cel puțin unul dintre canalele
    R, G, B este sub pragul 'prag_alb', pixelul este considerat obiect.
    """
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    pixels = img_rgb.load()

    m00 = 0   # Numărul total de pixeli obiect (moment de ordinul 0)
    m10 = 0   # Suma coordonatelor X (pentru centrul pe X)
    m01 = 0   # Suma coordonatelor Y (pentru centrul pe Y)
    coords = []

    for y in range(h):
        for x in range(w):
            r, g, b = pixels[x, y]
            # Un pixel este considerat obiect dacă nu este aproape alb
            if r < prag_alb or g < prag_alb or b < prag_alb:
                m00 += 1
                m10 += x
                m01 += y
                coords.append((x, y))

    if m00 == 0:
        return None  # Nu s-a detectat niciun obiect

    return {
        "m00": m00,                    # Suprafața obiectului (nr. pixeli)
        "xc":  m10 / m00,             # Centrul de greutate pe axa X
        "yc":  m01 / m00,             # Centrul de greutate pe axa Y
        "coords": coords,             # Lista coordonatelor pixelilor obiect
        "w": w,                       # Lățimea imaginii
        "h": h                        # Înălțimea imaginii
    }


#  MOMENTUL DE ORDINUL 1 — centrul de greutate

def calculeaza_centru_greutate(img):
    """
    Calculează centrul de greutate al obiectului (momentul de ordinul 1).

    Momentul de ordinul 1 pe X: M10 = Σ x·f(x,y)
    Momentul de ordinul 1 pe Y: M01 = Σ y·f(x,y)
    Centrul: Xc = M10 / M00,  Yc = M01 / M00

    Returnează (xc, yc) sau None dacă imaginea nu conține obiect.

    """
    data = detecteaza_pixeli_obiect(img)
    if data is None:
        return None
    return data["xc"], data["yc"]


def deseneaza_centru(img, xc, yc, raza=6, culoare="red"):
    """
    Desenează un cerc roșu la centrul de greutate pe o copie a imaginii.

    Pseudocod:
      img_copie = copie(img)
      desenează cerc la (xc-raza, yc-raza, xc+raza, yc+raza)
    """
    copie = img.copy().convert("RGB")
    draw = ImageDraw.Draw(copie)
    draw.ellipse(
        [xc - raza, yc - raza, xc + raza, yc + raza],
        fill=culoare, outline="white", width=2
    )
    return copie


#  MOMENTELE CENTRALE DE ORDINUL 2 — distribuția masei față de centru

def calculeaza_momente_centrale(img):
    """
    Calculează momentele centrale de ordinul 2: mu20, mu02, mu11.

    Momentele centrale măsoară distribuția masei obiectului față de centrul
    său de greutate — echivalentul momentelor de inerție din fizică.

    Formule:
        mu20 = Σ (x - Xc)²      (variația pe X)
        mu02 = Σ (y - Yc)²      (variația pe Y)
        mu11 = Σ (x - Xc)(y - Yc)  (corelația XY — orientarea)

    Returnează dict {mu20, mu02, mu11} sau None.

    """
    data = detecteaza_pixeli_obiect(img)
    if data is None:
        return None

    xc, yc = data["xc"], data["yc"]
    coords = data["coords"]

    # Momentele centrale (față de centrul de greutate)
    mu20 = sum((x - xc) ** 2 for x, y in coords)  # Inertie pe X
    mu02 = sum((y - yc) ** 2 for x, y in coords)  # Inertie pe Y
    mu11 = sum((x - xc) * (y - yc) for x, y in coords)  # Corelatie XY

    return {"mu20": mu20, "mu02": mu02, "mu11": mu11}



#  MATRICEA DE COVARIANȚĂ — orientarea și forma obiectului


def calculeaza_covarianta(img):
    """
    Calculează matricea de covarianță normalizată a obiectului.

    Matricea de covarianță descrie forma și orientarea obiectului:
        | m20  m11 |
        | m11  m02 |
    unde m20, m02, m11 sunt momentele centrale normalizate (împărțite la M00).

    Vectorii proprii ai acestei matrice dau axele principale ale obiectului.
    Valorile proprii dau alungirea pe fiecare axă.

    Returnează dict {m20, m02, m11} sau None.
    """
    data = detecteaza_pixeli_obiect(img)
    if data is None:
        return None

    m00 = data["m00"]
    xc, yc = data["xc"], data["yc"]
    coords = data["coords"]

    # Momente centrale normalizate (împărțite la numărul de pixeli)
    m20 = sum((x - xc) ** 2 for x, y in coords) / m00
    m02 = sum((y - yc) ** 2 for x, y in coords) / m00
    m11 = sum((x - xc) * (y - yc) for x, y in coords) / m00

    return {"m20": m20, "m02": m02, "m11": m11}



#  PROIECȚII — distribuția pe axe


def calculeaza_proiectii(img):
    """
    Calculează proiecțiile orizontală și verticală ale obiectului.

    Proiecția verticală pv[x]  = nr. pixeli obiect pe coloana x
    Proiecția orizontală ph[y] = nr. pixeli obiect pe linia y

    Util pentru: detectarea marginilor, segmentarea textului,
                 analiza distribuției spațiale a obiectului.

    Returnează dict {pv: list, ph: list} sau None.

    Pseudocod:
      pv = [0] * w; ph = [0] * h
      pentru fiecare pixel obiect (x,y):
        pv[x] += 1; ph[y] += 1
    """
    data = detecteaza_pixeli_obiect(img)
    if data is None:
        return None

    w, h = data["w"], data["h"]
    coords = data["coords"]

    pv = [0] * w   # Proiecție verticală (pe coloane)
    ph = [0] * h   # Proiecție orizontală (pe linii)

    for x, y in coords:
        pv[x] += 1  # Incrementăm coloana x
        ph[y] += 1  # Incrementăm linia y

    return {"pv": pv, "ph": ph, "w": w, "h": h}


#  SNR — Raport Semnal/Zgomot (Signal-to-Noise Ratio)

def calculeaza_snr_singura(img):
    """
    Calculează SNR pentru o singură imagine.

    Semnal = media intensității pixelilor (canalul R)
    Zgomot = media lui |255 - intensitate| (distanța față de saturație)

    Formula: SNR = 10 * log10(semnal_mediu² / zgomot_mediu²)  [dB]

    Un SNR mare înseamnă imagine cu contrast ridicat (mult semnal util).
    Un SNR mic înseamnă imagine aproape gri sau cu mult zgomot.

    Returnează valoarea SNR în dB (float) sau None dacă zgomotul = 0.
    """
    src = img.convert("RGB")
    w, h = src.size
    px = src.load()

    suma_semnal = 0
    suma_zgomot = 0
    n = w * h

    for y in range(h):
        for x in range(w):
            s = px[x, y][0]          # Canalul R ca semnal
            z = abs(255 - s)         # Distanța față de alb = zgomot estimat
            suma_semnal += s
            suma_zgomot += z

    media_s = suma_semnal / n
    media_z = suma_zgomot / n

    if media_z == 0:
        return float('inf')  # Imagine perfect albă = zgomot zero

    return 10 * math.log10((media_s ** 2) / (media_z ** 2))


def calculeaza_snr_doua(img1, img2):
    """
    Calculează SNR comparând două imagini (originală vs. procesată).

    Semnal = diferența absolută între pixelii corespunzători
    Zgomot = valoarea absolută a pixelilor din imaginea originală

    Formula: SNR = 10 * log10(media_diferenta² / media_original²)  [dB]

    Util pentru a evalua calitatea unui filtru sau algoritm de compresie:
    un SNR mai mare înseamnă că diferența introdusă este mică față de semnal.

    Returnează valoarea SNR în dB (float) sau None.

    """
    src1 = img1.convert("RGB")
    src2 = img2.convert("RGB")

    # Verificăm că imaginile au aceleași dimensiuni
    w1, h1 = src1.size
    w2, h2 = src2.size
    w = min(w1, w2)
    h = min(h1, h2)

    px1 = src1.load()
    px2 = src2.load()

    suma_semnal = 0
    suma_zgomot = 0
    n = w * h

    for y in range(h):
        for x in range(w):
            semnal = abs(px1[x, y][0] - px2[x, y][0])  # Diferența R
            zgomot = abs(px1[x, y][0])                  # Referința originală
            suma_semnal += semnal
            suma_zgomot += zgomot

    media_s = suma_semnal / n
    media_z = suma_zgomot / n

    if media_z == 0 or media_s == 0:
        return float('inf')

    return 10 * math.log10((media_s ** 2) / (media_z ** 2))


#  RAPORT COMPLET DE ANALIZĂ

def raport_complet(img):
    """
    Generează un raport complet de analiză pentru imaginea dată.

    Calculează toate metricile disponibile și le returnează ca dict:
        - suprafata: numărul de pixeli obiect
        - centru: (xc, yc) centrul de greutate
        - momente: (mu20, mu02, mu11) momentele centrale
        - covarianta: (m20, m02, m11) matricea de covarianță
        - snr_singur: SNR al imaginii curente
        - bbox: bounding box al obiectului (xmin, ymin, xmax, ymax)

    Returnează dict cu toate rezultatele sau None dacă nu există obiect.
    """
    data = detecteaza_pixeli_obiect(img)
    if data is None:
        return None

    coords = data["coords"]
    m00 = data["m00"]
    xc, yc = data["xc"], data["yc"]

    # Momente centrale
    mu20 = sum((x - xc) ** 2 for x, y in coords)
    mu02 = sum((y - yc) ** 2 for x, y in coords)
    mu11 = sum((x - xc) * (y - yc) for x, y in coords)

    # Matricea de covarianță normalizată
    m20 = mu20 / m00
    m02 = mu02 / m00
    m11 = mu11 / m00

    # Bounding box
    xs = [x for x, y in coords]
    ys = [y for x, y in coords]
    bbox = (min(xs), min(ys), max(xs), max(ys))

    # SNR
    snr = calculeaza_snr_singura(img)

    # Orientarea estimată (unghiul axei principale din matricea de covarianță)
    # Folosim formula: theta = 0.5 * atan2(2*m11, m20-m02)
    orientare_rad = 0.5 * math.atan2(2 * m11, m20 - m02)
    orientare_grade = math.degrees(orientare_rad)

    return {
        "suprafata":      m00,
        "centru":         (round(xc, 2), round(yc, 2)),
        "momente":        (round(mu20, 1), round(mu02, 1), round(mu11, 1)),
        "covarianta":     (round(m20, 2), round(m02, 2), round(m11, 2)),
        "snr_singur":     round(snr, 2) if snr != float('inf') else "∞",
        "bbox":           bbox,
        "orientare_grade": round(orientare_grade, 1),
    }



#  ETICHETARE COMPONENTE 


def label_connected_components(img):
    """
    Etichetează componentele conexe ale obiectelor din imagine (BFS/flood-fill).

    Algoritmul:
      1. Convertim imaginea la grayscale
      2. Parcurgem fiecare pixel; dacă e întunecat (< 128) și neetichetat:
         - Pornim BFS din acel pixel
         - Propagăm eticheta la toți vecinii 8-conectați întunecați
      3. Rezultat: matrice labels[y][x] = numărul obiectului (0 = fundal)

    Returnează (labels, num_labels).
    """
    gray = img.convert("L")
    w, h = gray.size
    pix = gray.load()
    labels = [[0] * w for _ in range(h)]
    label = 0

    for i in range(h):
        for j in range(w):
            # Pixel întunecat (obiect) și neetichetat
            if pix[j, i] < 128 and labels[i][j] == 0:
                label += 1
                labels[i][j] = label
                queue = deque([(i, j)])

                while queue:
                    qi, qj = queue.popleft()
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            ni, nj = qi + di, qj + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                if pix[nj, ni] < 128 and labels[ni][nj] == 0:
                                    labels[ni][nj] = label
                                    queue.append((ni, nj))

    return labels, label


def generate_label_colors(num_labels):
    """
    Generează culori distincte pentru fiecare etichetă.
    Folosește metoda unghiului de aur (137.508°) pentru distribuție uniformă în spațiul HSV.
    """
    colors = {0: (255, 255, 255)}  # Fundal = alb
    for lbl in range(1, num_labels + 1):
        hue = (lbl * 137.508) % 360
        h60 = hue / 60.0
        i = int(h60) % 6
        f = h60 - int(h60)
        q, t = int((1 - f) * 255), int(f * 255)
        palette = [
            (255, t, 0), (q, 255, 0), (0, 255, t),
            (0, q, 255), (t, 0, 255), (255, 0, q)
        ]
        colors[lbl] = palette[i]
    return colors


def render_labeled_image(labels, colors, w, h):
    """Redă imaginea colorată pe baza matricei de etichete."""
    result = Image.new("RGB", (w, h), (255, 255, 255))
    pix = result.load()
    for i in range(h):
        for j in range(w):
            pix[j, i] = colors.get(labels[i][j], (255, 255, 255))
    return result


def extract_object_mask(labels, target_label, w, h):
    """
    Extrage masca unui obiect specific pe baza etichetei sale.
    Returnează imaginea mască (alb/negru) și lista de coordonate.
    """
    mask_img = Image.new("RGB", (w, h), (255, 255, 255))
    pix = mask_img.load()
    coords = []
    for i in range(h):
        for j in range(w):
            if labels[i][j] == target_label:
                pix[j, i] = (0, 0, 0)  # Obiectul selectat = negru
                coords.append((j, i))
    return mask_img, coords


def compute_sobel_orientation(mask_img):
    """
    Calculează direcția de alungire a obiectului folosind operatorul Sobel.
    Găsește pixelul cu magnitudinea maximă și returnează unghiul gradientului.
    """
    gray = mask_img.convert("L")
    w, h = gray.size
    pix = gray.load()
    max_mag = 0.0
    orientation = 0.0
    peak_x = peak_y = 0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # Gradienții Sobel pe X și Y
            gx = (pix[x+1, y-1] + 2*pix[x+1, y] + pix[x+1, y+1]
                  - pix[x-1, y-1] - 2*pix[x-1, y] - pix[x-1, y+1])
            gy = (pix[x-1, y+1] + 2*pix[x, y+1] + pix[x+1, y+1]
                  - pix[x-1, y-1] - 2*pix[x, y-1] - pix[x+1, y-1])
            mag = math.sqrt(gx * gx + gy * gy)
            if mag > max_mag:
                max_mag = mag
                orientation = math.atan2(gy, gx)
                peak_x, peak_y = x, y

    return math.degrees(orientation), max_mag, peak_x, peak_y