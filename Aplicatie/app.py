"""
app.py — Advanced Bitmap Editor
================================
Aplicație tkinter pentru procesarea imaginilor cu design dark/profesional.

Meniuri disponibile:
  📁 Fisier        — Încărcare, Export, Salveaza/Reseteaza, LZW (compress/decompress)
  ✨ Filtre        — Grayscale x3, Binarizare, Negativ, CMYK, YUV, YCbCr, HSV, RGB back
                     Mediere, Median, Minim, Maxim, Accentuare
  🔬 Lab 6 Filtre  — Laplacian, Eliminare zgomot Gaussian, LoG
  🧵 Contururi     — Vertical, Orizontal, Sobel V/H, Scharr V/H, Canny
  🧪 Transformări  — Fourier, Floyd-Steinberg, LZW compress/decompress, Huffman, RLE
  📊 Histogramă    — Afișare histogramă (R/G/B/Gray), Egalizare
  🔍 Analiză       — Centru M1, Momente M2, Covarianță, Proiecții, SNR, Raport complet
  🏷  Lab 4        — Etichetare BFS, Selectare obiect, Direcție Sobel
  🔬 Lab 5 Morfo   — Dilatare, Eroziune, Deschidere, Închidere

Structura UI:
  - Top bar cu dropdown-uri (navbar dark)
  - Sidebar stânga (parametri dinamici per operație selectată)
  - Canvas central (imaginea curentă)
  - Status bar jos

Autori: refactorizat complet din structura originală
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math
from collections import deque

# Importăm modulele de filtre, transformări și analiză
from filtre import *
from transformari import *
from analiza import *


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Construire dropdown personalizat
# ══════════════════════════════════════════════════════════════════════════════

def make_dropdown_button(parent, root, label, color, items):
    """
    Construiește un buton cu meniu dropdown personalizat (dark theme).

    Parametri:
        parent — frame-ul în care se plasează butonul
        root   — fereastra principală (pentru poziționare dropdown)
        label  — textul butonului
        color  — culoarea de fundal a butonului
        items  — list de: (text, callback) | None (separator)
    """
    state = {"visible": False}

    # Frame-ul dropdown (invizibil inițial)
    dropdown = tk.Frame(root, bg="#1e1e2e", bd=1, relief=tk.SOLID)

    for item in items:
        if item is None:
            # Separator orizontal
            tk.Frame(dropdown, bg="#333355", height=1).pack(
                fill=tk.X, padx=6, pady=2)
            continue
        text, cmd = item
        tk.Button(
            dropdown, text=text,
            bg="#1e1e2e", fg="#cdd6f4",
            activebackground="#313244", activeforeground="white",
            anchor="w", padx=18, pady=5,
            borderwidth=0, cursor="hand2",
            font=("Segoe UI", 9),
            command=lambda c=cmd: [
                dropdown.place_forget(),
                state.__setitem__("visible", False),
                c()
            ]
        ).pack(fill=tk.X)

    def toggle():
        """Afișează/ascunde dropdown-ul la click pe buton."""
        if state["visible"]:
            dropdown.place_forget()
            state["visible"] = False
        else:
            root.update_idletasks()
            x = btn.winfo_rootx() - root.winfo_rootx()
            y = btn.winfo_rooty() - root.winfo_rooty() + btn.winfo_height()
            dropdown.place(x=x, y=y, width=280)
            dropdown.lift()
            state["visible"] = True

    def close_on_outside(event):
        """Închide dropdown-ul dacă utilizatorul dă click în altă parte."""
        if state["visible"]:
            wx = dropdown.winfo_rootx()
            wy = dropdown.winfo_rooty()
            ww = dropdown.winfo_width()
            wh = dropdown.winfo_height()
            if not (wx <= event.x_root <= wx + ww and wy <= event.y_root <= wy + wh):
                dropdown.place_forget()
                state["visible"] = False

    root.bind("<Button-1>", close_on_outside, add="+")

    btn = tk.Button(
        parent, text=label, bg=color, fg="white",
        padx=13, pady=0, borderwidth=0, cursor="hand2",
        font=("Segoe UI", 9), command=toggle
    )
    btn.pack(side=tk.LEFT, padx=3, pady=10)
    return btn


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCȚII AJUTĂTOARE: Lab 4 — Etichetare BFS + Sobel
# ══════════════════════════════════════════════════════════════════════════════

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

                # BFS: propagăm eticheta la toți vecinii 8-conectați
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


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCȚII: Lab 5 — Egalizare histogramă + Morfologie
# ══════════════════════════════════════════════════════════════════════════════

def equalize_histogram(img):
    """
    Egalizează histograma imaginii pentru îmbunătățirea contrastului.

    Algoritmul:
      1. Calculăm histograma h[i] pentru fiecare nivel de gri i ∈ [0,255]
      2. Calculăm histograma cumulativă hc[i]
      3. Transformarea: T(i) = (hc[i] - hc[0]) * 255 / (N - hc[0])
         unde N = numărul total de pixeli
    """
    gray = img.convert("L")
    w, h = gray.size
    pix = gray.load()
    N = w * h

    # Calculăm histograma
    hist = [0] * 256
    for y in range(h):
        for x in range(w):
            hist[pix[x, y]] += 1

    # Histograma cumulativă
    hc = [0] * 256
    hc[0] = hist[0]
    for i in range(1, 256):
        hc[i] = hc[i-1] + hist[i]

    # Transformarea de egalizare
    hc0 = hc[0]
    denom = max(N - hc0, 1)
    T = [int(round((hc[i] - hc0) * 255 / denom)) for i in range(256)]

    # Aplicăm transformarea
    result = Image.new("L", (w, h))
    rpix = result.load()
    for y in range(h):
        for x in range(w):
            rpix[x, y] = T[pix[x, y]]
    return result.convert("RGB")


def _to_binary(img, threshold=128):
    """Convertește imaginea la matrice binară (1=obiect negru, 0=fundal alb)."""
    gray = img.convert("L")
    w, h = gray.size
    pix = gray.load()
    return [[1 if pix[x, y] < threshold else 0
             for x in range(w)] for y in range(h)], w, h


def _from_binary(matrix, w, h):
    """Reconstruiește imaginea PIL dintr-o matrice binară."""
    result = Image.new("L", (w, h), 255)
    pix = result.load()
    for y in range(h):
        for x in range(w):
            if matrix[y][x] == 1:
                pix[x, y] = 0
    return result.convert("RGB")


def dilate(img, kernel_size=3, iterations=1):
    """
    Operația de dilatare morfologică.
    Extinde obiectele (pixeli negri) cu kernel_size x kernel_size pe iterations iterații.
    Un pixel devine obiect dacă cel puțin un vecin din kernelul structural e obiect.
    """
    matrix, w, h = _to_binary(img)
    half = kernel_size // 2
    for _ in range(iterations):
        new_m = [[0] * w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                found = False
                for ky in range(-half, half + 1):
                    for kx in range(-half, half + 1):
                        ny, nx = y + ky, x + kx
                        if 0 <= ny < h and 0 <= nx < w and matrix[ny][nx] == 1:
                            found = True
                            break
                    if found:
                        break
                new_m[y][x] = 1 if found else 0
        matrix = new_m
    return _from_binary(matrix, w, h)


def erode(img, kernel_size=3, iterations=1):
    """
    Operația de eroziune morfologică.
    Subțiază obiectele (pixeli negri). Un pixel rămâne obiect doar dacă
    toți vecinii din kernelul structural sunt și ei obiect.
    """
    matrix, w, h = _to_binary(img)
    half = kernel_size // 2
    for _ in range(iterations):
        new_m = [[0] * w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                if matrix[y][x] == 0:
                    continue
                all_one = True
                for ky in range(-half, half + 1):
                    for kx in range(-half, half + 1):
                        ny, nx = y + ky, x + kx
                        if not (0 <= ny < h and 0 <= nx < w) or matrix[ny][nx] == 0:
                            all_one = False
                            break
                    if not all_one:
                        break
                new_m[y][x] = 1 if all_one else 0
        matrix = new_m
    return _from_binary(matrix, w, h)


def opening(img, kernel_size=3, iterations=1):
    """
    Deschidere morfologică = Eroziune urmată de Dilatare.
    Elimină zgomotul mic (obiecte mici) fără a modifica semnificativ forma.
    """
    result = img
    for _ in range(iterations):
        result = erode(result, kernel_size, 1)
        result = dilate(result, kernel_size, 1)
    return result


def closing(img, kernel_size=3, iterations=1):
    """
    Închidere morfologică = Dilatare urmată de Eroziune.
    Umple găurile mici din obiecte fără a modifica semnificativ forma.
    """
    result = img
    for _ in range(iterations):
        result = dilate(result, kernel_size, 1)
        result = erode(result, kernel_size, 1)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  CLASA PRINCIPALĂ — AdvancedBitmapEditor
# ══════════════════════════════════════════════════════════════════════════════

class AdvancedBitmapEditor:
    """
    Editor avansat de imagini bitmap cu interfață dark/profesională.

    Stare internă:
        original_img  — imaginea de bază (după ultimul Salveaza)
        display_img   — imaginea afișată curent (preview sau original)
        active_mode   — operația curentă activă (pentru live-preview cu slidere)
        _labels/_num_labels/_label_colors — stare Lab 4 (etichetare)
        _lzw_data     — datele LZW comprimate pentru decomprimare ulterioară
        _huffman_data — datele Huffman comprimate
        _rle_data     — datele RLE comprimate
        vars          — dicționar de tk.Var pentru toți parametrii ajustabili
    """

    # Paleta implicită Floyd-Steinberg
    FS_PALETTE = [
        (0, 0, 0), (255, 255, 255),
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255)
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Bitmap Editor — Lab Procesare Imagini")
        self.root.geometry("1600x960")
        self.root.configure(bg="#0d0d14")

        # Starea imaginilor
        self.original_img = None
        self.display_img  = None
        self.active_mode  = "none"

        # Starea Lab 4
        self._labels       = None
        self._num_labels   = 0
        self._label_colors = {}

        # Starea compresi
        self._lzw_data     = None    # (coduri, w, h) pentru decomprimare LZW
        self._huffman_data = None    # (coduri, bit_string, w, h) Huffman
        self._rle_data     = None    # (encoded, w, h) RLE

        # Starea Canny (parametri ajustabili)
        # Toți parametrii sunt stocați ca tk.Var și legați la slidere/spinbox
        self.vars = {
            # Parametri globali
            "bright":        tk.DoubleVar(value=1.0),
            "contrast":      tk.DoubleVar(value=1.0),
            # Canale RGB
            "r": tk.DoubleVar(value=1.0), "g": tk.DoubleVar(value=1.0),
            "b": tk.DoubleVar(value=1.0),
            # CMYK
            "c": tk.DoubleVar(value=1.0), "m": tk.DoubleVar(value=1.0),
            "y_c": tk.DoubleVar(value=1.0), "k": tk.DoubleVar(value=1.0),
            # YUV
            "y_luma": tk.DoubleVar(value=1.0),
            "u":  tk.DoubleVar(value=1.0), "v":  tk.DoubleVar(value=1.0),
            # YCbCr
            "cb": tk.DoubleVar(value=1.0), "cr": tk.DoubleVar(value=1.0),
            # Gray
            "gray_gain": tk.DoubleVar(value=1.0),
            # Binarizare
            "thresh": tk.DoubleVar(value=128),
            # HSV
            "h": tk.DoubleVar(value=1.0),
            "s": tk.DoubleVar(value=1.0),
            "v_hsv": tk.DoubleVar(value=1.0),
            # Accentuare
            "alpha": tk.DoubleVar(value=0.6),
            # Morfologie
            "morph_k":  tk.IntVar(value=3),
            "morph_it": tk.IntVar(value=1),
            # Lab 4
            "label_sel": tk.IntVar(value=1),
            # Canny
            "canny_low":  tk.IntVar(value=50),
            "canny_high": tk.IntVar(value=150),
            "canny_iter": tk.IntVar(value=1),
            # SNR (pentru compararea cu imaginea originală memorată)
        }

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    #  CONSTRUIRE INTERFAȚĂ
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        """Construiește întreaga interfață: top bar, sidebar, canvas, status bar."""

        # ── Top navigation bar ──────────────────────────────────────────────
        top_bar = tk.Frame(self.root, bg="#161622", height=56)
        top_bar.pack(side=tk.TOP, fill=tk.X)


        tk.Frame(top_bar, bg="#333355", width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=4)

        # ── Meniu Fisier ────────────────────────────────────────────────────
        make_dropdown_button(top_bar, self.root, "📁 Fișier ▾", "#222233", [
            ("📂  Încarcă imagine",           self.load_image),
            ("💾  Exportă (Save As)",         self.export_to_disk),
            None,
            ("✅  Salvează modificările",      self.confirm_save),
            ("❌  Resetează imaginea",         self.cancel_edits),
            None,
            ("🗜  Comprimă LZW → .lzw",       self._compress_lzw_file),
            ("📂  Deschide fișier .lzw",       self._open_lzw_file),
        ])

        # ── Meniu Filtre clasice ─────────────────────────────────────────────
        make_dropdown_button(top_bar, self.root, "✨ Filtre ▾", "#1a2a3a", [
            ("⬛  Negativ",                    lambda: self.preview_filter("negative")),
            ("🔲  Binarizare",                 lambda: self.preview_filter("binarize")),
            None,
            ("🩶  Gray (medie)",               lambda: self.preview_filter("gray(1)")),
            ("🩶  Gray (luminanță)",            lambda: self.preview_filter("gray(2)")),
            ("🩶  Gray (desaturat)",            lambda: self.preview_filter("gray(3)")),
            None,
            ("🎨  CMYK",                       lambda: self.preview_filter("cmyk")),
            ("📡  YUV",                        lambda: self.preview_filter("yuv")),
            ("📺  YCbCr",                      lambda: self.preview_filter("ycbcr")),
            ("🔄  RGB Back (inv. YCbCr)",      lambda: self.preview_filter("rgb_back")),
            ("🌈  HSV",                        lambda: self.preview_filter("hsv")),
            None,
            ("〰  Mediere 3×3",                lambda: self.preview_filter("mediere")),
            ("📊  Median 3×3",                 lambda: self.preview_filter("median")),
            ("🔽  Minim 3×3",                  lambda: self.preview_filter("minim")),
            ("🔼  Maxim 3×3",                  lambda: self.preview_filter("maxim")),
            ("✨  Accentuare",                 lambda: self.preview_filter("accentuare")),
            ("🔲  Laplacian",                  lambda: self.preview_filter("laplacian")),
            ("🌫  Eliminare zgomot Gaussian",           lambda: self.preview_filter("gaussian_denoise")),
            ("🔗  LoG (Laplacian of Gaussian)",lambda: self.preview_filter("log")),
        ])


        # ── Meniu Detectie contur ────────────────────────────────────────────
        make_dropdown_button(top_bar, self.root, "🧵 Contururi ▾", "#2a1a3a", [
            ("↕  Contur Vertical",             lambda: self.preview_filter("contur_v")),
            ("↔  Contur Orizontal",            lambda: self.preview_filter("contur_h")),
            None,
            ("🔍  Sobel Vertical",             lambda: self.preview_filter("sobel_v")),
            ("🔍  Sobel Orizontal",            lambda: self.preview_filter("sobel_h")),
            None,
            ("⚡  Scharr Vertical",            lambda: self.preview_filter("scharr_v")),
            ("⚡  Scharr Orizontal",           lambda: self.preview_filter("scharr_h")),
            None,
            ("🎯  Canny Edge Detection",       lambda: self._apply_canny()),
        ])

        # ── Meniu Transformări ───────────────────────────────────────────────
        make_dropdown_button(top_bar, self.root, "🧪 Transformări ▾", "#3a1a2a", [
            ("📡  Transformata Fourier",       lambda: self._apply_transform("fourier")),
            ("🎨  Floyd-Steinberg Dithering",  lambda: self._apply_transform("floyd")),
            None,
            ("🗜  Comprimă LZW (preview)",     lambda: self._apply_transform("lzw_compress")),
            ("🔓  Decomprimă LZW",             lambda: self._apply_transform("lzw_decompress")),
            None,
            ("📦  Huffman Encode",             lambda: self._apply_transform("huffman_enc")),
            ("📦  Huffman Decode",             lambda: self._apply_transform("huffman_dec")),
            None,
            ("📏  RLE Encode",                 lambda: self._apply_transform("rle_enc")),
            ("📏  RLE Decode",                 lambda: self._apply_transform("rle_dec")),
        ])

        # ── Meniu Histogramă ─────────────────────────────────────────────────
        make_dropdown_button(top_bar, self.root, "📊 Histogramă ▾", "#1a3a3a", [
            ("📊  Afișează histograma",        self.show_histogram),
            ("📈  Egalizare histogramă",       lambda: self._apply_lab5("equalize")),
        ])

        # ── Meniu Analiza ────────────────────────────────────────────────────
        make_dropdown_button(top_bar, self.root, "🔍 Analiză ▾", "#2a2a1a", [
            ("📍  Centru greutate (M1)",        self.show_center_m1),
            ("📐  Momente centrale (M2)",       self.show_moments_m2),
            ("🧮  Matrice covarianță",          self.show_covariance),
            ("📉  Proiecții H/V",               self.show_projections),
            None,
            ("📶  SNR (imagine curentă)",       self.show_snr_single),
            ("📶  SNR (original vs. display)",  self.show_snr_two),
            None,
            ("📋  Raport complet",              self.show_full_report),
        ])

        # ── Meniu Lab 4 ──────────────────────────────────────────────────────
        make_dropdown_button(top_bar, self.root, "🏷  Lab 4 ▾", "#3a2a1a", [
            ("🏷  Etichetare BFS",              self._apply_labeling),
            ("🎯  Selectare obiect",            self._select_labeled_object),
            ("📐  Direcție Sobel",              self._apply_sobel),
        ])

        # ── Meniu Lab 5 Morfologie ────────────────────────────────────────────
        make_dropdown_button(top_bar, self.root, "🔬 Lab 5 ▾", "#3a1a1a", [
            ("⬛  Dilatare",                   lambda: self._apply_lab5("dilate")),
            ("⬜  Eroziune",                   lambda: self._apply_lab5("erode")),
            None,
            ("🔓  Deschidere (Ero→Dil)",        lambda: self._apply_lab5("opening")),
            ("🔒  Închidere (Dil→Ero)",         lambda: self._apply_lab5("closing")),
        ])

        # ── Corp principal ────────────────────────────────────────────────────
        main = tk.Frame(self.root, bg="#0d0d14")
        main.pack(expand=True, fill=tk.BOTH)

        # Canvas central (imaginea) — packat primul, sidebar va fi pe dreapta
        self.canvas = tk.Canvas(main, bg="#0d0d14", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH,
                         padx=16, pady=16)

        # Sidebar dreapta — ascuns inițial, apare doar când e selectată o operație
        self.sidebar = tk.Frame(main, bg="#111120", width=290)
        self.sidebar.pack_propagate(False)
        self._build_sidebar()
        # Nu facem pack() acum — sidebar-ul e ascuns la start

        # ── Status bar jos ────────────────────────────────────────────────────
        status = tk.Frame(self.root, bg="#0a0a18", height=30)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="  Nicio imagine încărcată.")
        tk.Label(
            status, textvariable=self.status_var,
            fg="#6c7086", bg="#0a0a18",
            font=("Consolas", 8), anchor="w"
        ).pack(side=tk.LEFT, padx=10, pady=5)

    def _build_sidebar(self):
        """
        Construiește sidebar-ul cu:
          - Titlul operației active
          - Butoane Salvează/Resetează
          - Secțiunea de parametri dinamici (se schimbă cu filtrul activ)
        Sidebar-ul e ascuns implicit și apare doar când e selectată o operație.
        """
        sb = self.sidebar

        # ── Titlu operație activă ─────────────────────────────────────────────
        self.sidebar_title_var = tk.StringVar(value="")
        self.sidebar_title_lbl = tk.Label(
            sb, textvariable=self.sidebar_title_var,
            fg="#89b4fa", bg="#111120",
            font=("Consolas", 9, "bold"),
            wraplength=270, justify="left"
        )
        self.sidebar_title_lbl.pack(pady=(12, 4), padx=10, anchor="w")

        # ── Control editare ───────────────────────────────────────────────────
        ctrl = tk.Frame(sb, bg="#111120", padx=8)
        ctrl.pack(fill=tk.X, padx=8, pady=(0, 6))

        tk.Button(
            ctrl, text="✅  Salvează modificările",
            command=self.confirm_save,
            bg="#1a3a1a", fg="#a6e3a1",
            font=("Segoe UI", 9, "bold"),
            borderwidth=0, cursor="hand2", pady=5
        ).pack(fill=tk.X, pady=2)

        tk.Button(
            ctrl, text="❌  Resetează imaginea",
            command=self.cancel_edits,
            bg="#3a1a1a", fg="#f38ba8",
            font=("Segoe UI", 9, "bold"),
            borderwidth=0, cursor="hand2", pady=5
        ).pack(fill=tk.X, pady=2)

        # Separator
        tk.Frame(sb, bg="#333355", height=1).pack(fill=tk.X, padx=8, pady=4)

        # ── Container dinamic — se reconstruiește la fiecare operație ──────────
        self.dyn_frame = tk.Frame(sb, bg="#111120", padx=8)
        self.dyn_frame.pack(fill=tk.X, expand=True)

        # Spinbox pentru etichetare (Lab 4) — creat o singură dată, ascuns implicit
        self._lab4_frame = tk.LabelFrame(
            sb, text=" 🏷  ETICHETARE ",
            fg="#b4befe", bg="#111120",
            font=("Consolas", 8, "bold"),
            padx=8, pady=4
        )
        lf = tk.Frame(self._lab4_frame, bg="#111120")
        lf.pack(fill=tk.X)
        tk.Label(lf, text="Etichetă:", fg="#7f849c", bg="#111120",
                 font=("Consolas", 8)).grid(row=0, column=0, sticky="w")
        self.label_spin = tk.Spinbox(
            lf, from_=1, to=999,
            textvariable=self.vars["label_sel"],
            width=5, bg="#21262d", fg="#cdd6f4",
            buttonbackground="#30363d", font=("Consolas", 9), relief=tk.FLAT
        )
        self.label_spin.grid(row=0, column=1, padx=4)
        tk.Label(
            self._lab4_frame, text="(rulează Etichetare mai întâi)",
            fg="#45475a", bg="#111120", font=("Consolas", 7)
        ).pack(anchor="w")

    def _make_slider(self, parent, label, key, lo=0.0, hi=2.0, res=0.05):
        """
        Creează un slider cu etichetă în frame-ul dat.
        La schimbare, se declanșează _on_slider_change() pentru live-preview.
        """
        tk.Label(
            parent, text=label, fg="#6c7086", bg="#111120",
            font=("Consolas", 8)
        ).pack(anchor="w")
        tk.Scale(
            parent, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
            variable=self.vars[key], bg="#111120", fg="#cdd6f4",
            highlightthickness=0, troughcolor="#313244",
            command=lambda _: self._on_slider_change()
        ).pack(fill=tk.X, pady=(0, 4))

    # ─────────────────────────────────────────────────────────────────────────
    #  SIDEBAR DINAMIC — se actualizează la fiecare filtru selectat
    # ─────────────────────────────────────────────────────────────────────────

    def update_sidebar_dynamic(self, mode):
        """
        Afișează/ascunde sidebar-ul și construiește parametrii specifici operației active.
        Dacă mode este 'none' sau None, sidebar-ul se ascunde complet.
        """
        # ── Mapare mod → (titlu, listă parametri) ─────────────────────────────
        SIDEBAR_CONFIG = {
            # Filtre cu parametri de canal
            "negative":  ("⬛ Negativ",
                          [("Canal R", "r"), ("Canal G", "g"), ("Canal B", "b")]),
            "gray(1)":   ("🩶 Grayscale (medie)",
                          [("Gray Gain", "gray_gain")]),
            "gray(2)":   ("🩶 Grayscale (luminanță)",
                          [("Gray Gain", "gray_gain")]),
            "gray(3)":   ("🩶 Grayscale (desaturat)",
                          [("Gray Gain", "gray_gain")]),
            "binarize":  ("🔲 Binarizare",
                          [("Prag", "thresh", 0, 255, 1)]),
            "cmyk":      ("🎨 CMYK",
                          [("Cyan", "c"), ("Magenta", "m"),
                           ("Yellow", "y_c"), ("Black K", "k")]),
            "yuv":       ("📡 YUV",
                          [("Luma Y", "y_luma"), ("Chroma U", "u"), ("Chroma V", "v")]),
            "ycbcr":     ("📺 YCbCr",
                          [("Luma Y", "y_luma"), ("Chroma Cb", "cb"), ("Chroma Cr", "cr")]),
            "rgb_back":  ("🔄 RGB Back",
                          [("Canal R", "r"), ("Canal G", "g"), ("Canal B", "b")]),
            "hsv":       ("🌈 HSV",
                          [("Nuanță H", "h"), ("Saturație S", "s"), ("Valoare V", "v_hsv")]),
            # Filtre fără parametri ajustabili — sidebar cu global
            "mediere":   ("〰 Mediere 3×3",       None),
            "median":    ("📊 Median 3×3",         None),
            "minim":     ("🔽 Minim 3×3",          None),
            "maxim":     ("🔼 Maxim 3×3",          None),
            "laplacian": ("🔲 Laplacian",           None),
            "gaussian_denoise": ("🌫 Denoise Gaussian", None),
            "log":       ("🔗 LoG",                None),
            "contur_v":  ("↕ Contur Vertical",     None),
            "contur_h":  ("↔ Contur Orizontal",    None),
            "sobel_v":   ("🔍 Sobel Vertical",     None),
            "sobel_h":   ("🔍 Sobel Orizontal",    None),
            "scharr_v":  ("⚡ Scharr Vertical",    None),
            "scharr_h":  ("⚡ Scharr Orizontal",   None),
            # Filtre cu parametri speciali
            "accentuare": ("✨ Accentuare",
                           [("Alpha", "alpha", 0.1, 3.0, 0.1)]),
            "canny":     ("🎯 Canny Edge Detection", "canny_special"),
            # Morfologie
            "dilate":    ("⬛ Dilatare",           "morph_special"),
            "erode":     ("⬜ Eroziune",           "morph_special"),
            "opening":   ("🔓 Deschidere",         "morph_special"),
            "closing":   ("🔒 Închidere",          "morph_special"),
            "equalize":  ("📈 Egalizare histogramă", None),
            # Lab 4
            "labeling":  ("🏷 Etichetare BFS",     "lab4_special"),
            # Transformări (fără parametri)
            "fourier":       ("📡 Transformata Fourier",    None),
            "floyd":         ("🎨 Floyd-Steinberg",         None),
            "lzw_compress":  ("🗜 Compresie LZW",           None),
            "lzw_decompress":("🔓 Decompresie LZW",         None),
            "huffman_enc":   ("📦 Huffman Encode",          None),
            "huffman_dec":   ("📦 Huffman Decode",          None),
            "rle_enc":       ("📏 RLE Encode",              None),
            "rle_dec":       ("📏 RLE Decode",              None),
        }

        # ── Ascunde sidebar dacă nu e nicio operație activă ───────────────────
        if not mode or mode == "none":
            self.sidebar.pack_forget()
            return

        cfg = SIDEBAR_CONFIG.get(mode.lower())
        if not cfg:
            self.sidebar.pack_forget()
            return

        title, params = cfg

        # ── Afișează sidebar-ul pe dreapta ────────────────────────────────────
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Actualizează titlul ───────────────────────────────────────────────
        self.sidebar_title_var.set(f"⚙ {title}")

        # ── Curăță frame-ul dinamic ───────────────────────────────────────────
        for w in self.dyn_frame.winfo_children():
            w.destroy()
        self._lab4_frame.pack_forget()

        # ── Construiește parametrii specifici ─────────────────────────────────
        if params is None:
            # Filtre fără parametri — afișăm luminozitate/contrast global
            glob = tk.LabelFrame(
                self.dyn_frame, text=" 🌟  GLOBAL ",
                fg="#89dceb", bg="#111120",
                font=("Consolas", 8, "bold"),
                padx=8, pady=4
            )
            glob.pack(fill=tk.X, pady=4)
            self._make_slider(glob, "Luminozitate", "bright", 0.0, 3.0, 0.05)
            self._make_slider(glob, "Contrast",     "contrast", 0.0, 3.0, 0.05)

        elif params == "canny_special":
            # Parametri speciali Canny
            canny_f = tk.LabelFrame(
                self.dyn_frame, text=" 🎯  CANNY ",
                fg="#89b4fa", bg="#111120",
                font=("Consolas", 8, "bold"),
                padx=8, pady=4
            )
            canny_f.pack(fill=tk.X, pady=4)
            for lbl, key, lo, hi in [
                ("Prag inferior", "canny_low",  0, 255),
                ("Prag superior", "canny_high", 0, 255),
            ]:
                tk.Label(canny_f, text=lbl, fg="#7f849c", bg="#111120",
                         font=("Consolas", 8)).pack(anchor="w")
                tk.Scale(
                    canny_f, from_=lo, to=hi, resolution=1, orient=tk.HORIZONTAL,
                    variable=self.vars[key], bg="#111120", fg="white",
                    highlightthickness=0, troughcolor="#313244"
                ).pack(fill=tk.X, pady=(0, 2))
            rf_c = tk.Frame(canny_f, bg="#111120")
            rf_c.pack(fill=tk.X)
            tk.Label(rf_c, text="Iterații:", fg="#7f849c", bg="#111120",
                     font=("Consolas", 8)).grid(row=0, column=0, sticky="w")
            tk.Spinbox(
                rf_c, from_=1, to=10, textvariable=self.vars["canny_iter"],
                width=4, bg="#21262d", fg="#cdd6f4",
                buttonbackground="#30363d", font=("Consolas", 9), relief=tk.FLAT
            ).grid(row=0, column=1, padx=4)

        elif params == "morph_special":
            # Parametri morfologie
            morph = tk.LabelFrame(
                self.dyn_frame, text=" 🔬  MORFOLOGIE ",
                fg="#fab387", bg="#111120",
                font=("Consolas", 8, "bold"),
                padx=8, pady=4
            )
            morph.pack(fill=tk.X, pady=4)
            rf = tk.Frame(morph, bg="#111120")
            rf.pack(fill=tk.X)
            tk.Label(rf, text="Kernel:", fg="#7f849c", bg="#111120",
                     font=("Consolas", 8)).grid(row=0, column=0, sticky="w")
            tk.Spinbox(
                rf, from_=3, to=15, increment=2,
                textvariable=self.vars["morph_k"],
                width=4, bg="#21262d", fg="#cdd6f4",
                buttonbackground="#30363d", font=("Consolas", 9), relief=tk.FLAT
            ).grid(row=0, column=1, padx=4)
            tk.Label(rf, text="Iterații:", fg="#7f849c", bg="#111120",
                     font=("Consolas", 8)).grid(row=0, column=2, sticky="w", padx=(8, 0))
            tk.Spinbox(
                rf, from_=1, to=20, textvariable=self.vars["morph_it"],
                width=4, bg="#21262d", fg="#cdd6f4",
                buttonbackground="#30363d", font=("Consolas", 9), relief=tk.FLAT
            ).grid(row=0, column=3, padx=4)

        elif params == "lab4_special":
            # Parametri Lab 4 etichetare
            self._lab4_frame.pack(fill=tk.X, padx=8, pady=4)

        else:
            # Listă de slidere standard
            lf_dyn = tk.LabelFrame(
                self.dyn_frame, text=" ⚙  PARAMETRI ",
                fg="#f9e2af", bg="#111120",
                font=("Consolas", 8, "bold"),
                padx=8, pady=4
            )
            lf_dyn.pack(fill=tk.X, pady=4)
            for item in params:
                if len(item) == 5:
                    self._make_slider(lf_dyn, item[0], item[1], item[2], item[3], item[4])
                else:
                    self._make_slider(lf_dyn, item[0], item[1])
            # Adăugăm și global (luminozitate/contrast) pentru filtrele cu parametri
            glob = tk.LabelFrame(
                self.dyn_frame, text=" 🌟  GLOBAL ",
                fg="#89dceb", bg="#111120",
                font=("Consolas", 8, "bold"),
                padx=8, pady=4
            )
            glob.pack(fill=tk.X, pady=4)
            self._make_slider(glob, "Luminozitate", "bright", 0.0, 3.0, 0.05)
            self._make_slider(glob, "Contrast",     "contrast", 0.0, 3.0, 0.05)

    # ─────────────────────────────────────────────────────────────────────────
    #  HELPERS GENERALE
    # ─────────────────────────────────────────────────────────────────────────

    def _require_image(self):
        """Verifică dacă există o imagine încărcată. Afișează avertisment dacă nu."""
        if not self.original_img:
            messagebox.showwarning("Atenție", "Încarcă mai întâi o imagine.")
            return False
        return True

    def _set_status(self, text):
        """Actualizează bara de status din josul ferestrei."""
        self.status_var.set(f"  {text}")
        self.root.update_idletasks()

    def _draw_on_canvas(self, img):
        """Desenează imaginea pe canvas, scalată să se încadreze."""
        wc = self.canvas.winfo_width()
        hc = self.canvas.winfo_height()
        if wc < 10:
            wc, hc = 900, 700
        tmp = img.copy()
        tmp.thumbnail((wc - 20, hc - 20), Image.LANCZOS)
        self.tk_main = ImageTk.PhotoImage(tmp)
        self.canvas.delete("all")
        self.canvas.create_image(wc // 2, hc // 2, image=self.tk_main)

    def render_main(self):
        """Redă imaginea curentă pe canvas."""
        if self.display_img:
            self._draw_on_canvas(self.display_img)

    def apply_math(self, img, mode):
        """
        Aplică filtrul identificat prin 'mode' pe imaginea dată.
        Aplică și luminozitate/contrast globale după filtrare.

        Filtrul poate fi:
          - din filters_map_img (operează pe imagine întreagă)
          - din filters_map (pixel-cu-pixel)
        """
        m = mode.lower()
        v = {k: var.get() for k, var in self.vars.items()}

        # Filtre care operează pe imaginea întreagă (kernel-based, etc.)
        if m in filters_map_img:
            return filters_map_img[m](img, v)

        # Filtre pixel-cu-pixel
        w, h = img.size
        pixels = img.load()
        res = Image.new("RGB", (w, h))
        new_pix = res.load()
        b_f = v["bright"]
        c_f = v["contrast"]

        process_func = filters_map.get(
            m,
            lambda r, g, b, v: (
                int(r * v["r"]), int(g * v["g"]), int(b * v["b"]))
        )

        for x in range(w):
            for y in range(h):
                r, g, b = pixels[x, y]
                nr, ng, nb = process_func(r, g, b, v)
                new_pix[x, y] = (
                    max(0, min(255, int(nr * b_f * c_f))),
                    max(0, min(255, int(ng * b_f * c_f))),
                    max(0, min(255, int(nb * b_f * c_f)))
                )
        return res

    def _on_slider_change(self):
        """
        Callback pentru schimbarea sliderelor.
        Dacă există un filtru activ, se re-aplică live pentru preview instant.
        """
        if self.active_mode not in ("none", "") and self.original_img:
            self.display_img = self.apply_math(
                self.original_img.copy(), self.active_mode)
            self.render_main()

    # ─────────────────────────────────────────────────────────────────────────
    #  FIȘIER
    # ─────────────────────────────────────────────────────────────────────────

    def load_image(self):
        """Deschide dialog și încarcă imaginea selectată."""
        path = filedialog.askopenfilename(
            filetypes=[
                ("Imagini", "*.bmp *.png *.jpg *.jpeg *.tiff"),
                ("Toate", "*.*")
            ]
        )
        if path:
            self.original_img = Image.open(path).convert("RGB")
            self.display_img  = self.original_img.copy()
            self._labels = None  # Resetăm etichetele la imagine nouă
            w, h = self.original_img.size
            self._set_status(
                f"Imagine încărcată: {path.split('/')[-1]}  |  {w}×{h} px"
            )
            self.active_mode = "none"
            self.sidebar.pack_forget()
            self.render_main()

    def confirm_save(self):
        """Confirmă modificările: display_img devine noua imagine de bază."""
        if self.display_img:
            self.original_img = self.display_img.copy()
            self._set_status("✅ Modificările au fost salvate ca imagine de bază.")

    def cancel_edits(self):
        """Resetează toate sliderele și revine la imaginea de bază."""
        defaults = {
            "bright": 1.0, "contrast": 1.0,
            "r": 1.0, "g": 1.0, "b": 1.0,
            "c": 1.0, "m": 1.0, "y_c": 1.0, "k": 1.0,
            "y_luma": 1.0, "u": 1.0, "v": 1.0,
            "cb": 1.0, "cr": 1.0,
            "gray_gain": 1.0, "thresh": 128,
            "h": 1.0, "s": 1.0, "v_hsv": 1.0,
            "alpha": 0.6,
            "morph_k": 3, "morph_it": 1,
            "label_sel": 1,
            "canny_low": 50, "canny_high": 150, "canny_iter": 1,
        }
        for k, val in defaults.items():
            try:
                self.vars[k].set(val)
            except Exception:
                pass
        if self.original_img:
            self.display_img = self.original_img.copy()
            self.active_mode = "none"
            self.sidebar.pack_forget()
            self.render_main()
            self._set_status("Resetat la imaginea salvată.")

    def export_to_disk(self):
        """Exportă imaginea curentă în format BMP sau PNG."""
        if not self.display_img:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".bmp",
            filetypes=[("Bitmap", "*.bmp"), ("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if path:
            self.display_img.save(path)
            self._set_status(f"Exportat: {path.split('/')[-1]}")

    # ─────────────────────────────────────────────────────────────────────────
    #  FILTRE CLASICE — preview live
    # ─────────────────────────────────────────────────────────────────────────

    def preview_filter(self, mode):
        """
        Aplică filtrul identificat prin 'mode' și afișează preview-ul.
        Actualizează sidebar-ul dinamic cu parametrii relevanți.
        """
        if not self._require_image():
            return
        self.active_mode = mode
        self.update_sidebar_dynamic(mode)
        self._set_status(f"⏳ Aplicare: {mode} ...")
        self.display_img = self.apply_math(self.original_img.copy(), mode)
        self.render_main()
        self._set_status(
            f"Preview: {mode}  —  Apasă 'Salvează' pentru a confirma sau 'Resetează' pentru a anula.")

    # ─────────────────────────────────────────────────────────────────────────
    #  CANNY EDGE DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_canny(self):
        """
        Aplică Canny edge detection cu parametrii din sidebar.
        Parametrii: prag inferior, prag superior, număr iterații.
        """
        if not self._require_image():
            return
        low  = self.vars["canny_low"].get()
        high = self.vars["canny_high"].get()
        it   = self.vars["canny_iter"].get()
        self._set_status(
            f"⏳ Canny (low={low}, high={high}, iter={it}) ...")
        res = canny_edge_detection(self.original_img, low=low, high=high, iterations=it)
        self.display_img = res
        self.active_mode = "canny"
        self.update_sidebar_dynamic("canny")
        self.render_main()
        self._set_status(
            f"✔  Canny aplicat — prag inferior={low}, prag superior={high}, iterații={it}")

    # ─────────────────────────────────────────────────────────────────────────
    #  TRANSFORMĂRI (Fourier, Floyd-Steinberg, LZW, Huffman, RLE)
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_transform(self, operation):
        """Aplică transformarea specificată și afișează rezultatul."""
        if not self._require_image():
            return

        names = {
            "fourier":       "Transformata Fourier",
            "floyd":         "Floyd-Steinberg Dithering",
            "lzw_compress":  "Compresie LZW",
            "lzw_decompress":"Decompresie LZW",
            "huffman_enc":   "Codificare Huffman",
            "huffman_dec":   "Decodificare Huffman",
            "rle_enc":       "Codificare RLE",
            "rle_dec":       "Decodificare RLE",
        }
        self._set_status(f"⏳ {names.get(operation, operation)} ...")

        src = self.original_img
        res = None

        if operation == "fourier":
            # Transformata Fourier — spectrul de frecvențe
            res = transformata_fourier(src)

        elif operation == "floyd":
            # Floyd-Steinberg cu paleta implicită
            res = floyd_steinberg(src, self.FS_PALETTE)

        elif operation == "lzw_compress":
            # Comprimăm imaginea și o decomprimăm imediat pentru preview
            codes, w, h = compress_image_lzw(src)
            self._lzw_data = (codes, w, h)
            ratio = len(list(src.convert("L").getdata())) / max(len(codes), 1)
            res = decompress_image_lzw(codes, w, h)
            self._set_status(
                f"✔  LZW comprimat — {len(codes)} coduri | ratio ≈ {ratio:.2f}x")
            self.display_img = res
            self.active_mode = operation
            self.render_main()
            return

        elif operation == "lzw_decompress":
            # Decomprimăm din datele comprimate anterior
            if not self._lzw_data:
                messagebox.showwarning("Atenție",
                    "Nu există date LZW comprimate.\n"
                    "Rulați 'Comprimă LZW' mai întâi.")
                return
            codes, w, h = self._lzw_data
            res = decompress_image_lzw(codes, w, h)
            self._set_status("✔  LZW decomprimate cu succes.")

        elif operation == "huffman_enc":
            # Huffman encode — returnăm vizualizarea decodificată
            codes, bit_string, w, h = huffman_encode(src)
            self._huffman_data = (codes, bit_string, w, h)
            res = huffman_decode(codes, bit_string, w, h)
            self._set_status(
                f"✔  Huffman encode — {len(bit_string)} biți | "
                f"{len(codes)} simboluri unice")

        elif operation == "huffman_dec":
            if not self._huffman_data:
                messagebox.showwarning("Atenție",
                    "Nu există date Huffman.\nRulați 'Huffman Encode' mai întâi.")
                return
            codes, bit_string, w, h = self._huffman_data
            res = huffman_decode(codes, bit_string, w, h)
            self._set_status("✔  Huffman decodat cu succes.")

        elif operation == "rle_enc":
            # RLE encode — vizualizăm rezultatul decodat
            encoded, w, h = rle_encode(src)
            self._rle_data = (encoded, w, h)
            res = rle_decode(encoded, w, h)
            total_px = w * h
            self._set_status(
                f"✔  RLE encode — {len(encoded)} perechi (față de {total_px} pixeli) | "
                f"ratio ≈ {total_px / max(len(encoded), 1):.2f}x")

        elif operation == "rle_dec":
            if not self._rle_data:
                messagebox.showwarning("Atenție",
                    "Nu există date RLE.\nRulați 'RLE Encode' mai întâi.")
                return
            encoded, w, h = self._rle_data
            res = rle_decode(encoded, w, h)
            self._set_status("✔  RLE decodat cu succes.")

        if res:
            self.display_img = res
            self.active_mode = operation
            self.update_sidebar_dynamic(operation)
            self.render_main()
            if operation not in ("lzw_compress",):
                self._set_status(
                    f"✔  {names.get(operation, operation)} aplicat.  "
                    "Apasă 'Salvează' pentru a confirma.")

    # ─────────────────────────────────────────────────────────────────────────
    #  LZW ca fișier (compress/decompress de pe disc)
    # ─────────────────────────────────────────────────────────────────────────

    def _compress_lzw_file(self):
        """
        Comprimă imaginea curentă și o salvează ca fișier .lzw binar.
        Formatul fișierului: width(4B) + height(4B) + coduri ca int-uri (4B fiecare).
        """
        import struct
        if not self._require_image():
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".lzw",
            filetypes=[("LZW Compressed", "*.lzw")]
        )
        if not path:
            return
        self._set_status("⏳ Compresie LZW ...")
        codes, w, h = compress_image_lzw(self.original_img)
        with open(path, "wb") as f:
            f.write(struct.pack("<II", w, h))  # Dimensiunile imaginii
            for code in codes:
                f.write(struct.pack("<I", code))  # Fiecare cod LZW
        ratio = w * h / max(len(codes), 1)
        self._set_status(
            f"✔  Salvat: {path.split('/')[-1]}  |  "
            f"{len(codes)} coduri  |  ratio ≈ {ratio:.2f}x")

    def _open_lzw_file(self):
        """
        Deschide un fișier .lzw, îl decomprimă și afișează imaginea.
        Formatul fișierului: width(4B) + height(4B) + coduri int-uri (4B).
        """
        import struct
        path = filedialog.askopenfilename(
            filetypes=[("LZW Compressed", "*.lzw"), ("Toate", "*.*")]
        )
        if not path:
            return
        self._set_status("⏳ Decompresie LZW ...")
        try:
            with open(path, "rb") as f:
                w, h = struct.unpack("<II", f.read(8))
                raw = f.read()
                codes = list(struct.unpack(f"<{len(raw)//4}I", raw))
            img = decompress_image_lzw(codes, w, h)
            self.original_img = img
            self.display_img  = img.copy()
            self._lzw_data = (codes, w, h)
            self.sidebar.pack_forget()
            self.render_main()
            self._set_status(
                f"✔  Decomprimit: {path.split('/')[-1]}  |  {w}×{h} px")
        except Exception as e:
            messagebox.showerror("Eroare LZW", str(e))

    # ─────────────────────────────────────────────────────────────────────────
    #  LAB 4 — Etichetare + Sobel
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_labeling(self):
        """Rulează etichetarea BFS și colorează obiectele distinct."""
        if not self._require_image():
            return
        self._set_status("⏳ Etichetare componente conexe (BFS) ...")
        self._labels, self._num_labels = label_connected_components(
            self.original_img)
        self._label_colors = generate_label_colors(self._num_labels)
        w, h = self.original_img.size
        result = render_labeled_image(
            self._labels, self._label_colors, w, h)
        self.display_img = result
        self.label_spin.config(to=max(self._num_labels, 1))
        self.update_sidebar_dynamic("labeling")
        self.render_main()
        self._set_status(
            f"✔  Etichetare completă — {self._num_labels} obiecte detectate.  "
            "Selectează eticheta din sidebar.")

    def _select_labeled_object(self):
        """Extrage și afișează masca obiectului cu eticheta selectată."""
        if not self._require_image():
            return
        if self._labels is None:
            messagebox.showwarning("Atenție",
                "Rulează mai întâi 'Etichetare BFS'.")
            return
        lbl = self.vars["label_sel"].get()
        if not (1 <= lbl <= self._num_labels):
            messagebox.showwarning("Atenție",
                f"Eticheta trebuie să fie între 1 și {self._num_labels}.")
            return
        w, h = self.original_img.size
        mask_img, coords = extract_object_mask(self._labels, lbl, w, h)
        # Contur colorat în culoarea etichetei
        draw = ImageDraw.Draw(mask_img)
        draw.rectangle(
            [0, 0, w - 1, h - 1],
            outline=self._label_colors.get(lbl, (255, 0, 0)), width=3
        )
        self.display_img = mask_img
        self.render_main()
        self._set_status(
            f"✔  Obiect {lbl} selectat — {len(coords)} pixeli.  "
            "Apasă Sobel pentru direcție.")

    def _apply_sobel(self):
        """
        Calculează direcția de alungire a obiectului selectat folosind Sobel.
        Afișează vectorul orientării pe masca obiectului.
        """
        if not self._require_image():
            return
        if self._labels is None:
            messagebox.showwarning("Atenție",
                "Rulează mai întâi 'Etichetare BFS'.")
            return
        lbl = self.vars["label_sel"].get()
        if not (1 <= lbl <= self._num_labels):
            messagebox.showwarning("Atenție",
                f"Eticheta trebuie să fie între 1 și {self._num_labels}.")
            return
        self._set_status("⏳ Calcul Sobel ...")
        w, h = self.original_img.size
        mask_img, _ = extract_object_mask(self._labels, lbl, w, h)
        deg, mag, px, py = compute_sobel_orientation(mask_img)

        # Desenăm punctul de magnitudine maximă și vectorul orientării
        annotated = mask_img.copy()
        draw = ImageDraw.Draw(annotated)
        draw.ellipse([px - 6, py - 6, px + 6, py + 6],
                     outline="red", width=2)
        lx = int(px + 50 * math.cos(math.radians(deg)))
        ly = int(py + 50 * math.sin(math.radians(deg)))
        draw.line([px, py, lx, ly], fill="red", width=2)

        self.display_img = annotated
        self.render_main()
        self._set_status(
            f"✔  Sobel — Direcție: {deg:.1f}°  |  "
            f"Magnitudine: {mag:.1f}  |  Peak: ({px},{py})")

    # ─────────────────────────────────────────────────────────────────────────
    #  LAB 5 — Morfologie + Egalizare
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_lab5(self, operation):
        """Aplică operația morfologică sau egalizarea histogramei."""
        if not self._require_image():
            return
        ks = self.vars["morph_k"].get()
        it = self.vars["morph_it"].get()
        names = {
            "equalize": "Egalizare histogramă",
            "dilate":   f"Dilatare (k={ks}, iter={it})",
            "erode":    f"Eroziune (k={ks}, iter={it})",
            "opening":  f"Deschidere (k={ks}, iter={it})",
            "closing":  f"Închidere (k={ks}, iter={it})",
        }
        self._set_status(f"⏳ {names.get(operation)} ...")
        src = self.original_img
        if   operation == "equalize": res = equalize_histogram(src)
        elif operation == "dilate":   res = dilate(src, ks, it)
        elif operation == "erode":    res = erode(src, ks, it)
        elif operation == "opening":  res = opening(src, ks, it)
        elif operation == "closing":  res = closing(src, ks, it)
        else:
            return
        self.display_img = res
        self.active_mode = operation
        self.update_sidebar_dynamic(operation)
        self.render_main()
        self._set_status(
            f"✔  {names.get(operation)} aplicat.  "
            "Apasă 'Salvează' pentru a confirma.")

    # ─────────────────────────────────────────────────────────────────────────
    #  HISTOGRAMĂ
    # ─────────────────────────────────────────────────────────────────────────

    def show_histogram(self):
        """
        Afișează histograma imaginii curente (R, G, B și grayscale)
        într-o fereastră matplotlib separată.
        """
        if not self.display_img:
            return

        img_rgb = self.display_img.convert("RGB")
        w_i, h_i = img_rgb.size
        px = img_rgb.load()

        # Colectăm valorile canalelor
        ch_r = [px[x, y][0] for y in range(h_i) for x in range(w_i)]
        ch_g = [px[x, y][1] for y in range(h_i) for x in range(w_i)]
        ch_b = [px[x, y][2] for y in range(h_i) for x in range(w_i)]
        ch_gr = [(r + g + b) // 3 for r, g, b in zip(ch_r, ch_g, ch_b)]

        fig, axes = plt.subplots(2, 2, figsize=(10, 6), facecolor="#1a1a2e")
        fig.suptitle("Histogramă Imagine", color="#cdd6f4", fontsize=12)

        configs = [
            (axes[0, 0], ch_r,  "#f38ba8", "Canal Roșu"),
            (axes[0, 1], ch_g,  "#a6e3a1", "Canal Verde"),
            (axes[1, 0], ch_b,  "#89b4fa", "Canal Albastru"),
            (axes[1, 1], ch_gr, "#cdd6f4", "Grayscale"),
        ]

        for ax, data, color, title in configs:
            ax.set_facecolor("#11111b")
            ax.hist(data, bins=256, color=color, alpha=0.8)
            ax.set_title(title, color="#cdd6f4", fontsize=9)
            ax.tick_params(colors="#6c7086")
            for spine in ax.spines.values():
                spine.set_color("#313244")

        plt.tight_layout()
        plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    #  ANALIZĂ OBIECT
    # ─────────────────────────────────────────────────────────────────────────

    def show_center_m1(self):
        """Calculează și afișează centrul de greutate (M1)."""
        if not self.display_img:
            return
        result = calculeaza_centru_greutate(self.display_img)
        if result is None:
            messagebox.showwarning("Analiză", "Nu s-a detectat niciun obiect.")
            return
        xc, yc = result
        # Desenăm pe imagine
        copie = deseneaza_centru(self.display_img, xc, yc)
        self._draw_on_canvas(copie)
        messagebox.showinfo(
            "Moment M1 — Centru de greutate",
            f"Centrul de greutate (M1):\n\n"
            f"  Xc = {xc:.2f} px\n"
            f"  Yc = {yc:.2f} px"
        )

    def show_moments_m2(self):
        """Calculează și afișează momentele centrale de ordinul 2."""
        if not self.display_img:
            return
        result = calculeaza_momente_centrale(self.display_img)
        if result is None:
            messagebox.showwarning("Analiză", "Nu s-a detectat niciun obiect.")
            return
        messagebox.showinfo(
            "Momente Centrale M2",
            f"Momentele centrale de ordinul 2:\n\n"
            f"  μ20 (variație X) = {result['mu20']:.0f}\n"
            f"  μ02 (variație Y) = {result['mu02']:.0f}\n"
            f"  μ11 (corelație)  = {result['mu11']:.0f}"
        )

    def show_covariance(self):
        """Calculează și afișează matricea de covarianță."""
        if not self.display_img:
            return
        result = calculeaza_covarianta(self.display_img)
        if result is None:
            messagebox.showwarning("Analiză", "Nu s-a detectat niciun obiect.")
            return
        m20, m02, m11 = result["m20"], result["m02"], result["m11"]
        messagebox.showinfo(
            "Matricea de Covarianță",
            f"Matricea de covarianță normalizată:\n\n"
            f"  | {m20:8.2f}   {m11:8.2f} |\n"
            f"  | {m11:8.2f}   {m02:8.2f} |\n\n"
            f"Diagonala principală: variație pe X și Y\n"
            f"Elementele off-diagonal: orientarea obiectului"
        )

    def show_projections(self):
        """Afișează proiecțiile orizontală și verticală într-un grafic."""
        if not self.display_img:
            return
        result = calculeaza_proiectii(self.display_img)
        if result is None:
            messagebox.showwarning("Analiză", "Nu s-a detectat niciun obiect.")
            return
        pv, ph = result["pv"], result["ph"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#1a1a2e")
        fig.suptitle("Proiecții Imagine", color="#cdd6f4")

        axes[0].set_facecolor("#11111b")
        axes[0].plot(pv, color="#89b4fa", linewidth=1)
        axes[0].set_title("Proiecție Verticală (pe coloane)",
                           color="#cdd6f4", fontsize=9)
        axes[0].tick_params(colors="#6c7086")

        axes[1].set_facecolor("#11111b")
        axes[1].plot(ph, color="#f38ba8", linewidth=1)
        axes[1].set_title("Proiecție Orizontală (pe linii)",
                           color="#cdd6f4", fontsize=9)
        axes[1].tick_params(colors="#6c7086")

        for ax in axes:
            for spine in ax.spines.values():
                spine.set_color("#313244")

        plt.tight_layout()
        plt.show()

    def show_snr_single(self):
        """Calculează și afișează SNR-ul imaginii curente (o singură imagine)."""
        if not self.display_img:
            return
        snr = calculeaza_snr_singura(self.display_img)
        if snr == float('inf'):
            val = "∞ (imagine perfect albă)"
        else:
            val = f"{snr:.2f} dB"
        messagebox.showinfo(
            "SNR — o singură imagine",
            f"Signal-to-Noise Ratio (imagine curentă):\n\n"
            f"  SNR = {val}\n\n"
            f"Interpretare:\n"
            f"  SNR > 10 dB  → semnal dominant\n"
            f"  SNR < 0  dB  → zgomot dominant"
        )

    def show_snr_two(self):
        """Calculează SNR comparând imaginea originală cu cea afișată."""
        if not self.original_img or not self.display_img:
            return
        snr = calculeaza_snr_doua(self.original_img, self.display_img)
        if snr == float('inf'):
            val = "∞ (imagini identice sau fără semnal)"
        else:
            val = f"{snr:.2f} dB"
        messagebox.showinfo(
            "SNR — original vs. procesat",
            f"Comparare Original ↔ Display:\n\n"
            f"  SNR = {val}\n\n"
            f"Măsoară diferența introdusă de procesare."
        )

    def show_full_report(self):
        """Afișează un raport complet de analiză într-o fereastră text."""
        if not self.display_img:
            return
        report = raport_complet(self.display_img)
        if report is None:
            messagebox.showwarning("Analiză", "Nu s-a detectat niciun obiect.")
            return

        w_i, h_i = self.display_img.size
        text = (
            f"{'═'*45}\n"
            f"  RAPORT COMPLET DE ANALIZĂ\n"
            f"{'═'*45}\n\n"
            f"  Dimensiune imagine:  {w_i} × {h_i} px\n"
            f"  Suprafață obiect:    {report['suprafata']} px\n\n"
            f"  Centru de greutate:\n"
            f"      Xc = {report['centru'][0]}\n"
            f"      Yc = {report['centru'][1]}\n\n"
            f"  Momente centrale (M2):\n"
            f"      μ20 = {report['momente'][0]}\n"
            f"      μ02 = {report['momente'][1]}\n"
            f"      μ11 = {report['momente'][2]}\n\n"
            f"  Matricea de covarianță:\n"
            f"      | {report['covarianta'][0]:8.2f}   {report['covarianta'][2]:8.2f} |\n"
            f"      | {report['covarianta'][2]:8.2f}   {report['covarianta'][1]:8.2f} |\n\n"
            f"  Bounding Box:\n"
            f"      ({report['bbox'][0]}, {report['bbox'][1]}) → "
            f"({report['bbox'][2]}, {report['bbox'][3]})\n\n"
            f"  Orientare (axă principală):\n"
            f"      θ = {report['orientare_grade']}°\n\n"
            f"  SNR (imagine curentă):\n"
            f"      {report['snr_singur']} dB\n"
            f"{'═'*45}"
        )

        # Fereastră separată cu text derulabil
        win = tk.Toplevel(self.root)
        win.title("Raport Complet Analiză")
        win.configure(bg="#111120")
        win.geometry("420x500")

        txt = scrolledtext.ScrolledText(
            win, bg="#0d0d14", fg="#cdd6f4",
            font=("Consolas", 10), wrap=tk.WORD,
            borderwidth=0
        )
        txt.pack(expand=True, fill=tk.BOTH, padx=12, pady=12)
        txt.insert("1.0", text)
        txt.config(state="disabled")


# ══════════════════════════════════════════════════════════════════════════════
#  PUNCT DE INTRARE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedBitmapEditor(root)
    root.mainloop()