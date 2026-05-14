import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import deque
from filtre import *  # Importa toate funcțiile din filtre.py
from transformari import *


def make_dropdown_button(parent, root, label, color, items):
    state = {"visible": False}
    dropdown = tk.Frame(root, bg="#1e1e2e", bd=1, relief=tk.SOLID)

    for item in items:
        if item is None:
            tk.Frame(dropdown, bg="#333355", height=1).pack(fill=tk.X, padx=6, pady=2)
            continue
        text, cmd = item
        tk.Button(
            dropdown, text=text,
            bg="#1e1e2e", fg="#cdd6f4",
            activebackground="#313244", activeforeground="white",
            anchor="w", padx=18, pady=5,
            borderwidth=0, cursor="hand2",
            font=("Segoe UI", 9),
            command=lambda c=cmd: [dropdown.place_forget(),
                                   state.__setitem__("visible", False), c()]
        ).pack(fill=tk.X)

    def toggle():
        if state["visible"]:
            dropdown.place_forget(); state["visible"] = False
        else:
            root.update_idletasks()
            x = btn.winfo_rootx() - root.winfo_rootx()
            y = btn.winfo_rooty() - root.winfo_rooty() + btn.winfo_height()
            dropdown.place(x=x, y=y, width=300)
            dropdown.lift(); state["visible"] = True

    def close_on_outside(event):
        if state["visible"]:
            wx, wy = dropdown.winfo_rootx(), dropdown.winfo_rooty()
            ww, wh = dropdown.winfo_width(), dropdown.winfo_height()
            if not (wx <= event.x_root <= wx+ww and wy <= event.y_root <= wy+wh):
                dropdown.place_forget(); state["visible"] = False

    root.bind("<Button-1>", close_on_outside, add="+")

    btn = tk.Button(parent, text=label, bg=color, fg="white",
                    padx=13, pady=0, borderwidth=0, cursor="hand2",
                    font=("Segoe UI", 9), command=toggle)
    btn.pack(side=tk.LEFT, padx=4, pady=10)
    return btn


# ══════════════════════════════════════════════════════════════════════════════
#  LAB 4 — Etichetare componente conexe + Sobel
# ══════════════════════════════════════════════════════════════════════════════

def label_connected_components(img):
    gray = img.convert("L")
    w, h = gray.size
    pix = gray.load()
    labels = [[0]*w for _ in range(h)]
    label = 0
    for i in range(h):
        for j in range(w):
            if pix[j, i] < 128 and labels[i][j] == 0:
                label += 1
                labels[i][j] = label
                queue = deque([(i, j)])
                while queue:
                    qi, qj = queue.popleft()
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            ni, nj = qi+di, qj+dj
                            if 0 <= ni < h and 0 <= nj < w:
                                if pix[nj, ni] < 128 and labels[ni][nj] == 0:
                                    labels[ni][nj] = label
                                    queue.append((ni, nj))
    return labels, label


def generate_label_colors(num_labels):
    colors = {0: (255, 255, 255)}
    for lbl in range(1, num_labels+1):
        hue = (lbl * 137.508) % 360
        h60 = hue / 60.0
        i = int(h60) % 6
        f = h60 - int(h60)
        q, t = int((1-f)*255), int(f*255)
        palette = [(255,t,0),(q,255,0),(0,255,t),(0,q,255),(t,0,255),(255,0,q)]
        colors[lbl] = palette[i]
    return colors


def render_labeled_image(labels, colors, w, h):
    result = Image.new("RGB", (w, h), (255, 255, 255))
    pix = result.load()
    for i in range(h):
        for j in range(w):
            pix[j, i] = colors.get(labels[i][j], (255, 255, 255))
    return result


def extract_object_mask(labels, target_label, w, h):
    mask_img = Image.new("RGB", (w, h), (255, 255, 255))
    pix = mask_img.load()
    coords = []
    for i in range(h):
        for j in range(w):
            if labels[i][j] == target_label:
                pix[j, i] = (0, 0, 0)
                coords.append((j, i))
    return mask_img, coords


def compute_sobel_orientation(mask_img):
    gray = mask_img.convert("L")
    w, h = gray.size
    pix = gray.load()
    max_mag = 0.0; orientation = 0.0; peak_x = peak_y = 0
    for y in range(1, h-1):
        for x in range(1, w-1):
            gx = (pix[x+1,y-1] + 2*pix[x+1,y] + pix[x+1,y+1]
                 -pix[x-1,y-1] - 2*pix[x-1,y] - pix[x-1,y+1])
            gy = (pix[x-1,y+1] + 2*pix[x,y+1] + pix[x+1,y+1]
                 -pix[x-1,y-1] - 2*pix[x,y-1] - pix[x+1,y-1])
            mag = math.sqrt(gx*gx + gy*gy)
            if mag > max_mag:
                max_mag = mag; orientation = math.atan2(gy, gx)
                peak_x, peak_y = x, y
    return math.degrees(orientation), max_mag, peak_x, peak_y


# ══════════════════════════════════════════════════════════════════════════════
#  LAB 5 — Egalizare histograma + Morfologie
# ══════════════════════════════════════════════════════════════════════════════

def equalize_histogram(img):
    gray = img.convert("L")
    w, h = gray.size
    pix = gray.load()
    N = w * h
    hist = [0] * 256
    for y in range(h):
        for x in range(w):
            hist[pix[x, y]] += 1
    hc = [0] * 256
    hc[0] = hist[0]
    for i in range(1, 256):
        hc[i] = hc[i-1] + hist[i]
    hc0 = hc[0]
    denom = max(N - hc0, 1)
    T = [int(round((hc[i] - hc0) * 255 / denom)) for i in range(256)]
    result = Image.new("L", (w, h))
    rpix = result.load()
    for y in range(h):
        for x in range(w):
            rpix[x, y] = T[pix[x, y]]
    return result.convert("RGB")


def _to_binary(img, threshold=128):
    gray = img.convert("L")
    w, h = gray.size
    pix = gray.load()
    return [[1 if pix[x,y] < threshold else 0 for x in range(w)] for y in range(h)], w, h


def _from_binary(matrix, w, h):
    result = Image.new("L", (w, h), 255)
    pix = result.load()
    for y in range(h):
        for x in range(w):
            if matrix[y][x] == 1:
                pix[x, y] = 0
    return result.convert("RGB")


def dilate(img, kernel_size=3, iterations=1):
    matrix, w, h = _to_binary(img)
    half = kernel_size // 2
    for _ in range(iterations):
        new_m = [[0]*w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                found = False
                for ky in range(-half, half+1):
                    for kx in range(-half, half+1):
                        ny, nx = y+ky, x+kx
                        if 0 <= ny < h and 0 <= nx < w and matrix[ny][nx] == 1:
                            found = True; break
                    if found: break
                new_m[y][x] = 1 if found else 0
        matrix = new_m
    return _from_binary(matrix, w, h)


def erode(img, kernel_size=3, iterations=1):
    matrix, w, h = _to_binary(img)
    half = kernel_size // 2
    for _ in range(iterations):
        new_m = [[0]*w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                if matrix[y][x] == 0:
                    continue
                all_one = True
                for ky in range(-half, half+1):
                    for kx in range(-half, half+1):
                        ny, nx = y+ky, x+kx
                        if not (0 <= ny < h and 0 <= nx < w) or matrix[ny][nx] == 0:
                            all_one = False; break
                    if not all_one: break
                new_m[y][x] = 1 if all_one else 0
        matrix = new_m
    return _from_binary(matrix, w, h)


def opening(img, kernel_size=3, iterations=1):
    result = img
    for _ in range(iterations):
        result = erode(result, kernel_size, 1)
        result = dilate(result, kernel_size, 1)
    return result


def closing(img, kernel_size=3, iterations=1):
    result = img
    for _ in range(iterations):
        result = dilate(result, kernel_size, 1)
        result = erode(result, kernel_size, 1)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  CLASA PRINCIPALA
# ══════════════════════════════════════════════════════════════════════════════

class AdvancedBitmapEditor:

    FS_PALETTE = [
        (0,0,0), (255,255,255),
        (255,0,0), (0,255,0), (0,0,255),
        (255,255,0), (0,255,255), (255,0,255)
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Bitmap Editor")
        self.root.geometry("1550x950")
        self.root.configure(bg="#121212")

        self.original_img  = None   # ultima versiune confirmata (Salveaza)
        self.display_img   = None   # versiunea afisata curent (poate fi preview)

        self.active_mode   = "none"
        self.previews      = {}

        # Starea Lab 4
        self._labels       = None
        self._num_labels   = 0
        self._label_colors = {}

        self.vars = {
            "bright":    tk.DoubleVar(value=1.0),
            "contrast":  tk.DoubleVar(value=1.0),
            "r":  tk.DoubleVar(value=1.0), "g": tk.DoubleVar(value=1.0),
            "b":  tk.DoubleVar(value=1.0),
            "c":  tk.DoubleVar(value=1.0), "m": tk.DoubleVar(value=1.0),
            "y_c":   tk.DoubleVar(value=1.0), "k":  tk.DoubleVar(value=1.0),
            "y_luma":tk.DoubleVar(value=1.0),
            "u":  tk.DoubleVar(value=1.0), "v":  tk.DoubleVar(value=1.0),
            "cb": tk.DoubleVar(value=1.0), "cr": tk.DoubleVar(value=1.0),
            "gray_gain": tk.DoubleVar(value=1.0),
            "thresh":    tk.DoubleVar(value=128),
            "h":     tk.DoubleVar(value=1.0),
            "s":     tk.DoubleVar(value=1.0),
            "v_hsv": tk.DoubleVar(value=1.0),
            "alpha":    tk.DoubleVar(value=0.6),
            "morph_k":  tk.IntVar(value=3),
            "morph_it": tk.IntVar(value=1),
            "label_sel":tk.IntVar(value=1),
        }

        self.setup_ui()

    # -------------------------------------------------------------------------
    #  UI SETUP
    # -------------------------------------------------------------------------

    def setup_ui(self):
        top_bar = tk.Frame(self.root, bg="#1a1a1a", height=60)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        make_dropdown_button(top_bar, self.root, "📁 Fisier ▾", "#333", [
            ("📂  Incarca BMP",          self.load_image),
            ("💾  Exporta Fisier",        self.export_to_disk),
            None,
            ("✅  Salveaza modificarile",  self.confirm_save),
            ("❌  Reseteaza Imaginea",     self.cancel_edits),
        ])

        make_dropdown_button(top_bar, self.root, "✨ Filtre ▾", "#444", [
            ("⬛  Negative",                 lambda: self.preview_filter("negative")),
            ("🔲  Binarizare",               lambda: self.preview_filter("binarize")),
            ("🩶  Gray (medie)",             lambda: self.preview_filter("gray(1)")),
            ("🩶  Gray (luminanta)",          lambda: self.preview_filter("gray(2)")),
            ("🩶  Gray (desaturat)",          lambda: self.preview_filter("gray(3)")),
            ("🎨  CMYK",                     lambda: self.preview_filter("cmyk")),
            ("📡  YUV",                      lambda: self.preview_filter("yuv")),
            ("📺  YCbCr",                    lambda: self.preview_filter("ycbcr")),
            ("🌈  HSV",                      lambda: self.preview_filter("hsv")),
            ("🔄  RGB Back",                 lambda: self.preview_filter("rgb_back")),
            ("〰  Filtru Mediere 3x3",       lambda: self.preview_filter("mediere")),
            ("📊  Filtru Median 3x3",        lambda: self.preview_filter("median")),
            ("🔽  Filtru Minim 3x3",         lambda: self.preview_filter("minim")),
            ("🔼  Filtru Maxim 3x3",         lambda: self.preview_filter("maxim")),
            ("✨  Filtru Accentuare",        lambda: self.preview_filter("accentuare")),
        ])

        make_dropdown_button(top_bar, self.root, "🧪 Transformari ▾", "#7b2d8b", [
            ("📡  Transformata Fourier",     lambda: self._apply_lab3("fourier")),
            ("🎨  Floyd-Steinberg",          lambda: self._apply_lab3("floyd")),
        ])

        make_dropdown_button(top_bar, self.root, "📊 Histograma ▾", "#1565c0", [
            ("📊  Afiseaza Histograma",       self.show_histogram),
            ("📈  Egalizare Histograma",      lambda: self._apply_lab5("equalize")),
        ])

        make_dropdown_button(top_bar, self.root, "🔍 Analiza Obiect ▾", "#37474f", [
            ("📍  Centru de Greutate (M1)",     self.show_center_m1),
            ("📐  Momente Centrale (M2)",        self.show_moments_m2),
            ("🧮  Matrice Covarianta",           self.show_covariance),
            ("📉  Proiectii Orizontal/Vertical", self.show_projections),
        ])

        make_dropdown_button(top_bar, self.root, "🏷 Lab 4 — Etichetare ▾", "#5c3d9e", [
            ("🏷  Etichetare & Colorare (BFS)",   self._apply_labeling),
            ("🎯  Selectare Obiect dupa Eticheta", self._select_labeled_object),
            ("📐  Directie Alungire (Sobel)",      self._apply_sobel),
        ])

        make_dropdown_button(top_bar, self.root, "🔬 Lab 5 — Morfologie ▾", "#6c3a00", [
            ("⬛  Dilatare",              lambda: self._apply_lab5("dilate")),
            ("⬜  Eroziune",              lambda: self._apply_lab5("erode")),
            None,
            ("🔓  Deschidere",            lambda: self._apply_lab5("opening")),
            ("🔒  Inchidere",             lambda: self._apply_lab5("closing")),
        ])

        # Corp principal
        self.main_container = tk.Frame(self.root, bg="#121212")
        self.main_container.pack(expand=True, fill=tk.BOTH)

        # Sidebar
        self.sidebar = tk.Frame(self.main_container, bg="#1a1a1a", width=300)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        # Canvas principal
        self.canvas = tk.Canvas(self.main_container, bg="#121212", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=20, pady=20)

        # Bara de status
        self.status_bar = tk.Frame(self.root, bg="#1a1a1a", height=32)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Nicio imagine incarcata.")
        tk.Label(self.status_bar, textvariable=self.status_var,
                 fg="#a0a0b0", bg="#1a1a1a",
                 font=("Courier New", 9), anchor="w").pack(side=tk.LEFT, padx=12, pady=6)

        # Galerie (ascunsa initial)
        self.bottom_panel = tk.Frame(self.root, bg="#181818", height=200)

    def _build_sidebar(self):
        # Salvare / Reset
        ctrl = tk.LabelFrame(self.sidebar, text=" CONTROL EDITARE ",
                              fg="#4caf50", bg="#1a1a1a", padx=10, pady=8)
        ctrl.pack(fill=tk.X, padx=10, pady=(12, 4))
        tk.Button(ctrl, text="✅  Salveaza modificarile", command=self.confirm_save,
                  bg="#2e7d32", fg="white", font=("Segoe UI", 9, "bold"),
                  borderwidth=0, cursor="hand2", pady=5).pack(fill=tk.X, pady=2)
        tk.Button(ctrl, text="❌  Reseteaza Imaginea", command=self.cancel_edits,
                  bg="#c62828", fg="white", font=("Segoe UI", 9, "bold"),
                  borderwidth=0, cursor="hand2", pady=5).pack(fill=tk.X, pady=2)

        # Parametri globali
        tk.Label(self.sidebar, text="⚙  PARAMETRI GLOBALI",
                 fg="cyan", bg="#1a1a1a", font=("Arial", 9, "bold")).pack(pady=(8,2))
        uni = tk.Frame(self.sidebar, bg="#1a1a1a", padx=10)
        uni.pack(fill=tk.X)
        self._make_slider(uni, "Luminozitate", "bright")
        self._make_slider(uni, "Contrast",     "contrast")

        # Parametri dinamici per filtru
        tk.Label(self.sidebar, text="⚙  PARAMETRI FILTRU",
                 fg="#f9e2af", bg="#1a1a1a", font=("Arial", 9, "bold")).pack(pady=(8,2))
        self.dyn_frame = tk.Frame(self.sidebar, bg="#1a1a1a", padx=10)
        self.dyn_frame.pack(fill=tk.X)

        # Lab 3
        lab3 = tk.LabelFrame(self.sidebar, text=" LAB 3 — Filtre ",
                              fg="#c084fc", bg="#1a1a1a", padx=10, pady=6)
        lab3.pack(fill=tk.X, padx=10, pady=4)
        self._make_slider(lab3, "Alpha accentuare", "alpha", 0.1, 2.0, 0.1)

        # Lab 5
        lab5 = tk.LabelFrame(self.sidebar, text=" LAB 5 — Morfologie ",
                              fg="#fb923c", bg="#1a1a1a", padx=10, pady=6)
        lab5.pack(fill=tk.X, padx=10, pady=4)
        rf = tk.Frame(lab5, bg="#1a1a1a"); rf.pack(fill=tk.X)
        tk.Label(rf, text="Kernel:", fg="#cdd6f4", bg="#1a1a1a",
                 font=("Courier New", 9)).grid(row=0, column=0, sticky="w")
        tk.Spinbox(rf, from_=3, to=15, increment=2, textvariable=self.vars["morph_k"],
                   width=4, bg="#21262d", fg="#f0f6fc", buttonbackground="#30363d",
                   font=("Courier New", 10), relief=tk.FLAT).grid(row=0, column=1, padx=4)
        tk.Label(rf, text="Iteratii:", fg="#cdd6f4", bg="#1a1a1a",
                 font=("Courier New", 9)).grid(row=0, column=2, sticky="w", padx=(8,0))
        tk.Spinbox(rf, from_=1, to=20, textvariable=self.vars["morph_it"],
                   width=4, bg="#21262d", fg="#f0f6fc", buttonbackground="#30363d",
                   font=("Courier New", 10), relief=tk.FLAT).grid(row=0, column=3, padx=4)

        # Lab 4
        lab4 = tk.LabelFrame(self.sidebar, text=" LAB 4 — Etichetare ",
                              fg="#818cf8", bg="#1a1a1a", padx=10, pady=6)
        lab4.pack(fill=tk.X, padx=10, pady=4)
        lf = tk.Frame(lab4, bg="#1a1a1a"); lf.pack(fill=tk.X)
        tk.Label(lf, text="Eticheta:", fg="#cdd6f4", bg="#1a1a1a",
                 font=("Courier New", 9)).grid(row=0, column=0, sticky="w")
        self.label_spin = tk.Spinbox(lf, from_=1, to=999,
                                     textvariable=self.vars["label_sel"],
                                     width=5, bg="#21262d", fg="#f0f6fc",
                                     buttonbackground="#30363d",
                                     font=("Courier New", 10), relief=tk.FLAT)
        self.label_spin.grid(row=0, column=1, padx=4)
        tk.Label(lab4, text="(ruleaza Etichetare mai intai)",
                 fg="#6b7280", bg="#1a1a1a", font=("Courier New", 8)).pack(anchor="w")

    def _make_slider(self, parent, label, key, lo=0.0, hi=2.0, res=0.05):
        tk.Label(parent, text=label, fg="#888", bg="#1a1a1a",
                 font=("Arial", 8)).pack(anchor="w")
        tk.Scale(parent, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                 variable=self.vars[key], bg="#1a1a1a", fg="white",
                 highlightthickness=0,
                 command=lambda _: self._on_slider_change()
                 ).pack(fill=tk.X, pady=(0, 4))

    # -------------------------------------------------------------------------
    #  HELPERS
    # -------------------------------------------------------------------------

    def apply_math(self, img, mode):
        m = mode.lower()
        v = {k: var.get() for k, var in self.vars.items()}

        # Filtre care lucreaza pe imagine intreaga
        if m in filters_map_img:
            return filters_map_img[m](img, v)

        # Filtre pixel cu pixel
        w, h = img.size; pixels = img.load()
        res = Image.new("RGB", (w, h)); new_pix = res.load()
        b_f, c_f = v["bright"], v["contrast"]

        process_func = filters_map.get(m, lambda r, g, b, v: (
            int(r * v["r"]), int(g * v["g"]), int(b * v["b"])))

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

    def _require_image(self):
        if not self.original_img:
            messagebox.showwarning("Atentie", "Incarca mai intai o imagine BMP.")
            return False
        return True

    def _set_status(self, text):
        self.status_var.set(text)

    def render_main(self):
        if self.display_img:
            self._draw_on_canvas(self.display_img)

    def _draw_on_canvas(self, img):
        wc = self.canvas.winfo_width()
        hc = self.canvas.winfo_height()
        if wc < 10: wc, hc = 900, 650
        tmp = img.copy(); tmp.thumbnail((wc, hc), Image.LANCZOS)
        self.tk_main = ImageTk.PhotoImage(tmp)
        self.canvas.delete("all")
        self.canvas.create_image(wc//2, hc//2, image=self.tk_main)

    def _on_slider_change(self):
        if self.active_mode not in ("none", "") and self.original_img:
            self.display_img = self.apply_math(self.original_img.copy(), self.active_mode)
            self.render_main()

    # -------------------------------------------------------------------------
    #  FISIER
    # -------------------------------------------------------------------------

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Imagini", "*.bmp *.png *.jpg *.jpeg"), ("Toate", "*.*")])
        if path:
            self.original_img = Image.open(path).convert("RGB")
            self.display_img  = self.original_img.copy()
            self._labels = None
            self._set_status(
                f"Imagine incarcata: {path.split('/')[-1]}  |  "
                f"{self.original_img.size[0]}x{self.original_img.size[1]} px")
            self.render_main()

    def confirm_save(self):
        if self.display_img:
            self.original_img = self.display_img.copy()
            self._set_status("✅ Modificarile au fost salvate ca imagine de baza.")

    def cancel_edits(self):
        for k, v in self.vars.items():
            if   k == "thresh":   v.set(128)
            elif k == "morph_k":  v.set(3)
            elif k == "morph_it": v.set(1)
            elif k == "alpha":    v.set(0.6)
            elif k == "label_sel":v.set(1)
            else:
                try: v.set(1.0)
                except: pass
        if self.original_img:
            self.display_img = self.original_img.copy()
            self.active_mode = "none"
            self.update_sidebar_dynamic("none")
            self.render_main()
            self._set_status("Resetat la imaginea salvata.")

    def export_to_disk(self):
        if not self.display_img: return
        path = filedialog.asksaveasfilename(
            defaultextension=".bmp",
            filetypes=[("Bitmap","*.bmp"),("PNG","*.png")])
        if path:
            self.display_img.save(path)
            self._set_status(f"Exportat: {path.split('/')[-1]}")

    # -------------------------------------------------------------------------
    #  LAB 3 — aplica direct pe canvas
    # -------------------------------------------------------------------------

    def _apply_lab3(self, operation):
        if not self._require_image(): return
        names = {
            "fourier":    "Transformata Fourier",
            "mediere":    "Filtru Mediere 3x3",
            "median":     "Filtru Median 3x3",
            "minim":      "Filtru Minim 3x3",
            "maxim":      "Filtru Maxim 3x3",
            "accentuare": "Filtru Accentuare",
            "floyd":      "Floyd-Steinberg Dithering",
        }
        self._set_status(f"⏳ Procesare: {names.get(operation)} ...")
        self.root.update_idletasks()

        src   = self.original_img
        alpha = self.vars["alpha"].get()

        if   operation == "fourier":    res = transformata_fourier(src)
        elif operation == "floyd":      res = floyd_steinberg(src, self.FS_PALETTE)
        else: return

        self.display_img = res
        self.active_mode = operation
        self.update_sidebar_dynamic("none")
        self.render_main()
        self._set_status(
            f"✔  {names.get(operation)} aplicat.  "
            f"Apasa 'Salveaza modificarile' pentru a confirma sau 'Reseteaza' pentru a anula.")

    # -------------------------------------------------------------------------
    #  LAB 4 — Etichetare (aplica direct pe canvas)
    # -------------------------------------------------------------------------

    def _apply_labeling(self):
        if not self._require_image(): return
        self._set_status("⏳ Etichetare componente conexe (BFS) ...")
        self.root.update_idletasks()
        self._labels, self._num_labels = label_connected_components(self.original_img)
        self._label_colors = generate_label_colors(self._num_labels)
        w, h = self.original_img.size
        result = render_labeled_image(self._labels, self._label_colors, w, h)
        self.display_img = result
        self.label_spin.config(to=max(self._num_labels, 1))
        self.render_main()
        self._set_status(
            f"✔  Etichetare completa — {self._num_labels} obiecte.  "
            f"Selecteaza eticheta din sidebar si apasa 'Selectare Obiect'.")

    def _select_labeled_object(self):
        if not self._require_image(): return
        if self._labels is None:
            messagebox.showwarning("Atentie", "Ruleaza mai intai 'Etichetare & Colorare'.")
            return
        lbl = self.vars["label_sel"].get()
        if lbl < 1 or lbl > self._num_labels:
            messagebox.showwarning("Atentie",
                f"Eticheta trebuie sa fie intre 1 si {self._num_labels}.")
            return
        w, h = self.original_img.size
        mask_img, coords = extract_object_mask(self._labels, lbl, w, h)
        draw = ImageDraw.Draw(mask_img)
        draw.rectangle([0,0,w-1,h-1],
                       outline=self._label_colors.get(lbl,(255,0,0)), width=3)
        self.display_img = mask_img
        self.render_main()
        self._set_status(
            f"✔  Obiect {lbl} selectat — {len(coords)} pixeli.  "
            f"Apasa Sobel pentru directie sau Salveaza/Reseteaza.")

    def _apply_sobel(self):
        if not self._require_image(): return
        if self._labels is None:
            messagebox.showwarning("Atentie", "Ruleaza mai intai 'Etichetare & Colorare'.")
            return
        lbl = self.vars["label_sel"].get()
        if lbl < 1 or lbl > self._num_labels:
            messagebox.showwarning("Atentie",
                f"Eticheta trebuie sa fie intre 1 si {self._num_labels}.")
            return
        self._set_status("⏳ Calcul Sobel ...")
        self.root.update_idletasks()
        w, h = self.original_img.size
        mask_img, _ = extract_object_mask(self._labels, lbl, w, h)
        deg, mag, px, py = compute_sobel_orientation(mask_img)
        annotated = mask_img.copy()
        draw = ImageDraw.Draw(annotated)
        draw.ellipse([px-6, py-6, px+6, py+6], outline="red", width=2)
        lx = int(px + 40 * math.cos(math.radians(deg)))
        ly = int(py + 40 * math.sin(math.radians(deg)))
        draw.line([px, py, lx, ly], fill="red", width=2)
        self.display_img = annotated
        self.render_main()
        self._set_status(
            f"✔  Sobel — Directie: {deg:.1f}°  |  Magnitudine: {mag:.1f}  |  Peak: ({px},{py})")

    # -------------------------------------------------------------------------
    #  LAB 5 — Morfologie (aplica direct pe canvas)
    # -------------------------------------------------------------------------

    def _apply_lab5(self, operation):
        if not self._require_image(): return
        ks = self.vars["morph_k"].get()
        it = self.vars["morph_it"].get()
        names = {
            "equalize": "Egalizare Histograma",
            "dilate":   f"Dilatare (k={ks}, iter={it})",
            "erode":    f"Eroziune (k={ks}, iter={it})",
            "opening":  f"Deschidere (k={ks}, iter={it})",
            "closing":  f"Inchidere (k={ks}, iter={it})",
        }
        self._set_status(f"⏳ {names.get(operation)} ...")
        self.root.update_idletasks()
        src = self.original_img
        if   operation == "equalize": res = equalize_histogram(src)
        elif operation == "dilate":   res = dilate(src, ks, it)
        elif operation == "erode":    res = erode(src, ks, it)
        elif operation == "opening":  res = opening(src, ks, it)
        elif operation == "closing":  res = closing(src, ks, it)
        else: return
        self.display_img = res
        self.active_mode = operation
        self.render_main()
        self._set_status(
            f"✔  {names.get(operation)} aplicat.  "
            f"Apasa 'Salveaza modificarile' pentru a confirma sau 'Reseteaza' pentru a anula.")

    # -------------------------------------------------------------------------
    #  FILTRE CLASICE
    # -------------------------------------------------------------------------

    def update_sidebar_dynamic(self, mode):
        for w in self.dyn_frame.winfo_children(): w.destroy()
        config = {
            "none":     [("Rosu (R)","r"),("Verde (G)","g"),("Albastru (B)","b")],
            "negative": [("Canal R","r"),("Canal G","g"),("Canal B","b")],
            "gray(1)":  [("Gray Gain","gray_gain")],
            "gray(2)":  [("Gray Gain","gray_gain")],
            "gray(3)":  [("Gray Gain","gray_gain")],
            "binarize": [("Prag","thresh", 0, 255, 1)],
            "cmyk":     [("Cyan","c"),("Magenta","m"),("Yellow","y_c"),("Black (K)","k")],
            "yuv":      [("Luma (Y)","y_luma"),("Chroma (U)","u"),("Chroma (V)","v")],
            "ycbcr":    [("Luma (Y)","y_luma"),("Chroma (Cb)","cb"),("Chroma (Cr)","cr")],
            "rgb_back": [("Canal R","r"),("Canal G","g"),("Canal B","b")],
            "hsv":      [("Nuance (H)","h"),("Saturation (S)","s"),("Value (V)","v_hsv")],
        }
        for item in config.get(mode.lower(), []):
            if len(item) == 5:
                self._make_slider(self.dyn_frame, item[0], item[1], item[2], item[3], item[4])
            else:
                self._make_slider(self.dyn_frame, item[0], item[1])

    def preview_filter(self, mode):
        if not self._require_image(): return
        self.active_mode = mode
        self.update_sidebar_dynamic(mode)
        self.display_img = self.apply_math(self.original_img.copy(), mode)
        self.render_main()
        self._set_status(
            f"Preview: {mode}  —  Apasa 'Salveaza modificarile' sau 'Reseteaza'.")

    # -------------------------------------------------------------------------
    #  HISTOGRAMA & ANALIZA
    # -------------------------------------------------------------------------

    def show_histogram(self):
        if not self.display_img: return
        pix = self.display_img.convert("L").getdata()
        plt.figure("Histograma", figsize=(5,3))
        plt.hist(pix, bins=256, color='gray')
        plt.title("Histograma grayscale")
        plt.tight_layout(); plt.show()

    def verify_object(self):
        if not self.display_img: return None
        img = self.display_img.convert("RGB")
        w, h = img.size; pixels = img.load()
        m00 = m10 = m01 = 0; coords = []
        for y in range(h):
            for x in range(w):
                r, g, b = pixels[x, y]
                if r < 240 or g < 240 or b < 240:
                    m00 += 1; m10 += x; m01 += y; coords.append((x, y))
        if m00 == 0:
            messagebox.showwarning("Analiza", "Nu a fost detectat niciun obiect.")
            return None
        return {"m00": m00, "xc": m10/m00, "yc": m01/m00,
                "coords": coords, "w": w, "h": h}

    def show_center_m1(self):
        data = self.verify_object()
        if not data: return
        copy = self.display_img.copy().convert("RGB")
        draw = ImageDraw.Draw(copy)
        xc, yc = data["xc"], data["yc"]
        draw.ellipse([xc-5, yc-5, xc+5, yc+5], fill="red", outline="white")
        self._draw_on_canvas(copy)
        messagebox.showinfo("Moment M1",
            f"Centrul de greutate:\nXc: {xc:.2f}\nYc: {yc:.2f}")

    def show_moments_m2(self):
        data = self.verify_object()
        if not data: return
        xc, yc = data["xc"], data["yc"]
        mu20 = sum((p[0]-xc)**2 for p in data["coords"])
        mu02 = sum((p[1]-yc)**2 for p in data["coords"])
        mu11 = sum((p[0]-xc)*(p[1]-yc) for p in data["coords"])
        messagebox.showinfo("Momente Centrale M2",
            f"mu20: {mu20:.0f}\nmu02: {mu02:.0f}\nmu11: {mu11:.0f}")

    def show_covariance(self):
        data = self.verify_object()
        if not data: return
        m00, xc, yc = data["m00"], data["xc"], data["yc"]
        m20 = sum((p[0]-xc)**2 for p in data["coords"]) / m00
        m02 = sum((p[1]-yc)**2 for p in data["coords"]) / m00
        m11 = sum((p[0]-xc)*(p[1]-yc) for p in data["coords"]) / m00
        messagebox.showinfo("Matrice Covarianta",
            f"| {m20:.2f}   {m11:.2f} |\n| {m11:.2f}   {m02:.2f} |")

    def show_projections(self):
        data = self.verify_object()
        if not data: return
        ph = [0]*data["h"]; pv = [0]*data["w"]
        for x, y in data["coords"]:
            pv[x] += 1; ph[y] += 1
        plt.figure("Proiectii", figsize=(10, 4))
        plt.subplot(121); plt.plot(pv, color='blue'); plt.title("Proiectie Verticala")
        plt.subplot(122); plt.plot(ph, color='red');  plt.title("Proiectie Orizontala")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedBitmapEditor(root)
    root.mainloop()