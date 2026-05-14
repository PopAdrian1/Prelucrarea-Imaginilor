import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt
import math
from collections import deque

# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAR: Factory pentru butoane dropdown
# ══════════════════════════════════════════════════════════════════════════════

def make_dropdown_button(parent, root, label, color, items):
    """
    Creează un buton în bara de sus care deschide un meniu dropdown.

    Parametri:
        parent : frame-ul barei de sus.
        root   : fereastra Tk principală (pentru poziționare dropdown).
        label  : textul butonului.
        color  : culoarea de fundal a butonului.
        items  : listă de (text, callback) sau None pentru separator.

    Returnează butonul creat.
    """
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
    """
    Etichetează componentele conexe (BFS, N8-vecinătate).

    Parcurge imaginea pixel cu pixel. La fiecare pixel de obiect (< 128)
    neetichetat, atribuie o etichetă nouă și o propagă tuturor vecinilor N8
    prin BFS (coadă FIFO).

    Returnează:
        labels (list[list[int]]): matrice H×W; 0=fundal, ≥1=obiect.
        num_labels (int): numărul de obiecte.
    """
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
    """
    Paletă HSV distinctă pentru etichete (unghi de aur 137.5°).
    Returnează dict {etichetă -> (R,G,B)}.
    """
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
    """Construiește imaginea RGB colorată din matricea de etichete."""
    result = Image.new("RGB", (w, h), (255, 255, 255))
    pix = result.load()
    for i in range(h):
        for j in range(w):
            pix[j, i] = colors.get(labels[i][j], (255, 255, 255))
    return result


def extract_object_mask(labels, target_label, w, h):
    """
    Extrage masca unui obiect: pixelii cu eticheta țintă → negru, rest → alb.

    Returnează:
        mask_img (PIL.Image): imagine cu obiectul izolat.
        coords (list): coordonatele (x,y) ale pixelilor obiectului.
    """
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
    """
    Direcția de alungire via operatorul Sobel.

    Nuclee 3×3:
        Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
        Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]

    Returnează (unghi_grade, magnitudine_max, peak_x, peak_y).
    """
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


class LabelingWindow:
    """Fereastră: etichetare BFS + selectare obiect + Sobel."""

    def __init__(self, parent, source_image):
        self.win = tk.Toplevel(parent)
        self.win.title("🏷️ Etichetare Componente Conexe")
        self.win.geometry("1100x700")
        self.win.configure(bg="#0d1117")
        self.source_image = source_image.convert("RGB")
        self.w, self.h = self.source_image.size
        self.labels, self.num_labels = label_connected_components(self.source_image)
        self.colors = generate_label_colors(self.num_labels)
        self.labeled_img = render_labeled_image(self.labels, self.colors, self.w, self.h)
        self.selected_label = None
        self.tk_img_left = self.tk_img_right = None
        self._current_mask = None
        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self.win, bg="#161b22", height=50)
        header.pack(fill=tk.X)
        tk.Label(header, text=f"  Obiecte detectate: {self.num_labels}",
                 fg="#58a6ff", bg="#161b22",
                 font=("Courier New", 12, "bold")).pack(side=tk.LEFT, padx=15, pady=12)
        body = tk.Frame(self.win, bg="#0d1117")
        body.pack(expand=True, fill=tk.BOTH)
        left = tk.Frame(body, bg="#0d1117")
        left.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        tk.Label(left, text="Imagine etichetată", fg="#8b949e", bg="#0d1117",
                 font=("Courier New", 9)).pack()
        self.canvas_left = tk.Canvas(left, bg="#0d1117", highlightthickness=1,
                                     highlightbackground="#30363d")
        self.canvas_left.pack(expand=True, fill=tk.BOTH)
        right = tk.Frame(body, bg="#0d1117")
        right.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        tk.Label(right, text="Obiect selectat", fg="#8b949e", bg="#0d1117",
                 font=("Courier New", 9)).pack()
        self.canvas_right = tk.Canvas(right, bg="#0d1117", highlightthickness=1,
                                      highlightbackground="#30363d")
        self.canvas_right.pack(expand=True, fill=tk.BOTH)
        footer = tk.Frame(self.win, bg="#161b22", height=90)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(footer, text="Etichetă:", fg="#c9d1d9", bg="#161b22",
                 font=("Courier New", 10)).pack(side=tk.LEFT, padx=15, pady=20)
        self.label_var = tk.IntVar(value=1)
        tk.Spinbox(footer, from_=1, to=max(self.num_labels, 1),
                   textvariable=self.label_var, width=6,
                   bg="#21262d", fg="#f0f6fc", buttonbackground="#30363d",
                   font=("Courier New", 11), relief=tk.FLAT).pack(side=tk.LEFT, padx=5)
        tk.Button(footer, text="🎯  Selectează Obiect", command=self._select_object,
                  bg="#238636", fg="white", font=("Courier New", 10, "bold"),
                  padx=12, pady=6, relief=tk.FLAT, cursor="hand2").pack(side=tk.LEFT, padx=10)
        tk.Button(footer, text="📐  Calcul Sobel", command=self._compute_sobel,
                  bg="#1f6feb", fg="white", font=("Courier New", 10, "bold"),
                  padx=12, pady=6, relief=tk.FLAT, cursor="hand2").pack(side=tk.LEFT, padx=10)
        self.result_var = tk.StringVar(value="—")
        tk.Label(footer, textvariable=self.result_var, fg="#f0e68c", bg="#161b22",
                 font=("Courier New", 10, "bold"), wraplength=340).pack(side=tk.LEFT, padx=20)
        self.win.after(100, self._render_labeled)

    def _render_labeled(self):
        cw = self.canvas_left.winfo_width() or 480
        ch = self.canvas_left.winfo_height() or 500
        img = self.labeled_img.copy(); img.thumbnail((cw, ch))
        self.tk_img_left = ImageTk.PhotoImage(img)
        self.canvas_left.delete("all")
        self.canvas_left.create_image(cw//2, ch//2, image=self.tk_img_left)

    def _select_object(self):
        lbl = self.label_var.get()
        if lbl < 1 or lbl > self.num_labels:
            messagebox.showwarning("Atenție", f"Eticheta între 1 și {self.num_labels}.", parent=self.win)
            return
        self.selected_label = lbl
        mask_img, coords = extract_object_mask(self.labels, lbl, self.w, self.h)
        self._current_mask = mask_img
        draw = ImageDraw.Draw(mask_img)
        draw.rectangle([0,0,self.w-1,self.h-1], outline=self.colors.get(lbl,(255,0,0)), width=3)
        cw = self.canvas_right.winfo_width() or 480
        ch = self.canvas_right.winfo_height() or 500
        display = mask_img.copy(); display.thumbnail((cw, ch))
        self.tk_img_right = ImageTk.PhotoImage(display)
        self.canvas_right.delete("all")
        self.canvas_right.create_image(cw//2, ch//2, image=self.tk_img_right)
        self.result_var.set(f"Obiect {lbl} — {len(coords)} pixeli")

    def _compute_sobel(self):
        if self.selected_label is None:
            messagebox.showwarning("Atenție", "Selectează mai întâi un obiect.", parent=self.win)
            return
        deg, mag, px, py = compute_sobel_orientation(self._current_mask)
        annotated = self._current_mask.copy()
        draw = ImageDraw.Draw(annotated)
        draw.ellipse([px-6, py-6, px+6, py+6], outline="red", width=2)
        lx = int(px + 30*math.cos(math.radians(deg)))
        ly = int(py + 30*math.sin(math.radians(deg)))
        draw.line([px, py, lx, ly], fill="red", width=2)
        cw = self.canvas_right.winfo_width() or 480
        ch = self.canvas_right.winfo_height() or 500
        display = annotated.copy(); display.thumbnail((cw, ch))
        self.tk_img_right = ImageTk.PhotoImage(display)
        self.canvas_right.delete("all")
        self.canvas_right.create_image(cw//2, ch//2, image=self.tk_img_right)
        self.result_var.set(f"Direcție: {deg:.1f}°  |  Mag: {mag:.1f}  |  Peak: ({px},{py})")


# ══════════════════════════════════════════════════════════════════════════════
#  LAB 5 — Egalizare histogramă + Morfologie (fără OpenCV)
# ══════════════════════════════════════════════════════════════════════════════

def equalize_histogram(img):
    """
    Egalizarea histogramei (fără OpenCV).

    Pași:
      1. Calculează histograma h[i] a imaginii grayscale.
      2. Calculează histograma cumulativă hc[i] = Σ h[0..i].
      3. Aplică transformarea T(x) = (hc[x] - hc[0]) * 255 / (N - hc[0])
         unde N = numărul total de pixeli.

    Returnează PIL.Image în mod RGB.
    """
    gray = img.convert("L")
    w, h = gray.size
    pix = gray.load()
    N = w * h
    # Pasul 1: histograma
    hist = [0] * 256
    for y in range(h):
        for x in range(w):
            hist[pix[x, y]] += 1
    # Pasul 2: cumulativa
    hc = [0] * 256
    hc[0] = hist[0]
    for i in range(1, 256):
        hc[i] = hc[i-1] + hist[i]
    # Pasul 3: tabela de transformare
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
    """Conversia imagine → matrice binară (1=obiect întunecat, 0=fundal)."""
    gray = img.convert("L")
    w, h = gray.size
    pix = gray.load()
    return [[1 if pix[x,y] < threshold else 0 for x in range(w)] for y in range(h)], w, h


def _from_binary(matrix, w, h):
    """Conversia matrice binară → imagine PIL (1→negru, 0→alb)."""
    result = Image.new("L", (w, h), 255)
    pix = result.load()
    for y in range(h):
        for x in range(w):
            if matrix[y][x] == 1:
                pix[x, y] = 0
    return result.convert("RGB")


def dilate(img, kernel_size=3, iterations=1):
    """
    Dilatare morfologică (fără OpenCV).

    Un pixel de fundal devine obiect dacă cel puțin un vecin din kernelul
    pătrat de dimensiune kernel_size este obiect. Repetată de `iterations` ori.

    Parametri:
        img (PIL.Image): imagine binară.
        kernel_size (int): dimensiunea kernelului (impar).
        iterations (int): numărul de repetări.

    Returnează PIL.Image dilatată.
    """
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
    """
    Eroziune morfologică (fără OpenCV).

    Un pixel de obiect rămâne obiect doar dacă TOȚI vecinii din kernel
    sunt tot obiect. Repetată de `iterations` ori.

    Parametri:
        img (PIL.Image): imagine binară.
        kernel_size (int): dimensiunea kernelului (impar).
        iterations (int): numărul de repetări.

    Returnează PIL.Image erodată.
    """
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
    """
    Deschidere morfologică = Eroziune → Dilatare (repetată de iterations ori).

    Elimină zgomotul mic și detașează obiecte conectate subtil.
    """
    result = img
    for _ in range(iterations):
        result = erode(result, kernel_size, 1)
        result = dilate(result, kernel_size, 1)
    return result


def closing(img, kernel_size=3, iterations=1):
    """
    Închidere morfologică = Dilatare → Eroziune (repetată de iterations ori).

    Umple goluri mici și conectează contururi apropiate.
    """
    result = img
    for _ in range(iterations):
        result = dilate(result, kernel_size, 1)
        result = erode(result, kernel_size, 1)
    return result


class MorphologyWindow:
    """
    Fereastră Lab 5: Egalizare histogramă + Dilatare + Eroziune +
    Deschidere + Închidere. Afișează original și rezultat alăturat.
    """

    def __init__(self, parent, source_image, operation="equalize"):
        self.win = tk.Toplevel(parent)
        self.win.title("🔬 Lab 5 — Morfologie & Histogramă")
        self.win.geometry("1150x720")
        self.win.configure(bg="#0d1117")
        self.source_image = source_image.convert("RGB")
        self.result_image = None
        self.tk_left = self.tk_right = None
        self._editor_callback = None
        self._build_ui()
        self.win.after(120, lambda: self._run(operation))

    def _build_ui(self):
        header = tk.Frame(self.win, bg="#161b22")
        header.pack(fill=tk.X)
        tk.Label(header, text="  🔬 Lab 5 — Morfologie & Egalizare Histogramă",
                 fg="#cba6f7", bg="#161b22",
                 font=("Courier New", 12, "bold")).pack(side=tk.LEFT, padx=15, pady=10)

        body = tk.Frame(self.win, bg="#0d1117")
        body.pack(expand=True, fill=tk.BOTH)

        left = tk.Frame(body, bg="#0d1117")
        left.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        tk.Label(left, text="Original", fg="#8b949e", bg="#0d1117",
                 font=("Courier New", 9)).pack()
        self.canvas_orig = tk.Canvas(left, bg="#0d1117", highlightthickness=1,
                                     highlightbackground="#30363d")
        self.canvas_orig.pack(expand=True, fill=tk.BOTH)

        right = tk.Frame(body, bg="#0d1117")
        right.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.result_label_var = tk.StringVar(value="Rezultat")
        tk.Label(right, textvariable=self.result_label_var, fg="#8b949e", bg="#0d1117",
                 font=("Courier New", 9)).pack()
        self.canvas_result = tk.Canvas(right, bg="#0d1117", highlightthickness=1,
                                       highlightbackground="#30363d")
        self.canvas_result.pack(expand=True, fill=tk.BOTH)

        footer = tk.Frame(self.win, bg="#161b22")
        footer.pack(fill=tk.X, side=tk.BOTTOM, pady=8)

        # Parametri
        pf = tk.Frame(footer, bg="#161b22")
        pf.pack(side=tk.LEFT, padx=12)
        tk.Label(pf, text="Kernel:", fg="#cdd6f4", bg="#161b22",
                 font=("Courier New", 9)).grid(row=0, column=0, sticky="w")
        self.kernel_var = tk.IntVar(value=3)
        tk.Spinbox(pf, from_=3, to=15, increment=2, textvariable=self.kernel_var,
                   width=4, bg="#21262d", fg="#f0f6fc", buttonbackground="#30363d",
                   font=("Courier New", 10), relief=tk.FLAT).grid(row=0, column=1, padx=5)
        tk.Label(pf, text="Iterații:", fg="#cdd6f4", bg="#161b22",
                 font=("Courier New", 9)).grid(row=0, column=2, sticky="w", padx=(10,0))
        self.iter_var = tk.IntVar(value=1)
        tk.Spinbox(pf, from_=1, to=20, textvariable=self.iter_var,
                   width=4, bg="#21262d", fg="#f0f6fc", buttonbackground="#30363d",
                   font=("Courier New", 10), relief=tk.FLAT).grid(row=0, column=3, padx=5)

        # Butoane operații
        ops = [
            ("📊 Egalizare", "equalize", "#6c71c4"),
            ("⬛ Dilatare",  "dilate",   "#2aa198"),
            ("⬜ Eroziune",  "erode",    "#268bd2"),
            ("🔓 Deschidere","opening",  "#859900"),
            ("🔒 Închidere", "closing",  "#cb4b16"),
        ]
        bf = tk.Frame(footer, bg="#161b22")
        bf.pack(side=tk.LEFT, padx=15)
        for text, op, color in ops:
            tk.Button(bf, text=text, bg=color, fg="white",
                      font=("Courier New", 9, "bold"), padx=8, pady=5,
                      relief=tk.FLAT, cursor="hand2",
                      command=lambda o=op: self._run(o)).pack(side=tk.LEFT, padx=3)

        tk.Button(footer, text="💾 Trimite în Editor",
                  command=self._send_to_editor,
                  bg="#2e7d32", fg="white", font=("Courier New", 9, "bold"),
                  padx=10, pady=5, relief=tk.FLAT, cursor="hand2").pack(side=tk.RIGHT, padx=20)

        self.win.after(100, self._render_orig)

    def _render_orig(self):
        cw = self.canvas_orig.winfo_width() or 480
        ch = self.canvas_orig.winfo_height() or 500
        img = self.source_image.copy(); img.thumbnail((cw, ch))
        self.tk_left = ImageTk.PhotoImage(img)
        self.canvas_orig.delete("all")
        self.canvas_orig.create_image(cw//2, ch//2, image=self.tk_left)

    def _run(self, operation):
        ks = self.kernel_var.get()
        it = self.iter_var.get()
        names = {
            "equalize": "Egalizare Histogramă",
            "dilate":   f"Dilatare (k={ks}, iter={it})",
            "erode":    f"Eroziune (k={ks}, iter={it})",
            "opening":  f"Deschidere (k={ks}, iter={it})",
            "closing":  f"Închidere (k={ks}, iter={it})",
        }
        self.result_label_var.set(names.get(operation, "Rezultat"))
        src = self.source_image
        if   operation == "equalize": res = equalize_histogram(src)
        elif operation == "dilate":   res = dilate(src, ks, it)
        elif operation == "erode":    res = erode(src, ks, it)
        elif operation == "opening":  res = opening(src, ks, it)
        elif operation == "closing":  res = closing(src, ks, it)
        else: return
        self.result_image = res
        cw = self.canvas_result.winfo_width() or 480
        ch = self.canvas_result.winfo_height() or 500
        display = res.copy(); display.thumbnail((cw, ch))
        self.tk_right = ImageTk.PhotoImage(display)
        self.canvas_result.delete("all")
        self.canvas_result.create_image(cw//2, ch//2, image=self.tk_right)

    def _send_to_editor(self):
        if self.result_image and self._editor_callback:
            self._editor_callback(self.result_image)
            self.win.destroy()
        else:
            messagebox.showinfo("Info", "Rulează mai întâi o operație.", parent=self.win)

    def set_editor_callback(self, cb):
        self._editor_callback = cb


# ══════════════════════════════════════════════════════════════════════════════
#  CLASA PRINCIPALĂ
# ══════════════════════════════════════════════════════════════════════════════

class AdvancedBitmapEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Bitmap Editor")
        self.root.geometry("1550x950")
        self.root.configure(bg="#121212")
        self.original_img = None
        self.display_img = None
        self.active_mode = "none"
        self.previews = {}
        self.vars = {
            "bright": tk.DoubleVar(value=1.0), "contrast": tk.DoubleVar(value=1.0),
            "r": tk.DoubleVar(value=1.0), "g": tk.DoubleVar(value=1.0), "b": tk.DoubleVar(value=1.0),
            "c": tk.DoubleVar(value=1.0), "m": tk.DoubleVar(value=1.0),
            "y_c": tk.DoubleVar(value=1.0), "k": tk.DoubleVar(value=1.0),
            "y_luma": tk.DoubleVar(value=1.0), "u": tk.DoubleVar(value=1.0), "v": tk.DoubleVar(value=1.0),
            "cb": tk.DoubleVar(value=1.0), "cr": tk.DoubleVar(value=1.0),
            "gray_gain": tk.DoubleVar(value=1.0), "thresh": tk.DoubleVar(value=128),
            "h": tk.DoubleVar(value=1.0), "s": tk.DoubleVar(value=1.0), "v_hsv": tk.DoubleVar(value=1.0)
        }
        self.setup_ui()

    def setup_ui(self):
        top_bar = tk.Frame(self.root, bg="#1a1a1a", height=60)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        # ── Meniu 1: Fișier ───────────────────────────────────────────────────
        make_dropdown_button(top_bar, self.root, "📁 Fișier ▾", "#333", [
            ("📂  Încarcă BMP",           self.load_image),
            ("💾  Exportă Fișier",         self.export_to_disk),
            None,
            ("✅  Salvează modificările",   self.confirm_save),
            ("❌  Resetează Imaginea",      self.cancel_edits),
        ])

        # ── Meniu 2: Filtre ───────────────────────────────────────────────────
        make_dropdown_button(top_bar, self.root, "✨ Filtre ▾", "#444", [
            ("🖼️  Galerie Filtre",           self.toggle_gallery),
            None,
            ("⬛  Negative",                  lambda: self.preview_filter("negative")),
            ("🔲  Binarizare",                lambda: self.preview_filter("binarize")),
            ("🩶  Gray (medie)",              lambda: self.preview_filter("gray(1)")),
            ("🩶  Gray (luminanță)",           lambda: self.preview_filter("gray(2)")),
            ("🩶  Gray (desaturat)",           lambda: self.preview_filter("gray(3)")),
            ("🎨  CMYK",                      lambda: self.preview_filter("cmyk")),
            ("📡  YUV",                       lambda: self.preview_filter("yuv")),
            ("📺  YCbCr",                     lambda: self.preview_filter("ycbcr")),
            ("🌈  HSV",                       lambda: self.preview_filter("hsv")),
            ("🔄  RGB Back",                  lambda: self.preview_filter("rgb_back")),
        ])

        # ── Meniu 3: Histogramă ───────────────────────────────────────────────
        make_dropdown_button(top_bar, self.root, "📊 Histogramă ▾", "#1565c0", [
            ("📊  Afișează Histogramă",        self.show_histogram),
            ("📈  Egalizare (preview rapid)",   self._quick_equalize),
        ])

        # ── Meniu 4: Analiză Obiect ───────────────────────────────────────────
        make_dropdown_button(top_bar, self.root, "🔍 Analiză Obiect ▾", "#37474f", [
            ("📍  Centru de Greutate (M1)",      self.show_center_m1),
            ("📐  Momente Centrale (M2)",         self.show_moments_m2),
            ("🧮  Matrice Covarianță",            self.show_covariance),
            ("📉  Proiecții Orizontal/Vertical",  self.show_projections),
        ])

        # ── Meniu 5: Lab 4 — Etichetare ──────────────────────────────────────
        make_dropdown_button(top_bar, self.root, "🏷️ Lab 4 — Etichetare ▾", "#5c3d9e", [
            ("🏷️  Etichetare & Colorare (BFS)",    self._open_labeling_full),
            ("🎯  Selectare Obiect după Etichetă",  self._open_labeling_select),
            ("📐  Direcție Alungire (Sobel)",       self._open_labeling_sobel),
        ])

        # ── Meniu 6: Lab 5 — Morfologie ──────────────────────────────────────
        make_dropdown_button(top_bar, self.root, "🔬 Lab 5 — Morfologie ▾", "#6c3a00", [
            ("📈  Egalizare Histogramă",   lambda: self._open_morphology("equalize")),
            None,
            ("⬛  Dilatare",               lambda: self._open_morphology("dilate")),
            ("⬜  Eroziune",               lambda: self._open_morphology("erode")),
            None,
            ("🔓  Deschidere",             lambda: self._open_morphology("opening")),
            ("🔒  Închidere",              lambda: self._open_morphology("closing")),
        ])

        # ── Corp principal ────────────────────────────────────────────────────
        self.main_container = tk.Frame(self.root, bg="#121212")
        self.main_container.pack(expand=True, fill=tk.BOTH)

        self.sidebar = tk.Frame(self.main_container, bg="#1a1a1a", width=320)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        edit_frame = tk.LabelFrame(self.sidebar, text=" CONTROL EDITARE ",
                                   fg="#4caf50", bg="#1a1a1a", padx=10, pady=10)
        edit_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Button(edit_frame, text="✅ Salvează modificările", command=self.confirm_save,
                  bg="#2e7d32", fg="white", borderwidth=0).pack(fill=tk.X, pady=2)
        tk.Button(edit_frame, text="❌ Resetează Imaginea", command=self.cancel_edits,
                  bg="#c62828", fg="white", borderwidth=0).pack(fill=tk.X, pady=2)

        tk.Label(self.sidebar, text="⚙️ PARAMETRI FILTRU",
                 fg="cyan", bg="#1a1a1a", font=("Arial", 9, "bold")).pack(pady=5)
        uni_frame = tk.Frame(self.sidebar, bg="#1a1a1a", padx=10)
        uni_frame.pack(fill=tk.X)
        self.create_slider(uni_frame, "Luminozitate Globală", "bright")
        self.create_slider(uni_frame, "Contrast Global", "contrast")
        self.dynamic_slider_container = tk.Frame(self.sidebar, bg="#1a1a1a", padx=10)
        self.dynamic_slider_container.pack(fill=tk.X, expand=True)

        self.canvas = tk.Canvas(self.main_container, bg="#121212", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=20, pady=20)

        self.bottom_panel = tk.Frame(self.root, bg="#181818", height=200)
        self.update_sidebar_dynamic("none")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _require_image(self):
        if not self.display_img:
            messagebox.showwarning("Atenție", "Încarcă mai întâi o imagine BMP.")
            return False
        return True

    # ── Lab 4 handlers ────────────────────────────────────────────────────────

    def _open_labeling_full(self):
        if not self._require_image(): return
        w = LabelingWindow(self.root, self.display_img)
        w.win.after(200, w._render_labeled)

    def _open_labeling_select(self):
        if not self._require_image(): return
        w = LabelingWindow(self.root, self.display_img)
        w.win.after(200, w._render_labeled)
        w.win.after(600, lambda: messagebox.showinfo(
            "Selectare", "Alege eticheta din spinbox și apasă 🎯.", parent=w.win))

    def _open_labeling_sobel(self):
        if not self._require_image(): return
        w = LabelingWindow(self.root, self.display_img)
        w.win.after(200, w._render_labeled)
        w.win.after(600, lambda: messagebox.showinfo(
            "Sobel", "1. Selectează obiectul 🎯\n2. Apasă 📐 Calcul Sobel.", parent=w.win))

    # ── Lab 5 handlers ────────────────────────────────────────────────────────

    def _quick_equalize(self):
        """Egalizare rapidă direct în editor (fără fereastră nouă)."""
        if not self._require_image(): return
        self.display_img = equalize_histogram(self.display_img)
        self.render_main()

    def _open_morphology(self, operation):
        if not self._require_image(): return
        w = MorphologyWindow(self.root, self.display_img, operation)
        w.set_editor_callback(self._receive_from_morphology)

    def _receive_from_morphology(self, img):
        self.display_img = img
        self.render_main()

    # ── Analiză obiect (originale nemodificate) ───────────────────────────────

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
            messagebox.showwarning("Analiză", "Nu a fost detectat niciun obiect.")
            return None
        xc, yc = m10/m00, m01/m00
        return {"m00": m00, "xc": xc, "yc": yc, "coords": coords, "w": w, "h": h}

    def show_center_m1(self):
        data = self.verify_object()
        if not data: return
        copy = self.display_img.copy().convert("RGB")
        draw = ImageDraw.Draw(copy)
        xc, yc = data["xc"], data["yc"]
        draw.ellipse([xc-5, yc-5, xc+5, yc+5], fill="red", outline="white")
        self.render_with_custom(copy)
        messagebox.showinfo("Moment M1", f"Centrul de greutate:\nXc: {xc:.2f}\nYc: {yc:.2f}")

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
        messagebox.showinfo("Matrice Covarianță",
                            f"| {m20:.2f}   {m11:.2f} |\n| {m11:.2f}   {m02:.2f} |")

    def show_projections(self):
        data = self.verify_object()
        if not data: return
        ph = [0]*data["h"]; pv = [0]*data["w"]
        for x, y in data["coords"]:
            pv[x] += 1; ph[y] += 1
        plt.figure("Proiecții", figsize=(10, 4))
        plt.subplot(121); plt.plot(pv, color='blue'); plt.title("Proiecție Verticală")
        plt.subplot(122); plt.plot(ph, color='red');  plt.title("Proiecție Orizontală")
        plt.tight_layout(); plt.show()

    # ── Filtre & UI (originale nemodificate) ──────────────────────────────────

    def create_slider(self, parent, label, key, is_thresh=False):
        tk.Label(parent, text=label, fg="#888", bg="#1a1a1a", font=("Arial", 8)).pack(anchor="w")
        f, t = (0, 255) if is_thresh else (0.0, 2.0)
        res = 1 if is_thresh else 0.05
        tk.Scale(parent, from_=f, to=t, resolution=res, orient=tk.HORIZONTAL,
                 variable=self.vars[key], bg="#1a1a1a", fg="white", highlightthickness=0,
                 command=lambda _: self.preview_filter(self.active_mode)).pack(fill=tk.X, pady=(0,5))

    def update_sidebar_dynamic(self, mode):
        for widget in self.dynamic_slider_container.winfo_children(): widget.destroy()
        config = {
            "none":     [("Roșu (R)","r"),("Verde (G)","g"),("Albastru (B)","b")],
            "negative": [("Canal R","r"),("Canal G","g"),("Canal B","b")],
            "gray(1)":  [("Gray Gain","gray_gain")], "gray(2)":  [("Gray Gain","gray_gain")],
            "gray(3)":  [("Gray Gain","gray_gain")],
            "binarize": [("Prag (0-255)","thresh",True)],
            "cmyk":     [("Cyan","c"),("Magenta","m"),("Yellow","y_c"),("Black (K)","k")],
            "yuv":      [("Luma (Y)","y_luma"),("Chroma (U)","u"),("Chroma (V)","v")],
            "ycbcr":    [("Luma (Y)","y_luma"),("Chroma (Cb)","cb"),("Chroma (Cr)","cr")],
            "rgb_back": [("Canal R","r"),("Canal G","g"),("Canal B","b")],
            "hsv":      [("Nuance (H)","h"),("Saturation (S)","s"),("Value (V)","v_hsv")],
        }
        for item in config.get(mode.lower(), []):
            if len(item) == 3: self.create_slider(self.dynamic_slider_container, item[0], item[1], item[2])
            else:              self.create_slider(self.dynamic_slider_container, item[0], item[1])

    def apply_math(self, img, mode):
        m = mode.lower(); w, h = img.size; pixels = img.load()
        res = Image.new("RGB", (w, h)); new_pix = res.load()
        v = {k: var.get() for k, var in self.vars.items()}
        b_f, c_f = v["bright"], v["contrast"]
        for x in range(w):
            for y in range(h):
                r, g, b = pixels[x, y]
                if m == "gray(1)":   val=int(((r+g+b)//3)*v["gray_gain"]); nr=ng=nb=val
                elif m == "gray(2)": val=int((0.299*r+0.587*g+0.114*b)*v["gray_gain"]); nr=ng=nb=val
                elif m == "gray(3)": val=int(((max(r,g,b)+min(r,g,b))//2)*v["gray_gain"]); nr=ng=nb=val
                elif m == "binarize":
                    val=255 if int(0.299*r+0.587*g+0.114*b)>v["thresh"] else 0; nr=ng=nb=val
                elif m == "cmyk":
                    rf,gf,bf_n=r/255,g/255,b/255; k_f=(1-max(rf,gf,bf_n))*v["k"]
                    if k_f<1:
                        nr=int(((1-rf-k_f)/(1-k_f))*255*v["c"])
                        ng=int(((1-gf-k_f)/(1-k_f))*255*v["m"])
                        nb=int(((1-bf_n-k_f)/(1-k_f))*255*v["y_c"])
                    else: nr=ng=nb=0
                elif m == "negative": nr,ng,nb=int((255-r)*v["r"]),int((255-g)*v["g"]),int((255-b)*v["b"])
                elif m == "yuv":
                    y_v=int((0.3*r+0.6*g+0.1*b)*v["y_luma"])
                    nr,ng,nb=y_v,int((0.74*(r-y_v)+0.27*(b-y_v))*v["u"])+128,int((0.48*(r-y_v)+0.41*(b-y_v))*v["v"])+128
                elif m in ("ycbcr","rgb_back"):
                    y_v=int((0.299*r+0.587*g+0.114*b)*v["y_luma"])
                    cb=int((-0.1687*r-0.3313*g+0.5*b)*v["cb"])+128
                    cr=int((0.5*r-0.4187*g-0.0813*b)*v["cr"])+128
                    if m=="ycbcr": nr,ng,nb=y_v,cb,cr
                    else:
                        nr=int((y_v+1.402*(cr-128))*v["r"])
                        ng=int((y_v-0.34414*(cb-128)-0.71414*(cr-128))*v["g"])
                        nb=int((y_v+1.772*(cb-128))*v["b"])
                elif m == "hsv":
                    rf,gf,bf_n=r/255,g/255,b/255
                    M,mn=max(rf,gf,bf_n),min(rf,gf,bf_n); C=M-mn; V=M
                    S=(C/V) if V else 0
                    if C:
                        if M==rf:   H=60*(((gf-bf_n)/C)%6)
                        elif M==gf: H=60*((bf_n-rf)/C+2)
                        else:       H=60*((rf-gf)/C+4)
                    else: H=0
                    nr,ng,nb=int((H/360)*255*v["h"]),int(S*255*v["s"]),int(V*255*v["v_hsv"])
                else: nr,ng,nb=int(r*v["r"]),int(g*v["g"]),int(b*v["b"])
                new_pix[x,y]=(max(0,min(255,int(nr*b_f*c_f))),
                               max(0,min(255,int(ng*b_f*c_f))),
                               max(0,min(255,int(nb*b_f*c_f))))
        return res

    def render_with_custom(self, img):
        w_c, h_c = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w_c < 10: w_c, h_c = 800, 600
        img_r = img.copy(); img_r.thumbnail((w_c, h_c))
        self.tk_main = ImageTk.PhotoImage(img_r)
        self.canvas.delete("all")
        self.canvas.create_image(w_c//2, h_c//2, image=self.tk_main)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Bitmap", "*.bmp")])
        if path:
            self.original_img = Image.open(path).convert("RGB")
            self.display_img = self.original_img.copy(); self.render_main()

    def toggle_gallery(self):
        if self.bottom_panel.winfo_ismapped(): self.bottom_panel.pack_forget()
        else: self.bottom_panel.pack(side=tk.BOTTOM, fill=tk.X); self.generate_gallery()

    def generate_gallery(self):
        for w in self.bottom_panel.winfo_children(): w.destroy()
        canvas = tk.Canvas(self.bottom_panel, bg="#181818", height=160, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        frame = tk.Frame(canvas, bg="#181818")
        canvas.create_window((0,0), window=frame, anchor="nw")
        modes = [("Original","none"),("Negative","negative"),("Binarize","binarize"),
                 ("Gray Avg","gray(1)"),("Gray Lum","gray(2)"),("Gray Desat","gray(3)"),
                 ("CMYK","cmyk"),("YUV","yuv"),("YCbCr","ycbcr"),("HSV","hsv"),("RGB Back","rgb_back")]
        thumb = self.original_img.copy(); thumb.thumbnail((100,100))
        for name, m in modes:
            p = ImageTk.PhotoImage(self.apply_math(thumb.copy(), m))
            self.previews[name] = p
            btn_f = tk.Frame(frame, bg="#181818", padx=10); btn_f.pack(side=tk.LEFT)
            tk.Button(btn_f, image=p, command=lambda md=m: self.preview_filter(md)).pack()
            tk.Label(btn_f, text=name, fg="white", bg="#181818", font=("Arial",7)).pack()
        frame.update_idletasks(); canvas.config(scrollregion=canvas.bbox("all"))

    def preview_filter(self, mode):
        self.active_mode = mode; self.update_sidebar_dynamic(mode)
        if self.original_img:
            self.display_img = self.apply_math(self.original_img.copy(), mode); self.render_main()

    def render_main(self):
        if self.display_img: self.render_with_custom(self.display_img)

    def confirm_save(self):
        if self.display_img: self.original_img = self.display_img.copy()

    def cancel_edits(self):
        for k, v in self.vars.items(): v.set(128 if k=="thresh" else 1.0)
        if self.original_img: self.display_img = self.original_img.copy(); self.render_main()

    def export_to_disk(self):
        if not self.display_img: return
        path = filedialog.asksaveasfilename(defaultextension=".bmp")
        if path: self.display_img.save(path)

    def show_histogram(self):
        if not self.display_img: return
        pix = self.display_img.convert("L").getdata()
        plt.figure("Histograma", figsize=(5,3)); plt.hist(pix, bins=256, color='gray'); plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedBitmapEditor(root)
    root.mainloop()