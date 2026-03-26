import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

class AdvancedBitmapEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Prelucrare Imagini")
        self.root.geometry("1450x950")
        self.root.configure(bg="#121212")

        self.original_img = None 
        self.display_img = None  
        self.active_mode = "none"
        self.previews = {}
        
        # Dicționar cu toți parametrii tăi originali
        self.vars = {
            "bright": tk.DoubleVar(value=1.0), "contrast": tk.DoubleVar(value=1.0),
            "r": tk.DoubleVar(value=1.0), "g": tk.DoubleVar(value=1.0), "b": tk.DoubleVar(value=1.0),
            "c": tk.DoubleVar(value=1.0), "m": tk.DoubleVar(value=1.0), "y_c": tk.DoubleVar(value=1.0), "k": tk.DoubleVar(value=1.0),
            "y_luma": tk.DoubleVar(value=1.0), "u": tk.DoubleVar(value=1.0), "v": tk.DoubleVar(value=1.0),
            "cb": tk.DoubleVar(value=1.0), "cr": tk.DoubleVar(value=1.0),
            "gray_gain": tk.DoubleVar(value=1.0),
            "thresh": tk.DoubleVar(value=128),
            "h": tk.DoubleVar(value=1.0), 
            "s": tk.DoubleVar(value=1.0), 
            "v": tk.DoubleVar(value=1.0)
        }

        self.setup_ui()

    def setup_ui(self):
        # Meniu Superior
        top_bar = tk.Frame(self.root, bg="#1a1a1a", height=60)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top_bar, text="📁 Upload", command=self.load_image, bg="#333", fg="white", padx=15, borderwidth=0).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(top_bar, text="✨ Filtre", command=self.toggle_gallery, bg="#444", fg="white", padx=15, borderwidth=0).pack(side=tk.LEFT, padx=5, pady=10)
        tk.Button(top_bar, text="💾 Save", command=self.export_to_disk, 
                  bg="#2e7d32", fg="white", padx=15, borderwidth=0, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5, pady=10)
        tk.Button(top_bar, text="Histograma", command=self.show_histogram, bg="#333", fg="white", padx=15, borderwidth=0).pack(side=tk.LEFT, padx=10, pady=10)

        self.main_container = tk.Frame(self.root, bg="#121212")

        self.main_container.pack(expand=True, fill=tk.BOTH)

        # Sidebar
        self.sidebar = tk.Frame(self.main_container, bg="#1a1a1a", width=280)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        
        tk.Label(self.sidebar, text="⚙️ AJUSTĂRI ", fg="cyan", bg="#1a1a1a", font=("Arial", 10, "bold")).pack(pady=15)
        
        # SECȚIUNEA FIXĂ
        fixed_ctrl = tk.LabelFrame(self.sidebar, text=" Control Universal ", fg="#888", bg="#1a1a1a", padx=10, pady=10)
        fixed_ctrl.pack(fill=tk.X, padx=10, pady=5)
        self.create_slider(fixed_ctrl, "Luminozitate", "bright")
        self.create_slider(fixed_ctrl, "Contrast", "contrast")

        # SECȚIUNEA DINAMICĂ
        self.dynamic_ctrl = tk.LabelFrame(self.sidebar, text=" Canale Filtru ", fg="yellow", bg="#1a1a1a", padx=10, pady=10)
        self.dynamic_ctrl.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.dynamic_slider_container = tk.Frame(self.dynamic_ctrl, bg="#1a1a1a")
        self.dynamic_slider_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.main_container, bg="#121212", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=20, pady=20)

        self.bottom_panel = tk.Frame(self.root, bg="#181818", height=220)
        self.update_sidebar_dynamic("none")

    def create_slider(self, parent, label, key, is_thresh=False):
        tk.Label(parent, text=label, fg="#aaa", bg="#1a1a1a", font=("Arial", 8)).pack(anchor="w")
        
        # Dacă e slider-ul de prag, folosim 0-255, altfel 0-2 pentru gain/canale
        f, t = (0, 255) if is_thresh else (0.0, 2.0)
        res = 1 if is_thresh else 0.05

        tk.Scale(parent, from_=f, to=t, resolution=res, orient=tk.HORIZONTAL,
                 variable=self.vars[key], bg="#1a1a1a", fg="white", highlightthickness=0,
                 command=lambda _: self.preview_filter(self.active_mode)).pack(fill=tk.X, pady=(0, 10))

    def update_sidebar_dynamic(self, mode):
        for widget in self.dynamic_slider_container.winfo_children(): widget.destroy()
        
        m = mode.lower()
        if m == "binarize":
            self.create_slider(self.dynamic_slider_container, "Prag (0-255)", "thresh", True)
        else:
            config = {
                "none": [("Roșu (R)", "r"), ("Verde (G)", "g"), ("Albastru (B)", "b")],
                "negative": [("Canal R", "r"), ("Canal G", "g"), ("Canal B", "b")],
                "gray(1)": [("Gray scale", "gray_gain")], 
                "gray(2)": [("Gray scale", "gray_gain")], 
                "gray(3)": [("Gray scale", "gray_gain")],
                "cmyk": [("Cyan", "c"), ("Magenta", "m"), ("Yellow", "y_c"), ("Black (K)", "k")],
                "yuv": [("Luma (Y)", "y_luma"), ("Chroma (U)", "u"), ("Chroma (V)", "v")],
                "ycbcr": [("Luma (Y)", "y_luma"), ("Chroma (Cb)", "cb"), ("Chroma (Cr)", "cr")],
                "rgb_back": [("Canal R", "r"), ("Canal G", "g"), ("Canal B", "b")],
                "hsv": [("Nuance (H)", "h"), ("Saturation (S)", "s"), ("Value (V)", "v")]
            }
            active_ctrls = config.get(m, [])
            for label, key in active_ctrls:
                self.create_slider(self.dynamic_slider_container, label, key)
        
        tk.Button(self.dynamic_slider_container, text="Reset tot", command=self.reset_params, 
                  bg="#333", fg="#fff", borderwidth=0).pack(pady=10, fill=tk.X)

    def reset_params(self):
        for k, v in self.vars.items():
            v.set(128 if k == "thresh" else 1.0)
        self.preview_filter(self.active_mode)

    def apply_math(self, img, mode):
        m = mode.lower()
        w, h = img.size
        pixels = img.load()
        res = Image.new("RGB", (w, h))
        new_pix = res.load()
        v = {k: var.get() for k, var in self.vars.items()}
        b_f, c_f = v["bright"], v["contrast"]

        for x in range(w):
            for y in range(h):
                r, g, b = pixels[x, y]
                
                if m == "gray(1)": 
                    val = int(((r+g+b)//3) * v["gray_gain"])
                    nr, ng, nb = val, val, val
                elif m == "gray(2)": 
                    val = int((0.299*r+0.587*g+0.114*b)*v["gray_gain"])
                    nr, ng, nb = val, val, val
                elif m == "gray(3)": 
                    val = int(((max(r,g,b)+min(r,g,b))//2)*v["gray_gain"])
                    nr, ng, nb = val, val, val
                elif m == "binarize":
                    gray = int(0.299*r + 0.587*g + 0.114*b)
                    val = 255 if gray > v["thresh"] else 0
                    nr, ng, nb = val, val, val
                elif m == "cmyk":
                    rf, gf, bf_n = r/255.0, g/255.0, b/255.0
                    k_final = (1.0-max(rf, gf, bf_n)) * v["k"]
                    if k_final < 1.0:
                        nr = int(((1.0-rf-k_final)/(1.0-k_final))*255 * v["c"])
                        ng = int(((1.0-gf-k_final)/(1.0-k_final))*255 * v["m"])
                        nb = int(((1.0-bf_n-k_final)/(1.0-k_final))*255 * v["y_c"])
                    else: nr = ng = nb = 0
                elif m == "negative": 
                    nr, ng, nb = int((255-r)*v["r"]), int((255-g)*v["g"]), int((255-b)*v["b"])
                elif m == "yuv":
                    y_v = int((0.3*r + 0.6*g + 0.1*b) * v["y_luma"])
                    u_v = int((0.74*(r - y_v) + 0.27*(b - y_v)) * v["u"]) + 128
                    v_v = int((0.48*(r - y_v) + 0.41*(b - y_v)) * v["v"]) + 128
                    nr, ng, nb = y_v, u_v, v_v
                elif m == "ycbcr" or m == "rgb_back":
                    y_v = int((0.299*r + 0.587*g + 0.114*b) * v["y_luma"])
                    cb = int((-0.1687*r - 0.3313*g + 0.5*b) * v["cb"]) + 128
                    cr = int((0.5*r - 0.4187*g - 0.0813*b) * v["cr"]) + 128
                    if m == "ycbcr": nr, ng, nb = y_v, cb, cr
                    else:
                        nr = int((y_v + 1.402 * (cr - 128)) * v["r"])
                        ng = int((y_v - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)) * v["g"])
                        nb = int((y_v + 1.772 * (cb - 128)) * v["b"])
                elif m == "hsv":
                    rf, gf, bf_n = r/255.0, g/255.0, b/255.0
                    M = max(rf, gf, bf_n)
                    min_val = min(rf, gf, bf_n) # am redenumit 'm' in 'min_val' ca sa nu se confunde cu modul
                    C = M - min_val
                    V = M
                    
                    S = (C / V) if V != 0 else 0

                    if C != 0:
                        if M == rf:
                            H = 60 * (((gf - bf_n) / C) % 6)
                        elif M == gf:
                            H = 60 * (((bf_n - rf) / C) + 2)
                        else:
                            H = 60 * (((rf - gf) / C) + 4)
                    else:
                        H = 0
                    
                    # Mapăm valorile pentru a fi vizibile ca imagine (0-255)
                    # H devine H/360 * 255, S devine S * 255, V devine V * 255
                    nr = int((H / 360) * 255 * v.get("h", 1.0))
                    ng = int(S * 255 * v.get("s", 1.0))
                    nb = int(V * 255 * v.get("v", 1.0))

                else: 
                    nr, ng, nb = int(r * v["r"]), int(g * v["g"]), int(b * v["b"])
                

                # Brightness & Contrast finale
                nr = int(nr * b_f * c_f); ng = int(ng * b_f * c_f); nb = int(nb * b_f * c_f)
                new_pix[x, y] = (max(0, min(255, nr)), max(0, min(255, ng)), max(0, min(255, nb)))
        return res

    # Restul metodelor (load_image, render_main, toggle_gallery, generate_gallery, etc.) 
    # rămân neschimbate față de versiunea ta funcțională.
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Bitmap", "*.bmp")])
        if path:
            self.original_img = Image.open(path).convert("RGB")
            self.display_img = self.original_img.copy()
            self.render_main()
            if self.bottom_panel.winfo_ismapped(): self.generate_gallery()

    def toggle_gallery(self):
        if not self.original_img: return
        if self.bottom_panel.winfo_ismapped(): self.bottom_panel.pack_forget()
        else:
            self.bottom_panel.pack(side=tk.BOTTOM, fill=tk.X)
            self.generate_gallery()

    def generate_gallery(self):
        for widget in self.bottom_panel.winfo_children(): widget.destroy()
        btn_frame = tk.Frame(self.bottom_panel, bg="#181818")
        btn_frame.pack(side=tk.RIGHT, padx=20, fill=tk.Y)
        tk.Button(btn_frame, text="✅ Salvează", command=self.confirm_save, bg="#2e7d32", fg="white", borderwidth=0, padx=20, pady=10).pack(pady=5, fill=tk.X)
        tk.Button(btn_frame, text="❌ Reset", command=self.cancel_edits, bg="#c62828", fg="white", borderwidth=0, padx=20, pady=10).pack(pady=5, fill=tk.X)

        container = tk.Frame(self.bottom_panel, bg="#181818")
        container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cv = tk.Canvas(container, bg="#181818", highlightthickness=0, height=180)
        sb = tk.Scrollbar(container, orient="horizontal", command=cv.xview)
        sf = tk.Frame(cv, bg="#181818")
        sf.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=sf, anchor="nw")
        cv.configure(xscrollcommand=sb.set)
        cv.pack(side=tk.TOP, fill=tk.BOTH, expand=True); sb.pack(side=tk.BOTTOM, fill=tk.X)

        modes = [("Original", "none"), ("Gray(1)", "gray(1)"), ("Gray(2)", "gray(2)"), 
                 ("Gray(3)", "gray(3)"), ("Alb/Negru", "binarize"), ("CMYK", "cmyk"), 
                 ("Negativ", "negative"), ("YUV", "yuv"), ("YCbCr", "ycbcr"), ("RGB Back", "rgb_back"), ("HSV", "hsv")]

        thumb_base = self.original_img.copy()
        thumb_base.thumbnail((100, 100))
        for name, m in modes:
            proc_thumb = self.apply_math(thumb_base.copy(), m)
            photo = ImageTk.PhotoImage(proc_thumb)
            self.previews[name] = photo 
            box = tk.Frame(sf, bg="#181818", padx=10)
            box.pack(side=tk.LEFT, pady=10)
            tk.Button(box, image=photo, command=lambda mode=m: self.preview_filter(mode), bg="#252525", borderwidth=1).pack()
            tk.Label(box, text=name, fg="#888", bg="#181818", font=("Arial", 8)).pack()

    def preview_filter(self, mode):
        if mode != self.active_mode:
            self.active_mode = mode
            self.update_sidebar_dynamic(mode)
        if self.original_img:
            self.display_img = self.apply_math(self.original_img.copy(), mode)
            self.render_main()

    def render_main(self):
        if self.display_img:
            self.root.update_idletasks()
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if w < 10: w, h = 800, 500
            img_r = self.display_img.copy(); img_r.thumbnail((w, h))
            self.tk_main = ImageTk.PhotoImage(img_r)
            self.canvas.delete("all")
            self.canvas.create_image(w//2, h//2, image=self.tk_main)

    def confirm_save(self):
        if self.display_img: self.original_img = self.display_img.copy(); messagebox.showinfo("Succes", "Imagine salvată!")

    def cancel_edits(self):
        self.reset_params(); self.active_mode = "none"
        self.display_img = self.original_img.copy(); self.render_main()

    def export_to_disk(self):
        if not self.display_img: return
        file_path = filedialog.asksaveasfilename(defaultextension=".bmp", filetypes=[("Bitmap", "*.bmp")])
        if file_path: self.display_img.save(file_path); messagebox.showinfo("Export", "Salvat!")

    def calculate_histogram(self):

        img = self.display_img
        width, height = img.size
        
        # Inițializăm histograma cu 256 de zerouri
        histogram = [0] * 256
        
        pixels = img.load()

        # Parcurgem imaginea
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                
                # Calculăm media pentru tonul de gri (la fel ca în codul tău Java)
                gray = (r + g + b) // 3
                
                # Incrementăm în histogramă
                histogram[gray] += 1
                
        return histogram

    def show_histogram(self):
        # 1. Verificăm dacă avem o imagine încărcată
        if not self.display_img:
            messagebox.showwarning("Atenție", "Încarcă o imagine mai întâi!")
            return
        
        # 2. Calculăm datele (folosind metoda ta existentă)
        hist_data = self.calculate_histogram()
        
        # 3. Afișăm graficul
        self.plot_histogram(hist_data)

    def plot_histogram(self, histogram):
        # Creăm graficul într-o fereastră separată (Matplotlib face asta automat cu .show())
        plt.figure("Analiză Histogramă", figsize=(8, 5))
        plt.bar(range(256), histogram, color='#555555', width=1.0)
        plt.title("Distribuția tonurilor de gri")
        plt.xlabel("Intensitate Pixel (0-255)")
        plt.ylabel("Frecvență (Număr Pixeli)")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk(); app = AdvancedBitmapEditor(root); root.mainloop()