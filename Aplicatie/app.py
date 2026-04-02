import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt

class AdvancedBitmapEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Bitmap Editor - Toate Filtrele + Analiză")
        self.root.geometry("1550x950")
        self.root.configure(bg="#121212")

        self.original_img = None 
        self.display_img = None  
        self.active_mode = "none"
        self.previews = {}
        
        # Variabile pentru toți parametrii posibili la filtre
        self.vars = {
            "bright": tk.DoubleVar(value=1.0), "contrast": tk.DoubleVar(value=1.0),
            "r": tk.DoubleVar(value=1.0), "g": tk.DoubleVar(value=1.0), "b": tk.DoubleVar(value=1.0),
            "c": tk.DoubleVar(value=1.0), "m": tk.DoubleVar(value=1.0), "y_c": tk.DoubleVar(value=1.0), "k": tk.DoubleVar(value=1.0),
            "y_luma": tk.DoubleVar(value=1.0), "u": tk.DoubleVar(value=1.0), "v": tk.DoubleVar(value=1.0),
            "cb": tk.DoubleVar(value=1.0), "cr": tk.DoubleVar(value=1.0),
            "gray_gain": tk.DoubleVar(value=1.0),
            "thresh": tk.DoubleVar(value=128),
            "h": tk.DoubleVar(value=1.0), "s": tk.DoubleVar(value=1.0), "v_hsv": tk.DoubleVar(value=1.0)
        }

        self.setup_ui()

    def setup_ui(self):
        # Bară de sus cu butoane
        top_bar = tk.Frame(self.root, bg="#1a1a1a", height=60)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top_bar, text="📁 Încarcă BMP", command=self.load_image, bg="#333", fg="white", padx=15, borderwidth=0).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(top_bar, text="✨ Galerie Filtre", command=self.toggle_gallery, bg="#444", fg="white", padx=15, borderwidth=0).pack(side=tk.LEFT, padx=5, pady=10)
        tk.Button(top_bar, text="💾 Exportă Fișier", command=self.export_to_disk, bg="#2e7d32", fg="white", padx=15, borderwidth=0).pack(side=tk.LEFT, padx=5, pady=10)
        tk.Button(top_bar, text="📊 Histograme", command=self.show_histogram, bg="#333", fg="white", padx=15, borderwidth=0).pack(side=tk.LEFT, padx=10, pady=10)

        self.main_container = tk.Frame(self.root, bg="#121212")
        self.main_container.pack(expand=True, fill=tk.BOTH)
        
        # Sidebar pentru controale și parametri
        self.sidebar = tk.Frame(self.main_container, bg="#1a1a1a", width=320)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        
        edit_frame = tk.LabelFrame(self.sidebar, text=" CONTROL EDITARE ", fg="#4caf50", bg="#1a1a1a", padx=10, pady=10)
        edit_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Button(edit_frame, text="✅ Salvează în Memorie", command=self.confirm_save, bg="#2e7d32", fg="white", borderwidth=0).pack(fill=tk.X, pady=2)
        tk.Button(edit_frame, text="❌ Resetează Imaginea", command=self.cancel_edits, bg="#c62828", fg="white", borderwidth=0).pack(fill=tk.X, pady=2)

        tk.Label(self.sidebar, text="⚙️ PARAMETRI FILTRU", fg="cyan", bg="#1a1a1a", font=("Arial", 9, "bold")).pack(pady=5)
        
        uni_frame = tk.Frame(self.sidebar, bg="#1a1a1a", padx=10)
        uni_frame.pack(fill=tk.X)
        self.create_slider(uni_frame, "Luminozitate Globală", "bright")
        self.create_slider(uni_frame, "Contrast Global", "contrast")

        self.dynamic_slider_container = tk.Frame(self.sidebar, bg="#1a1a1a", padx=10)
        self.dynamic_slider_container.pack(fill=tk.X, expand=True)

        tk.Label(self.sidebar, text="🔍 ANALIZĂ OBIECT", fg="yellow", bg="#1a1a1a", font=("Arial", 9, "bold")).pack(pady=10)
        analysis_frame = tk.Frame(self.sidebar, bg="#1a1a1a", padx=10)
        analysis_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=20)
        
        # Butoane mapate la funcțiile noi
        tk.Button(analysis_frame, text="📍 Centru de Greutate (M1)", command=self.show_center_m1, bg="#37474f", fg="white", anchor="w", padx=10).pack(fill=tk.X, pady=2)
        tk.Button(analysis_frame, text="📐 Momente Centrale (M2)", command=self.show_moments_m2, bg="#37474f", fg="white", anchor="w", padx=10).pack(fill=tk.X, pady=2)
        tk.Button(analysis_frame, text="🧮 Matrice Covarianță", command=self.show_covariance, bg="#37474f", fg="white", anchor="w", padx=10).pack(fill=tk.X, pady=2)
        tk.Button(analysis_frame, text="📉 Proiecții Orizontal/Vertical", command=self.show_projections, bg="#37474f", fg="white", anchor="w", padx=10).pack(fill=tk.X, pady=2)

        self.canvas = tk.Canvas(self.main_container, bg="#121212", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=20, pady=20)

        self.bottom_panel = tk.Frame(self.root, bg="#181818", height=200)
        self.update_sidebar_dynamic("none")

    # verificăm dacă există o imagine încărcată și calculăm datele necesare pentru analiză
    def verify_object(self):
        
        if not self.display_img:
            return None
        
        img = self.display_img.convert("RGB")
        w, h = img.size
        pixels = img.load()
        m00, m10, m01 = 0, 0, 0
        coords = []

        for y in range(h):
            for x in range(w):
                r, g, b = pixels[x, y]
                # Prag pentru a detecta obiectul (orice nu e alb)
                if r < 240 or g < 240 or b < 240:
                    m00 += 1 # Numărul de pixeli care fac parte din obiect
                    m10 += x # Sumă ponderată pe axa X
                    m01 += y # Sumă ponderată pe axa Y
                    coords.append((x, y)) 
        
        if m00 == 0:
            messagebox.showwarning("Analiză", "Nu a fost detectat niciun obiect (imaginea este albă).")
            return None

        # Calculăm centrul de greutate (Xc, Yc)     
        xc, yc = m10/m00, m01/m00
        return {"m00": m00, "xc": xc, "yc": yc, "coords": coords, "w": w, "h": h}

    # FUNCȚII DE ANALIZĂ SEPARATE
    def show_center_m1(self):
        data = self.verify_object()
        if not data: return

        copy = self.display_img.copy().convert("RGB")
        draw = ImageDraw.Draw(copy)
        xc, yc = data["xc"], data["yc"]
        
        # Desenăm punctul roșu
        draw.ellipse([xc-5, yc-5, xc+5, yc+5], fill="red", outline="white")
        
        # Afișăm imaginea cu punctul pe canvas
        self.render_with_custom(copy)
        messagebox.showinfo("Moment M1", f"Centrul de greutate:\nXc: {xc:.2f}\nYc: {yc:.2f}")

    def show_moments_m2(self):
        data = self.verify_object()
        if not data: return

        # Calculăm momentele centrale de ordin 2
        xc, yc = data["xc"], data["yc"]
        mu20 = sum((p[0]-xc)**2 for p in data["coords"]) # Momentul orizontal
        mu02 = sum((p[1]-yc)**2 for p in data["coords"]) # Momentul vertical
        mu11 = sum((p[0]-xc)*(p[1]-yc) for p in data["coords"]) # Momentul mixt

        messagebox.showinfo("Momente Centrale M2", 
                            f"Moment orizontal (mu20): {mu20:.0f}\n"
                            f"Moment vertical (mu02): {mu02:.0f}\n"
                            f"Moment mixt (mu11): {mu11:.0f}")

    # Matricea de covarianta
    def show_covariance(self):
        data = self.verify_object()
        if not data: return

        # Calculăm elementele matricei de covarianță
        m00, xc, yc = data["m00"], data["xc"], data["yc"]
        m20 = sum((p[0]-xc)**2 for p in data["coords"]) / m00 # Varianta pe axa X
        m02 = sum((p[1]-yc)**2 for p in data["coords"]) / m00 # Varianta pe axa Y
        m11 = sum((p[0]-xc)*(p[1]-yc) for p in data["coords"]) / m00 # Covarianța între X și Y

        messagebox.showinfo("Matrice Covarianță", 
                            f"Matricea:\n\n"
                            f"| {m20:.2f}   {m11:.2f} |\n"
                            f"| {m11:.2f}   {m02:.2f} |")

    # Proiecțiile orizontală și verticală
    def show_projections(self):
        data = self.verify_object()
        if not data: return

        ph = [0] * data["h"]
        pv = [0] * data["w"]
        
        for x, y in data["coords"]:
            pv[x] += 1
            ph[y] += 1

        plt.figure("Proiecții Obiect", figsize=(10, 4))
        plt.subplot(121)
        plt.plot(pv, color='blue')
        plt.title("Proiecție Verticală (pe axa X)")
        
        plt.subplot(122)
        plt.plot(ph, color='red')
        plt.title("Proiecție Orizontală (pe axa Y)")
        
        plt.tight_layout()
        plt.show()

    # Sidebar dinamică pentru parametrii specifici fiecărui filtru
    def create_slider(self, parent, label, key, is_thresh=False):
        tk.Label(parent, text=label, fg="#888", bg="#1a1a1a", font=("Arial", 8)).pack(anchor="w")
        f, t = (0, 255) if is_thresh else (0.0, 2.0)
        res = 1 if is_thresh else 0.05
        tk.Scale(parent, from_=f, to=t, resolution=res, orient=tk.HORIZONTAL, variable=self.vars[key], bg="#1a1a1a", fg="white", highlightthickness=0, command=lambda _: self.preview_filter(self.active_mode)).pack(fill=tk.X, pady=(0, 5))

    # Actualizarea dinamică a sidebar-ului în funcție de filtrul selectat
    def update_sidebar_dynamic(self, mode):
        for widget in self.dynamic_slider_container.winfo_children(): widget.destroy()
        m = mode.lower()
        config = {
            "none": [("Roșu (R)", "r"), ("Verde (G)", "g"), ("Albastru (B)", "b")],
            "negative": [("Canal R", "r"), ("Canal G", "g"), ("Canal B", "b")],
            "gray(1)": [("Gray Gain", "gray_gain")], "gray(2)": [("Gray Gain", "gray_gain")], "gray(3)": [("Gray Gain", "gray_gain")],
            "binarize": [("Prag (0-255)", "thresh", True)],
            "cmyk": [("Cyan", "c"), ("Magenta", "m"), ("Yellow", "y_c"), ("Black (K)", "k")],
            "yuv": [("Luma (Y)", "y_luma"), ("Chroma (U)", "u"), ("Chroma (V)", "v")],
            "ycbcr": [("Luma (Y)", "y_luma"), ("Chroma (Cb)", "cb"), ("Chroma (Cr)", "cr")],
            "rgb_back": [("Canal R", "r"), ("Canal G", "g"), ("Canal B", "b")],
            "hsv": [("Nuance (H)", "h"), ("Saturation (S)", "s"), ("Value (V)", "v_hsv")]
        }
        for item in config.get(m, []):
            if len(item) == 3: self.create_slider(self.dynamic_slider_container, item[0], item[1], item[2])
            else: self.create_slider(self.dynamic_slider_container, item[0], item[1])

    # Funcția centrală care aplică toate transformările matematice în funcție de modul selectat
    def apply_math(self, img, mode):
        m = mode.lower(); w, h = img.size; pixels = img.load()
        res = Image.new("RGB", (w, h)); new_pix = res.load()
        v = {k: var.get() for k, var in self.vars.items()}
        b_f, c_f = v["bright"], v["contrast"]

        for x in range(w):
            for y in range(h):
                r, g, b = pixels[x, y]
                if m == "gray(1)": val = int(((r+g+b)//3) * v["gray_gain"]); nr=ng=nb=val
                elif m == "gray(2)": val = int((0.299*r+0.587*g+0.114*b)*v["gray_gain"]); nr=ng=nb=val
                elif m == "gray(3)": val = int(((max(r,g,b)+min(r,g,b))//2)*v["gray_gain"]); nr=ng=nb=val
                elif m == "binarize":
                    gray = int(0.299*r + 0.587*g + 0.114*b)
                    val = 255 if gray > v["thresh"] else 0; nr=ng=nb=val
                elif m == "cmyk":
                    rf, gf, bf_n = r/255.0, g/255.0, b/255.0
                    k_final = (1.0-max(rf, gf, bf_n)) * v["k"]
                    if k_final < 1.0:
                        nr = int(((1.0-rf-k_final)/(1.0-k_final))*255 * v["c"])
                        ng = int(((1.0-gf-k_final)/(1.0-k_final))*255 * v["m"])
                        nb = int(((1.0-bf_n-k_final)/(1.0-k_final))*255 * v["y_c"])
                    else: nr = ng = nb = 0
                elif m == "negative": nr, ng, nb = int((255-r)*v["r"]), int((255-g)*v["g"]), int((255-b)*v["b"])
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
                    M, min_val = max(rf, gf, bf_n), min(rf, gf, bf_n); C = M - min_val; V = M
                    S = (C / V) if V != 0 else 0
                    if C != 0:
                        if M == rf: H = 60 * (((gf - bf_n) / C) % 6)
                        elif M == gf: H = 60 * (((bf_n - rf) / C) + 2)
                        else: H = 60 * (((rf - gf) / C) + 4)
                    else: H = 0
                    nr, ng, nb = int((H/360)*255*v["h"]), int(S*255*v["s"]), int(V*255*v["v_hsv"])
                else: nr, ng, nb = int(r * v["r"]), int(g * v["g"]), int(b * v["b"])
                
                nr = max(0, min(255, int(nr * b_f * c_f)))
                ng = max(0, min(255, int(ng * b_f * c_f)))
                nb = max(0, min(255, int(nb * b_f * c_f)))
                new_pix[x, y] = (nr, ng, nb)
        return res


    def render_with_custom(self, img):
        w_c, h_c = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w_c < 10: w_c, h_c = 800, 600
        img_r = img.copy(); img_r.thumbnail((w_c, h_c))
        self.tk_main = ImageTk.PhotoImage(img_r)
        self.canvas.delete("all"); self.canvas.create_image(w_c//2, h_c//2, image=self.tk_main)

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
        modes = [("Original", "none"), ("Negative", "negative"), ("Binarize", "binarize"), ("Gray Avg", "gray(1)"), ("Gray Lum", "gray(2)"), ("Gray Desat", "gray(3)"), ("CMYK", "cmyk"), ("YUV", "yuv"), ("YCbCr", "ycbcr"), ("HSV", "hsv"), ("RGB Back", "rgb_back")]
        thumb = self.original_img.copy(); thumb.thumbnail((100,100))
        for name, m in modes:
            p = ImageTk.PhotoImage(self.apply_math(thumb.copy(), m))
            self.previews[name] = p
            btn_f = tk.Frame(frame, bg="#181818", padx=10); btn_f.pack(side=tk.LEFT)
            tk.Button(btn_f, image=p, command=lambda md=m: self.preview_filter(md)).pack()
            tk.Label(btn_f, text=name, fg="white", bg="#181818", font=("Arial", 7)).pack()
        frame.update_idletasks(); canvas.config(scrollregion=canvas.bbox("all"))

    def preview_filter(self, mode):
        self.active_mode = mode; self.update_sidebar_dynamic(mode)
        if self.original_img: self.display_img = self.apply_math(self.original_img.copy(), mode); self.render_main()

    def render_main(self):
        if self.display_img: self.render_with_custom(self.display_img)

    def confirm_save(self):
        if self.display_img: self.original_img = self.display_img.copy(); messagebox.showinfo("OK", "Salvat în memorie!")

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
    root = tk.Tk(); app = AdvancedBitmapEditor(root); root.mainloop()