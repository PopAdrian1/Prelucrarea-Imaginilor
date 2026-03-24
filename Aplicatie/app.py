import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

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
        
        # Parametri universali + specifici
        self.vars = {
            "bright": tk.DoubleVar(value=1.0), "contrast": tk.DoubleVar(value=1.0),
            "r": tk.DoubleVar(value=1.0), "g": tk.DoubleVar(value=1.0), "b": tk.DoubleVar(value=1.0),
            "c": tk.DoubleVar(value=1.0), "m": tk.DoubleVar(value=1.0), "y_c": tk.DoubleVar(value=1.0), "k": tk.DoubleVar(value=1.0),
            "y_luma": tk.DoubleVar(value=1.0), "u": tk.DoubleVar(value=1.0), "v": tk.DoubleVar(value=1.0),
            "cb": tk.DoubleVar(value=1.0), "cr": tk.DoubleVar(value=1.0),
            "gray_gain": tk.DoubleVar(value=1.0),
            "thresh": tk.DoubleVar(value=128) 
        }

        self.setup_ui()

    def setup_ui(self):
        # Meniu Superior
        top_bar = tk.Frame(self.root, bg="#1a1a1a", height=60)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top_bar, text="📁 Upload", command=self.load_image, bg="#333", fg="white", padx=15, borderwidth=0).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(top_bar, text="✨ Filtre", command=self.toggle_gallery, bg="#444", fg="white", padx=15, borderwidth=0).pack(side=tk.LEFT, padx=5, pady=10)
        tk.Button(top_bar, text="💾 Salvează pe Disk", command=self.export_to_disk, 
                  bg="#2e7d32", fg="white", padx=15, borderwidth=0, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5, pady=10)
        
        # Main Layout
        self.main_container = tk.Frame(self.root, bg="#121212")
        self.main_container.pack(expand=True, fill=tk.BOTH)

        # Sidebar
        self.sidebar = tk.Frame(self.main_container, bg="#1a1a1a", width=280)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        
        # Titlu Sidebar
        tk.Label(self.sidebar, text="⚙️ AJUSTĂRI ", fg="cyan", bg="#1a1a1a", font=("Arial", 10, "bold")).pack(pady=15)
        
        # SECȚIUNEA FIXĂ: Luminozitate & Contrast (mereu prezente)
        fixed_ctrl = tk.LabelFrame(self.sidebar, text=" Control Universal ", fg="#888", bg="#1a1a1a", padx=10, pady=10)
        fixed_ctrl.pack(fill=tk.X, padx=10, pady=5)
        
        self.create_slider(fixed_ctrl, "Luminozitate", "bright")
        self.create_slider(fixed_ctrl, "Contrast", "contrast")

        # SECȚIUNEA DINAMICĂ: Canale specifice
        self.dynamic_ctrl = tk.LabelFrame(self.sidebar, text=" Canale Filtru ", fg="yellow", bg="#1a1a1a", padx=10, pady=10)
        self.dynamic_ctrl.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.dynamic_slider_container = tk.Frame(self.dynamic_ctrl, bg="#1a1a1a")
        self.dynamic_slider_container.pack(fill=tk.BOTH, expand=True)

        # Canvas
        self.canvas = tk.Canvas(self.main_container, bg="#121212", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=20, pady=20)

        # Panou Inferior
        self.bottom_panel = tk.Frame(self.root, bg="#181818", height=220)
        
        self.update_sidebar_dynamic("none")

    def create_slider(self, parent, label, key):
        tk.Label(parent, text=label, fg="#aaa", bg="#1a1a1a", font=("Arial", 8)).pack(anchor="w")
        tk.Scale(parent, from_=0.0, to=2.0, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.vars[key], bg="#1a1a1a", fg="white", highlightthickness=0,
                 command=lambda _: self.preview_filter(self.active_mode)).pack(fill=tk.X, pady=(0, 10))

    def update_sidebar_dynamic(self, mode):
        for widget in self.dynamic_slider_container.winfo_children(): widget.destroy()
        
        config = {
            "none": [("Roșu (R)", "r"), ("Verde (G)", "g"), ("Albastru (B)", "b")],
            "negative": [("Canal R", "r"), ("Canal G", "g"), ("Canal B", "b")],
            "gray(1)": [("Gain Gray", "gray_gain")], "gray(2)": [("Gain Gray", "gray_gain")], "gray(3)": [("Gain Gray", "gray_gain")],
            "cmyk": [("Cyan", "c"), ("Magenta", "m"), ("Yellow", "y_c"), ("Black (K)", "k")],
            "yuv": [("Luma (Y)", "y_luma"), ("Chroma (U)", "u"), ("Chroma (V)", "v")],
            "ycbcr": [("Luma (Y)", "y_luma"), ("Chroma (Cb)", "cb"), ("Chroma (Cr)", "cr")],
            "rgb_back": [("Canal R", "r"), ("Canal G", "g"), ("Canal B", "b")]
        }

        active_ctrls = config.get(mode.lower(), [])

        for label, key in active_ctrls:
            self.create_slider(self.dynamic_slider_container, label, key)
        
        tk.Button(self.dynamic_slider_container, text="Reset tot", command=self.reset_params, 
                  bg="#333", fg="#fff", borderwidth=0).pack(pady=10, fill=tk.X)

    def reset_params(self):
        for v in self.vars.values(): v.set(1.0)
        self.preview_filter(self.active_mode)

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
        
        # 1. Action Buttons (Fixe în dreapta)
        btn_frame = tk.Frame(self.bottom_panel, bg="#181818")
        btn_frame.pack(side=tk.RIGHT, padx=20, fill=tk.Y)

        tk.Button(btn_frame, text="✅ Salvează", command=self.confirm_save, bg="#2e7d32", fg="white", borderwidth=0, padx=20, pady=10).pack(pady=5, fill=tk.X)
        tk.Button(btn_frame, text="❌ Reset", command=self.cancel_edits, bg="#c62828", fg="white", borderwidth=0, padx=20, pady=10).pack(pady=5, fill=tk.X)

        # 2. Scroll Gallery
        container = tk.Frame(self.bottom_panel, bg="#181818")
        container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        cv = tk.Canvas(container, bg="#181818", highlightthickness=0, height=180)
        sb = tk.Scrollbar(container, orient="horizontal", command=cv.xview)
        sf = tk.Frame(cv, bg="#181818")

        sf.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=sf, anchor="nw")
        cv.configure(xscrollcommand=sb.set)

        cv.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.BOTTOM, fill=tk.X)

        modes = [("Original", "none"), ("Gray(1)", "gray(1)"), ("Gray(2)", "gray(2)"), 
                 ("Gray(3)", "gray(3)"),("Alb/Negru", "binarize"), ("CMYK", "cmyk"), ("Negativ", "negative"), 
                 ("YUV", "yuv"), ("YCbCr", "ycbcr"), ("RGB Back", "rgb_back")]

        thumb_base = self.original_img.copy()
        thumb_base.thumbnail((100, 100))

        for name, m in modes:
            # Aici rezolvăm problema: aplicăm filtrul pe miniatură înainte de afișare
            proc_thumb = self.apply_math(thumb_base.copy(), m)
            photo = ImageTk.PhotoImage(proc_thumb)
            self.previews[name] = photo # Garbage collection protection
            
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
            img_r = self.display_img.copy()
            img_r.thumbnail((w, h))
            self.tk_main = ImageTk.PhotoImage(img_r)
            self.canvas.delete("all")
            self.canvas.create_image(w//2, h//2, image=self.tk_main)

    def confirm_save(self):
        if self.display_img: self.original_img = self.display_img.copy(); messagebox.showinfo("Succes", "Imagine salvată!")

    def cancel_edits(self):
        self.reset_params(); self.active_mode = "none"
        self.display_img = self.original_img.copy(); self.render_main()
    
    def export_to_disk(self):
        if not self.display_img:
            messagebox.showwarning("Export", "Nu există nicio imagine procesată pentru salvare!")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".bmp",
                                                  filetypes=[("Bitmap", "*.bmp"), ("JPEG", "*.jpg"), ("PNG", "*.png")],
                                                  title="Salvează imaginea pe disk")
        if file_path:
            try:
                self.display_img.save(file_path)
                messagebox.showinfo("Export", f"Imaginea a fost salvată cu succes în:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Eroare", f"Nu s-a putut salva imaginea: {e}")

    def apply_math(self, img, mode):
        m = mode.lower()
        w, h = img.size
        pixels = img.load(); res = Image.new("RGB", (w, h)); new_pix = res.load()
        v = {k: var.get() for k, var in self.vars.items()}
        b_f, c_f = v["bright"], v["contrast"]

        for x in range(w):
            for y in range(h):
                r, g, b = pixels[x, y]
                
                if m == "gray(1)": 
                    val = int(((r+g+b)//3) * v["gray_gain"]); nr, ng, nb = val, val, val
                elif m == "gray(2)": 
                    val = int((0.299*r+0.587*g+0.114*b)*v["gray_gain"]); nr, ng, nb = val, val, val
                elif m == "gray(3)": 
                    val = int(((max(r,g,b)+min(r,g,b))//2)*v["gray_gain"]); nr, ng, nb = val, val, val
                elif m == "binarize":
                    # Mai întâi transformăm pixelul în intensitate gri (metoda Luminozitate)
                    gray = int(0.299*r + 0.587*g + 0.114*b)
                    
                    # Aplicăm pragul (folosim .get() pentru valoarea slider-ului)
                    prag = v["thresh"]
                    
                    if gray > prag:
                        nr = ng = nb = 255 # Alb
                    else:
                        nr = ng = nb = 0   # Negru
                        
                elif m == "cmyk":
                    rf, gf, bf_norm = r/255.0, g/255.0, b/255.0
                    k_final = (1.0-max(rf, gf, bf_norm)) * v["k"]
                    if k_final < 1.0:
                        nr = int(((1.0-rf-k_final)/(1.0-k_final))*255 * v["c"])
                        ng = int(((1.0-gf-k_final)/(1.0-k_final))*255 * v["m"])
                        nb = int(((1.0-bf_norm-k_final)/(1.0-k_final))*255 * v["y_c"])
                    else: nr = ng = nb = 0
                
                elif m == "negative": nr, ng, nb = int((255-r)*v["r"]), int((255-g)*v["g"]), int((255-b)*v["b"])
                
                elif m == "yuv":
                    y_v = int((0.3*r + 0.6*g + 0.1*b) * v["y_luma"])
                    u = int((0.74*(r - y_v) + 0.27*(b - y_v)) * v["u"]) + 128
                    v_v = int((0.48*(r - y_v) + 0.41*(b - y_v)) * v["v"]) + 128
                    nr, ng, nb = y_v, u, v_v
                
                elif m == "ycbcr" or m == "rgb_back":
                    y_v = int((0.299*r + 0.587*g + 0.114*b) * v["y_luma"])
                    cb = int((-0.1687*r - 0.3313*g + 0.5*b) * v["cb"]) + 128
                    cr = int((0.5*r - 0.4187*g - 0.0813*b) * v["cr"]) + 128
                    if m == "ycbcr": nr, ng, nb = y_v, cb, cr
                    else:
                        nr = int((y_v + 1.402 * (cr - 128)) * v["r"])
                        ng = int((y_v - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)) * v["g"])
                        nb = int((y_v + 1.772 * (cb - 128)) * v["b"])
                
                else: nr, ng, nb = int(r * v["r"]), int(g * v["g"]), int(b * v["b"])

                # Aplicație FINALĂ: Brightness & Contrast
                nr = int(nr * b_f * c_f); ng = int(ng * b_f * c_f); nb = int(nb * b_f * c_f)
                new_pix[x, y] = (max(0, min(255, nr)), max(0, min(255, ng)), max(0, min(255, nb)))
        return res

if __name__ == "__main__":
    root = tk.Tk(); app = AdvancedBitmapEditor(root); root.mainloop()