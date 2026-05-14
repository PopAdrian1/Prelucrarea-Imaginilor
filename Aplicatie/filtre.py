from PIL import Image

def filter_gray_1(r, g, b, v):
    val = int((0.299 * r + 0.587 * g + 0.114 * b) * v["gray_gain"])
    return val, val, val

def filter_gray_2(r, g, b, v):
    val=int((0.299*r+0.587*g+0.114*b)*v["gray_gain"]);
    return val, val, val

def filter_gray_3(r, g, b, v):
    val=int(((max(r,g,b)+min(r,g,b))//2)*v["gray_gain"]);
    return val, val, val

def binarizare(r, g, b, v):
    val=255 if int(0.299*r+0.587*g+0.114*b)>v["thresh"] else 0
    return val, val, val

def filtru_cmyk(r, g, b, v):
    rf,gf,bf_n=r/255,g/255,b/255
    k_f=(1-max(rf,gf,bf_n))*v["k"]

    nr=ng=nb=0

    if k_f<1:
        nr=int(((1-rf-k_f)/(1-k_f))*255*v["c"])
        ng=int(((1-gf-k_f)/(1-k_f))*255*v["m"])
        nb=int(((1-bf_n-k_f)/(1-k_f))*255*v["y_c"])
    
    return nr, ng, nb

def filtru_negativ(r, g, b, v):
    nr,ng,nb=int((255-r)*v["r"]),int((255-g)*v["g"]),int((255-b)*v["b"])
    return nr, ng, nb

def filtru_yuv(r, g, b, v):

    y_v=int((0.3*r+0.6*g+0.1*b)*v["y_luma"])
    nr=y_v
    ng=int((0.74*(r-y_v)+0.27*(b-y_v))*v["u"])+128
    nb=int((0.48*(r-y_v)+0.41*(b-y_v))*v["v"])+128

    return nr, ng, nb

def filtru_ycbcr(r, g, b, v):
    y_v=int((0.299*r+0.587*g+0.114*b)*v["y_luma"])
    cb=int((-0.1687*r-0.3313*g+0.5*b)*v["cb"])+128
    cr=int((0.5*r-0.4187*g-0.0813*b)*v["cr"])+128

    return y_v, cb, cr

def rgb_back(r, g, b, v):

    y_v, cb, cr = filtru_ycbcr(r, g, b, v)
    nr=int((y_v+1.402*(cr-128))*v["r"])
    ng=int((y_v-0.34414*(cb-128)-0.71414*(cr-128))*v["g"])
    nb=int((y_v+1.772*(cb-128))*v["b"])
    return nr, ng, nb

def filtru_hsv(r, g, b, v):
    
    rf,gf,bf_n=r/255,g/255,b/255
    M,mn=max(rf,gf,bf_n),min(rf,gf,bf_n); C=M-mn; V=M
    S=(C/V) if V else 0
    if C:
        if M==rf:   H=60*(((gf-bf_n)/C)%6)
        elif M==gf: H=60*((bf_n-rf)/C+2)
        else:       H=60*((rf-gf)/C+4)
    else: H=0
    nr,ng,nb=(int((H/360)*255*v["h"]), int(S*255*v["s"]), int(V*255*v["v_hsv"]))

    return nr, ng, nb

def apply_kernel_3x3(img, kernel):
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
            dst_px[i, j] = (
                max(0, min(255, int(sumr))),
                max(0, min(255, int(sumg))),
                max(0, min(255, int(sumb)))
            )
    return dst


def filtru_mediere(img):
    kernel = [[1/9]*3 for _ in range(3)]
    return apply_kernel_3x3(img, kernel)


def filtru_median(img):
    src = img.convert("RGB")
    w, h = src.size
    dst = Image.new("RGB", (w, h), (0, 0, 0))
    src_px = src.load()
    dst_px = dst.load()
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            sir = []
            for m in range(-1, 2):
                for n in range(-1, 2):
                    r, g, b = src_px[i + m, j + n]
                    sir.append(r)
            changed = True
            while changed:
                changed = False
                for idx in range(len(sir) - 1):
                    if sir[idx] > sir[idx + 1]:
                        sir[idx], sir[idx + 1] = sir[idx + 1], sir[idx]
                        changed = True
            med = sir[4]
            dst_px[i, j] = (med, med, med)
    return dst


def filtru_minim(img):
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
    kernel = [
        [0,    -1/4,  0   ],
        [-1/4,  1,   -1/4 ],
        [0,    -1/4,  0   ]
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
            nr = max(0, min(255, int(ro + alpha * sumr)))
            ng = max(0, min(255, int(go + alpha * sumg)))
            nb = max(0, min(255, int(bo + alpha * sumb)))
            dst_px[i, j] = (nr, ng, nb)
    return dst

filters_map = {
    "gray(1)":   filter_gray_1,
    "gray(2)":   filter_gray_2,
    "gray(3)":   filter_gray_3,
    "binarize":  binarizare,
    "cmyk":      filtru_cmyk,
    "negative":  filtru_negativ,
    "yuv":       filtru_yuv,
    "ycbcr":     filtru_ycbcr,
    "rgb_back":  rgb_back,
    "hsv":       filtru_hsv,
    
}

filters_map_img = {
    "mediere":    lambda img, v: filtru_mediere(img),
    "median":     lambda img, v: filtru_median(img),
    "minim":      lambda img, v: filtru_minim(img),
    "maxim":      lambda img, v: filtru_maxim(img),
    "accentuare": lambda img, v: filtru_accentuare(img, v["alpha"]),
}