"""Band-swap N1: HF (σ) 100 % từ member `--hf`, LF = trộn có trọng số các member `--lf a:w,b:w`.
  python bandswap.py --hf DIR_A --lf DIR_A:0.5,DIR_B:0.5 --out OUT [--sigma 4]"""
import argparse, os, glob, cv2, numpy as np
from multiprocessing import Pool
ap = argparse.ArgumentParser(); ap.add_argument("--hf", required=True); ap.add_argument("--lf", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--sigma", type=float, default=4.0); a = ap.parse_args()
lf = [(x.split(":")[0], float(x.split(":")[1])) for x in a.lf.split(",")]; sw = sum(w for _, w in lf)
os.makedirs(a.out, exist_ok=True)
def find(d, b):
    for e in (".png", ".jpg", ".JPG"):
        if os.path.exists(os.path.join(d, b + e)): return os.path.join(d, b + e)
def one(p):
    b = os.path.splitext(os.path.basename(p))[0]
    A = cv2.imread(p).astype(np.float32); Alf = cv2.GaussianBlur(A, (0, 0), a.sigma)
    L = sum(w * cv2.GaussianBlur(cv2.imread(find(d, b)).astype(np.float32), (0, 0), a.sigma) for d, w in lf) / sw
    cv2.imwrite(os.path.join(a.out, b + ".png"), np.clip(L + (A - Alf), 0, 255).round().astype(np.uint8))
with Pool(16) as pool: pool.map(one, sorted(glob.glob(os.path.join(a.hf, "*"))))
print("BANDSWAP_DONE", len(os.listdir(a.out)))
