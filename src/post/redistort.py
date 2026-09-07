"""redistort.py — 05/09: đưa ảnh ở khung PINHOLE (undistort) về khung GT (méo) của scene: cv2.remap với scene.redistort_map(K_test, W, H), viền BORDER_REPLICATE
(không có canvas đệm nên góc ảnh thiếu ~|k1|·r² → thay bằng lặp biên). Dùng cho output refiner tính trong khung undistort.
  python redistort.py --scene_dir S_reba --in DIR --out DIR"""
import argparse, os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa
from dataset import SceneData  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--scene_dir", required=True); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True); ap.add_argument("--fallback", default="", help="thư mục render REDISTORT có đệm (thô) để lấp góc thiếu thay vì lặp biên"); ap.add_argument("--interp", default="cubic", help="cubic|lanczos|linear — y67: trên ảnh hoàn hảo lanczos hơn cubic, nhưng y68 B trên output refiner thật: lanczos 60,13 < cubic 60,19 → giữ cubic"); a = ap.parse_args()
INTERP = {"cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4, "linear": cv2.INTER_LINEAR}[a.interp]
sc = SceneData(a.scene_dir, load_images=False, distorted=False, holdout_every=0); os.makedirs(a.out, exist_ok=True); n = 0; maps = {}
for tp in sc.test_poses:
    stem = os.path.splitext(tp.image_name)[0]; src = [p for e in (".png", ".jpg", ".JPG") if os.path.exists(p := os.path.join(a.inp, stem + e))]
    if not src: continue
    key = (tuple(np.round(tp.K.ravel(), 3)), tp.width, tp.height)
    if key not in maps: maps[key] = sc.redistort_map(tp.K, tp.width, tp.height)
    mapx, mapy = maps[key]; im = cv2.imread(src[0]); out = cv2.remap(im, mapx, mapy, interpolation=INTERP, borderMode=cv2.BORDER_REPLICATE)
    if a.fallback:
        fb = [p for e in (".png", ".jpg", ".JPG") if os.path.exists(p := os.path.join(a.fallback, stem + e))]
        if fb:
            F = cv2.imread(fb[0]); inside = (mapx >= 1) & (mapx <= im.shape[1] - 2) & (mapy >= 1) & (mapy <= im.shape[0] - 2)
            m = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 3)[..., None]; out = (out.astype(np.float32) * m + F.astype(np.float32) * (1 - m)).round().clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(a.out, stem + ".png"), out); n += 1
print(f"[y39] redistort {n} ảnh ({a.interp}) -> {a.out} | dist {np.round(np.asarray(sc.dist), 5).tolist()} | góc thiếu ~{100 * abs(float(sc.dist[0])) * 0.77:.1f}% bán kính")
