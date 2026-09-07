"""score_holdout.py — 07/09: chấm thước BTC (PSNR / SSIM Gaussian / LPIPS-vgg normalize=False tile 1024) cho view HOLDOUT trong dump refiner
(khung khử méo: pred = <dump>/<view>/render.png hoặc <pred>/<view>.png, GT = <dump>/<view>/gt_und.png). CSV cùng cột với score_btc → msc đọc được.
  python score_holdout.py --dump D [--pred P] --out scores/x.csv [--names 0033,0050]"""
import argparse, os, sys, math, csv, numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa
from score_btc import ssim_torch  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--dump", required=True); ap.add_argument("--pred", default=""); ap.add_argument("--out", required=True); ap.add_argument("--names", default=""); a = ap.parse_args(); dev = "cuda"
import lpips; lp = lpips.LPIPS(net="vgg").to(dev).eval()
def rd(p): return torch.from_numpy(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)).to(dev).float().div(255).permute(2, 0, 1)[None]
def metr(P, G):
    H, W = G.shape[-2:]; mse = float(((P - G) ** 2).mean()); psnr = 10 * math.log10(1 / max(mse, 1e-12)); ss = float(ssim_torch(P, G)); ls = []
    with torch.no_grad():
        for y in range(0, H, 1024):
            for x in range(0, W, 1024): ls.append(float(lp(P[..., y:y + 1024, x:x + 1024] * 2 - 1, G[..., y:y + 1024, x:x + 1024] * 2 - 1)))
    return psnr, ss, float(np.mean(ls))
views = sorted(d for d in os.listdir(a.dump) if os.path.isfile(os.path.join(a.dump, d, "gt_und.png")))
if a.names: views = [v for v in views if any(f"_{n}_" in v for n in a.names.split(","))]
rows = []
for v in views:
    pp = os.path.join(a.pred, v + ".png") if a.pred else os.path.join(a.dump, v, "render.png")
    if not os.path.exists(pp): print("  thiếu", pp, flush=True); continue
    G = rd(os.path.join(a.dump, v, "gt_und.png")); P = rd(pp)
    if P.shape != G.shape: P = torch.nn.functional.interpolate(P, size=G.shape[-2:], mode="bilinear", align_corners=False)
    ps, ss, l = metr(P, G); rows.append((v, ps, ss, l))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["name", "psnr", "ssim", "lpips"]); [w.writerow(r) for r in rows]
sc = [100 * (0.4 * (1 - r[3]) + 0.3 * r[2] + 0.3 * min(r[1], 50) / 50) for r in rows]
print(f"[p1] {os.path.basename(a.out)}: n={len(rows)} SCORE {np.mean(sc):.4f} PSNR {np.mean([r[1] for r in rows]):.3f} SSIM {np.mean([r[2] for r in rows]):.4f} LPIPS {np.mean([r[3] for r in rows]):.4f}", flush=True)
