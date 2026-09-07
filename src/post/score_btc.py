#!/usr/bin/env python3
"""Chấm ảnh render theo ĐÚNG bộ chấm của Ban Tổ Chức.

    Score = 0.4·(1 − LPIPS) + 0.3·SSIM + 0.3·clamp(PSNR/psnr_max, 0, 1)     # thang [0,1]
            × 100  →  thang bảng xếp hạng

Cấu hình đã được XÁC NHẬN trên hai model độc lập (24/08/2026) bằng cách tái tạo hai
bài nộp vòng chính thức rồi so với điểm BTC thật:

    Xác nhận trên hai bài nộp có điểm chính thức: lệch −0,016 và −0,142 điểm.

Giả thuyết bị loại (LPIPS thang ×100):
    LPIPS vgg  normalize=True    +5,78        SSIM skimage 7×7 uniform   −3,26
    LPIPS alex normalize=False   −6,16        SSIM skimage gaussian      −0,11
    LPIPS alex normalize=True    +9,43        SSIM 11×11 gaussian (dùng) −0,03

⚠ `normalize=False` với ảnh [0,1] nghĩa là LPIPS TƯỞNG ảnh đã ở thang [-1,1]. Trông như
  bug nhưng ĐÚNG là thứ BTC chạy — cố ý giữ, đừng "sửa". Vì thế quy ước chấm để CỨNG
  trong file này, không expose thành cờ — đổi một hằng số là đổi thước.

⚠ Chưa tách được: 11×11 gaussian của repo vs skimage(gaussian_weights=True) — cách nhau
  0,023 điểm, dưới sàn nhiễu tái tạo (0,016–0,142). Chỉ chốt được HỌ Gaussian.

Dùng:
    python3 score_btc.py --pred <thư mục render>              # GT tự dò theo kích thước
    python3 score_btc.py --pred <render> --gt <GT> --csv per_image.csv
    python3 score_btc.py --selftest                           # kiểm chính bộ chấm
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PSNR_MAX = 50.0                  # đề bài giấu; giải ngược từ số BTC → 49,995
LPIPS_NET = "vgg"
LPIPS_NORMALIZE = False
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # contest/

# Where to look for ground truth when --gt is not given.
#   1. $VT_GT_DIR            — set by config/recipe.env
#   2. $VT_SCENE/test/images — the scene currently being worked on
# Nothing is hard-coded: on the organiser's machine only these two apply.
def _gt_registry():
    cands = []
    for env in ("VT_GT_DIR", "GT_TEST"):
        if os.environ.get(env):
            cands.append(os.environ[env])
    if os.environ.get("VT_SCENE"):
        cands.append(os.path.join(os.environ["VT_SCENE"], "test", "images"))
    return cands


_win = None


def ssim_torch(a, b):
    """SSIM cửa sổ Gaussian 11×11 σ=1.5 — bản của repo, đã kiểm bit-exact."""
    global _win
    if _win is None or _win.device != a.device:
        g = torch.exp(-(torch.arange(11, dtype=torch.float32) - 5) ** 2 / (2 * 1.5 ** 2))
        g = g / g.sum()
        w = (g[:, None] @ g[None, :])
        _win = w.expand(3, 1, 11, 11).contiguous().to(a.device)
    w, ch = _win, a.shape[1]
    mu1, mu2 = F.conv2d(a, w, padding=5, groups=ch), F.conv2d(b, w, padding=5, groups=ch)
    m1s, m2s, m12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(a * a, w, padding=5, groups=ch) - m1s
    s2 = F.conv2d(b * b, w, padding=5, groups=ch) - m2s
    s12 = F.conv2d(a * b, w, padding=5, groups=ch) - m12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * m12 + C1) * (2 * s12 + C2))
            / ((m1s + m2s + C1) * (s1 + s2 + C2))).mean()


def lpips_tiled(fn, pr, gt, tile, overlap=192):
    """LPIPS theo ô chồng mép — dự phòng khi VRAM không đủ cho ảnh nguyên."""
    _, _, H, W = pr.shape
    if H <= tile and W <= tile:
        return fn(pr, gt, normalize=LPIPS_NORMALIZE).item()
    step = max(tile - overlap, 1)
    ys = sorted({min(y, max(H - tile, 0)) for y in range(0, H, step)} | {max(H - tile, 0)})
    xs = sorted({min(x, max(W - tile, 0)) for x in range(0, W, step)} | {max(W - tile, 0)})
    vals = [fn(pr[..., y:y + tile, x:x + tile], gt[..., y:y + tile, x:x + tile],
               normalize=LPIPS_NORMALIZE).item() for y in ys for x in xs]
    return float(np.mean(vals))


def load(path, device):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(device)


def img_size(path):
    with Image.open(path) as im:
        return im.size


def resolve_gt(pred_dir, explicit):
    """Trả về thư mục GT. Nếu không chỉ định: dò registry, chọn theo KÍCH THƯỚC render."""
    if explicit:
        return os.path.expanduser(explicit)
    env = os.environ.get("BTC_GT")
    if env:
        return os.path.expanduser(env)
    preds = sorted(f for f in os.listdir(pred_dir)
                   if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg"))
    if not preds:
        sys.exit(f"[LỖI] {pred_dir} không có ảnh nào")
    want = img_size(os.path.join(pred_dir, preds[0]))
    found = []
    for cand in _gt_registry():
        d = os.path.expanduser(cand)
        if not os.path.isdir(d):
            continue
        fs = sorted(glob.glob(os.path.join(glob.escape(d), "*")))
        fs = [f for f in fs if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")]
        if not fs:
            continue
        sz = img_size(fs[0])
        found.append((d, sz, len(fs)))
        if sz == want:
            print(f"[gt] tự dò: {d}  ({len(fs)} ảnh, {sz[0]}x{sz[1]})")
            return d
    print(f"[LỖI] không thư mục GT nào khớp kích thước render {want[0]}x{want[1]}.")
    for d, sz, n in found:
        print(f"       thấy: {d}  ({n} ảnh, {sz[0]}x{sz[1]})")
    if not found:
        print("       không thấy thư mục GT nào trong registry:")
        for c in _gt_registry():
            print(f"         {c}")
    sys.exit("       → truyền tay bằng --gt <thư mục>")


def score_dir(pred_dir, gt_dir, device, lp, tile, strict=True):
    """Chấm một scene. Trả về (agg, rows, cảnh_báo)."""
    gt_by = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir)
             if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")}
    pr_by = {os.path.splitext(f)[0]: f for f in os.listdir(pred_dir)
             if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")}
    missing, extra = sorted(set(gt_by) - set(pr_by)), sorted(set(pr_by) - set(gt_by))
    if strict and (missing or extra):
        if missing:
            print(f"[LỖI] THIẾU {len(missing)} ảnh so với GT — đề bài BTC: thiếu ảnh ở bất kỳ "
                  f"pose nào cũng ảnh hưởng kết quả. Không chấm phần giao.")
            for n in missing[:10]:
                print(f"        thiếu: {n}")
            if len(missing) > 10:
                print(f"        ... và {len(missing) - 10} ảnh nữa")
        if extra:
            print(f"[LỖI] THỪA {len(extra)} ảnh không có trong GT: {extra[:5]}")
        sys.exit("       → sửa thư mục render, hoặc --no_strict nếu CỐ Ý chấm phần giao")

    names = sorted(set(gt_by) & set(pr_by))
    if not names:
        sys.exit("[LỖI] không tên ảnh nào trùng giữa render và GT")
    rows, warn = [], []
    for i, n in enumerate(names):
        p = load(os.path.join(pred_dir, pr_by[n]), device)
        g = load(os.path.join(gt_dir, gt_by[n]), device)
        if p.shape != g.shape:
            sys.exit(f"[LỖI] lệch kích thước ở {n}: render {tuple(p.shape[2:])[::-1]} "
                     f"vs GT {tuple(g.shape[2:])[::-1]}. KHÔNG tự resize (sẽ làm sai điểm).")
        mse = torch.mean((g - p) ** 2).item()
        try:
            lv = lpips_tiled(lp, p, g, tile) if tile else lp(p, g, normalize=LPIPS_NORMALIZE).item()
        except torch.cuda.OutOfMemoryError:
            sys.exit(f"[LỖI] hết VRAM khi chạy LPIPS ở {n} ({tuple(p.shape[2:])[::-1]}).\n"
                     f"       → chạy lại với --lpips_tile 512")
        rows.append(dict(name=n, psnr=10 * math.log10(1.0 / max(mse, 1e-12)),
                         ssim=ssim_torch(p, g).item(), lpips=lv))
        del p, g
        if (i + 1) % 20 == 0:
            print(f"[score] {i + 1}/{len(names)}", flush=True)

    hot = [r["name"] for r in rows if r["psnr"] > PSNR_MAX]
    if hot:
        warn.append(f"{len(hot)} ảnh có PSNR > {PSNR_MAX:g} → clamp đang cắn, "
                    f"điểm không còn tuyến tính theo PSNR (vd {hot[0]})")
    if np.mean([r["lpips"] for r in rows]) < 1e-4:
        warn.append("LPIPS ≈ 0 — nghi --pred đang trỏ vào chính GT")
    agg = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "lpips")}
    agg["n"] = len(rows)
    agg["score"] = (0.4 * (1 - agg["lpips"]) + 0.3 * agg["ssim"]
                    + 0.3 * min(max(agg["psnr"] / PSNR_MAX, 0.0), 1.0))
    agg["score100"] = agg["score"] * 100
    return agg, rows, warn


def report(tag, agg, warn):
    print(f"\n[score_btc] {tag}")
    print(f"  n={agg['n']}  |  LPIPS {LPIPS_NET} normalize={LPIPS_NORMALIZE}  |  psnr_max={PSNR_MAX:g}")
    print(f"\n  PSNR   {agg['psnr']:10.4f}")
    print(f"  SSIM   {agg['ssim']:10.6f}")
    print(f"  LPIPS  {agg['lpips']:10.6f}")
    print(f"\n  SCORE  {agg['score']:10.6f}   →   {agg['score100']:.4f}   (thang bảng xếp hạng)")
    for w in warn:
        print(f"\n  ⚠ {w}")


def make_lpips(device):
    import lpips
    m = lpips.LPIPS(net=LPIPS_NET).to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def selftest(device, tile):
    """Cửa 1: tín hiệu đã biết. Cửa 2: tái lập điểm của hai bản tái tạo đã neo vào BTC."""
    import tempfile
    lp = make_lpips(device)
    ok = True

    print("=" * 70 + "\nCỬA 1 — tín hiệu đã biết\n" + "=" * 70)
    with tempfile.TemporaryDirectory() as td:
        g, p = os.path.join(td, "gt"), os.path.join(td, "pred")
        os.makedirs(g); os.makedirs(p)
        # Ảnh CÓ CẤU TRÚC (gradient + vân sin + khối), KHÔNG dùng nhiễu trắng: nhiễu
        # trắng không nén được nên JPEG q98 cho PSNR ~12dB, làm hỏng phép thử.
        yy, xx = np.mgrid[0:256, 0:384].astype(np.float32)
        for i in range(3):
            base = (0.5 + 0.25 * np.sin(xx / (7 + 4 * i)) * np.cos(yy / (11 + 3 * i))
                    + 0.20 * (xx / 384))
            arr = np.stack([base, np.roll(base, 40, 1), np.roll(base, 80, 0)], -1)
            arr[60 + 20 * i:130 + 20 * i, 90:220] = 0.85
            im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
            im.save(os.path.join(g, f"i{i}.png"))
            im.save(os.path.join(p, f"i{i}.png"))
        a, _, _ = score_dir(p, g, device, lp, tile)
        good = abs(a["score100"] - 100.0) < 1e-3
        ok &= good
        print(f"  pred ≡ gt        → {a['score100']:9.4f}   (phải 100.0000)   "
              f"{'ĐẬU' if good else 'TRƯỢT'}")
        for i in range(3):
            Image.open(os.path.join(g, f"i{i}.png")).save(
                os.path.join(p, f"i{i}.png".replace(".png", ".jpg")), quality=98)
            os.remove(os.path.join(p, f"i{i}.png"))
        a2, _, _ = score_dir(p, g, device, lp, tile)
        good2 = a2["score100"] < a["score100"] and a2["psnr"] > 40
        ok &= good2
        print(f"  JPEG q98         → {a2['score100']:9.4f}   PSNR {a2['psnr']:.2f} dB   "
              f"(phải thấp hơn 100 và PSNR>40)   {'ĐẬU' if good2 else 'TRƯỢT'}")

    print("\n" + "=" * 70 + "\nCỬA 2 — tái lập bản đã neo vào điểm BTC thật\n" + "=" * 70)
    # Optional regression anchors: renders whose organiser score you already know.
    # Format (config/score_anchors.json, not shipped):
    #   [{"name": "...", "dir": "...", "expect": 47.5466, "official": 47.5627}, ...]
    _anchors = os.path.join(_ROOT, "config", "score_anchors.json")
    cases = []
    if os.path.isfile(_anchors):
        with open(_anchors) as _f:
            cases = [(c["name"], c["dir"], float(c["expect"]), float(c.get("official", 0.0)))
                     for c in json.load(_f)]
    ran = False
    for name, rd, expect, btc in cases:
        rd = os.path.expanduser(rd)
        if not os.path.isdir(rd):
            print(f"  {name:12} BỎ QUA — không có {rd}")
            continue
        ran = True
        gt = resolve_gt(rd, None)
        a, _, _ = score_dir(rd, gt, device, lp, tile)
        good = abs(a["score100"] - expect) < 0.01
        ok &= good
        print(f"  {name:12} → {a['score100']:9.4f}   (phải {expect:.4f} ± 0.01"
              f" | BTC thật {btc:.4f})   {'ĐẬU' if good else 'TRƯỢT'}")
    if not ran:
        print("  (không có config/score_anchors.json — bỏ qua cửa neo, cửa 1 vẫn đủ để bắt lệch thước)")
    print("\n" + ("SELFTEST ĐẬU" if ok else "SELFTEST TRƯỢT"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description="Chấm render theo bộ chấm BTC (đã xác nhận trên 2 model).")
    ap.add_argument("--pred", help="thư mục ảnh render")
    ap.add_argument("--gt", default="", help="thư mục GT (bỏ trống = tự dò theo kích thước)")
    ap.add_argument("--csv", default="", help="xuất điểm từng ảnh")
    ap.add_argument("--json", default="", help="xuất tóm tắt dạng JSON")
    ap.add_argument("--multi_scene", action="store_true",
                    help="--pred/--gt chứa thư mục con theo scene; lấy trung bình QUA SCENE "
                         "(luật BTC). Bắt buộc phải truyền --gt.")
    ap.add_argument("--lpips_tile", type=int, default=0,
                    help="0 = ảnh nguyên (chuẩn). Đặt 512 nếu GPU nhỏ — xấp xỉ ~1e-3.")
    ap.add_argument("--no_strict", action="store_true",
                    help="cho phép chấm phần giao khi thiếu/thừa ảnh (MẶC ĐỊNH LÀ DỪNG)")
    ap.add_argument("--selftest", action="store_true", help="tự kiểm bộ chấm rồi thoát")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.selftest:
        sys.exit(selftest(device, a.lpips_tile))
    if not a.pred:
        ap.error("thiếu --pred (hoặc dùng --selftest)")
    pred = os.path.expanduser(a.pred)
    if not os.path.isdir(pred):
        sys.exit(f"[LỖI] không có thư mục {pred}")
    lp = make_lpips(device)
    strict = not a.no_strict

    if a.multi_scene:
        if not a.gt:
            sys.exit("[LỖI] --multi_scene bắt buộc phải có --gt (không tự dò được nhiều scene)")
        gtroot = os.path.expanduser(a.gt)
        scenes = sorted(d for d in os.listdir(pred)
                        if os.path.isdir(os.path.join(pred, d))
                        and os.path.isdir(os.path.join(gtroot, d)))
        if not scenes:
            sys.exit(f"[LỖI] không scene nào khớp giữa {pred} và {gtroot}")
        per, allrows = [], []
        for s in scenes:
            agg, rows, warn = score_dir(os.path.join(pred, s), os.path.join(gtroot, s),
                                        device, lp, a.lpips_tile, strict)
            report(f"scene {s}", agg, warn)
            agg["scene"] = s
            per.append(agg)
            for r in rows:
                r["scene"] = s
            allrows += rows
        final = float(np.mean([p["score100"] for p in per]))
        print("\n" + "=" * 70)
        print(f"ĐIỂM CUỐI = trung bình {len(per)} scene = {final:.4f}")
        print("=" * 70)
        out = {"scenes": per, "score100": final}
        rows = allrows
    else:
        gt = resolve_gt(pred, a.gt)
        if not os.path.isdir(gt):
            sys.exit(f"[LỖI] không có thư mục GT {gt}")
        agg, rows, warn = score_dir(pred, gt, device, lp, a.lpips_tile, strict)
        report(f"{pred}  vs  {gt}", agg, warn)
        out = agg

    if a.csv:
        keys = ["scene", "name", "psnr", "ssim", "lpips"] if a.multi_scene \
            else ["name", "psnr", "ssim", "lpips"]
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[score_btc] điểm từng ảnh -> {a.csv}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[score_btc] tóm tắt -> {a.json}")


if __name__ == "__main__":
    main()
