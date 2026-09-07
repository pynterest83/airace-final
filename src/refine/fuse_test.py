#!/usr/bin/env python3
"""Warp-fusion tham chiếu (IBR) trên holdout — Đòn 1, stage 0/1.

Ý tưởng: test/holdout view nằm xen kẽ trong cùng flight với train view
(overlap 86-90%), nên pixel THẬT sắc nét của ảnh train láng giềng có thể
warp sang target qua depth render từ model. Chỉ nhận pixel qua 4 cổng
(occlusion / đồng thuận chéo nguồn / đồng thuận LF với render / warp sắc
hơn render), và chỉ ghép BĂNG TẦN CAO (LF luôn của render → chặn trần
thiệt hại PSNR).

Baseline = đúng file trong <result_dir>/holdout_renders/ (đã qua pipeline
eval gốc, kể cả bilagrid nếu có) → Δ đo được so sánh 1-1 với SCORE_v của
run. Ckpt chỉ dùng để render DEPTH (bilagrid không đụng hình học).

Chạy trên VM:
  cd /srv/contest-workspace/t1-vair && source env_t1vair.sh
  python fuse_ibr.py --result_dir results/p2f4_mono05 --limit 5          # smoke
  python fuse_ibr.py --result_dir results/p2f4_mono05 --sweep 1          # stage 1
  python fuse_ibr.py --result_dir results/p2f4_mono05_ft --sweep 1       # nền ft

Chỉ in aggregate + per-image số; ảnh fused (nếu --save_fused) ở lại VM.
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa
from dataset import SceneData  # noqa: E402


# ---------------------------------------------------------------- metrics
def _gaussian_window(size=11, sigma=1.5, channels=3, device="cuda"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    w = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return w.expand(channels, 1, size, size).contiguous()


_ssim_win = None


def ssim_torch(img1, img2):  # NCHW [0,1] — copy nguyên văn trainer.py
    global _ssim_win
    if _ssim_win is None or _ssim_win.device != img1.device:
        _ssim_win = _gaussian_window(device=img1.device)
    w, ch = _ssim_win, img1.shape[1]
    mu1 = F.conv2d(img1, w, padding=5, groups=ch)
    mu2 = F.conv2d(img2, w, padding=5, groups=ch)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(img1 * img1, w, padding=5, groups=ch) - mu1_sq
    s2 = F.conv2d(img2 * img2, w, padding=5, groups=ch) - mu2_sq
    s12 = F.conv2d(img1 * img2, w, padding=5, groups=ch) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return m.mean()


def score_v(psnr, ssim, lpv):
    return 40.0 - 0.4 * 100 * lpv + 0.3 * 100 * ssim + 0.6 * psnr


def metrics(pred_nchw, gt_nchw, lp_vgg):
    mse = torch.mean((gt_nchw - pred_nchw) ** 2).item()
    psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
    ssim = ssim_torch(pred_nchw, gt_nchw).item()
    lpv = lp_vgg(pred_nchw, gt_nchw, normalize=False).item()
    return dict(psnr=psnr, ssim=ssim, lpips_vgg=lpv,
                score_v=score_v(psnr, ssim, lpv))


# ---------------------------------------------------------------- model/depth
def load_splats(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    s = ckpt["splats"]
    sh_degree = int(ckpt.get("cfg", {}).get("sh_degree", 3))
    splats = dict(
        means=s["means"].float().to(device),
        quats=s["quats"].float().to(device),
        scales=torch.exp(s["scales"].float()).to(device),
        opacities=torch.sigmoid(s["opacities"].float()).to(device),
        colors=torch.cat([s["sh0"], s["shN"]], 1).float().to(device),
    )
    return splats, sh_degree


@torch.no_grad()
def render_depth(splats, sh_degree, w2c, K, W, H, device):
    """-> depth (H,W), alpha (H,W). ED = expected z-depth của gsplat."""
    from gsplat.rendering import rasterization
    viewmat = torch.from_numpy(w2c).float().to(device)[None]
    Kt = torch.from_numpy(K).float().to(device)[None]
    renders, alphas, _ = rasterization(
        means=splats["means"], quats=splats["quats"], scales=splats["scales"],
        opacities=splats["opacities"], colors=splats["colors"],
        viewmats=viewmat, Ks=Kt, width=W, height=H, sh_degree=sh_degree,
        render_mode="RGB+ED", rasterize_mode="classic",
        near_plane=0.01, far_plane=1e10, packed=False)
    return renders[0, ..., 3], alphas[0, ..., 0]


def render_depth_moments(splats, w2c, K, W, H, device):
    """01/09: depth KHÔNG chỉ là 1 số — trả về E[z], std[z] (bề rộng phân bố dọc tia), alpha.
    Cách làm chuẩn trong văn liệu uncertainty-GS: rasterize lần 2 với 'màu' = (z, z^2, 0)
    rồi chuẩn hoá theo alpha:  Var = E[z^2] - E[z]^2.
    std lớn = tia xuyên nhiều mặt (mép nhà) hoặc sương/floater → chỗ expected-depth thiên vị nặng."""
    from gsplat.rendering import rasterization
    viewmat = torch.from_numpy(w2c).float().to(device)[None]
    Kt = torch.from_numpy(K).float().to(device)[None]
    means = splats["means"]
    z = (means @ viewmat[0, :3, :3].T + viewmat[0, :3, 3])[:, 2]
    cols = torch.stack([z, z * z, torch.zeros_like(z)], -1)
    renders, alphas, _ = rasterization(
        means=means, quats=splats["quats"], scales=splats["scales"],
        opacities=splats["opacities"], colors=cols,
        viewmats=viewmat, Ks=Kt, width=W, height=H, sh_degree=None,
        render_mode="RGB", rasterize_mode="classic", near_plane=0.01, far_plane=1e10, packed=False)
    A = alphas[0, ..., 0].clamp_min(1e-6)
    Ez = renders[0, ..., 0] / A
    Ez2 = renders[0, ..., 1] / A
    return Ez, (Ez2 - Ez ** 2).clamp_min(0).sqrt(), alphas[0, ..., 0]


# ---------------------------------------------------------------- geometry
def select_sources(scene, m_t, depth_t, alpha_t, K_np, args):
    """Chọn K nguồn: lọc góc trục quang (gimbal swing 56°/frame!) + overlap
    frustum qua depth model, xếp theo baseline tăng dần."""
    w2c_t, c_t = m_t.w2c(), m_t.center()
    axes = scene.w2c[:, 2, :3]                      # cam-z trong world, (N,3)
    axis_t = w2c_t[2, :3]
    cosang = (axes @ axis_t) / (np.linalg.norm(axes, axis=1) * np.linalg.norm(axis_t) + 1e-9)
    ang_ok = cosang > math.cos(math.radians(args.angle_max))

    H, W = depth_t.shape
    vs = np.linspace(0, H - 1, 8).astype(int)
    us = np.linspace(0, W - 1, 10).astype(int)
    uu, vv = np.meshgrid(us, vs)
    d = depth_t.cpu().numpy()[vv, uu]
    a = alpha_t.cpu().numpy()[vv, uu]
    ok = a > 0.5
    if ok.sum() < 8:
        return []
    fx, fy, cx, cy = K_np[0, 0], K_np[1, 1], K_np[0, 2], K_np[1, 2]
    x = (uu[ok] - cx) / fx * d[ok]
    y = (vv[ok] - cy) / fy * d[ok]
    Xc = np.stack([x, y, d[ok]], 1)                  # (M,3) cam target
    c2w = np.linalg.inv(w2c_t)
    Xw = Xc @ c2w[:3, :3].T + c2w[:3, 3]

    R = scene.w2c[:, :3, :3]                         # (N,3,3)
    t = scene.w2c[:, :3, 3]                          # (N,3)
    Xs = np.einsum("nij,mj->nmi", R, Xw) + t[:, None, :]   # (N,M,3)
    z = Xs[..., 2]
    u = Xs[..., 0] / np.clip(z, 1e-6, None) * fx + cx
    v = Xs[..., 1] / np.clip(z, 1e-6, None) * fy + cy
    inb = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    frac = inb.mean(axis=1)

    base = np.linalg.norm(scene.centers - c_t, axis=1)
    cand = np.where(ang_ok & (frac >= args.overlap_min) & (base > 1e-6))[0]
    return list(cand[np.argsort(base[cand])][: args.K])


def gauss_blur(t_nchw, sigma):
    k = max(3, int(sigma * 4) | 1)
    xs = torch.arange(k, device=t_nchw.device, dtype=torch.float32) - k // 2
    g = torch.exp(-xs ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    ch = t_nchw.shape[1]
    t = F.conv2d(t_nchw, g.view(1, 1, 1, k).expand(ch, 1, 1, k), padding=(0, k // 2), groups=ch)
    return F.conv2d(t, g.view(1, 1, k, 1).expand(ch, 1, k, 1), padding=(k // 2, 0), groups=ch)


def warp_source(img_s, depth_s, m_t, m_s, depth_t, K_np, device, extra_shift=None):
    """Warp ảnh nguồn sang target qua depth_t. -> (W_s (1,3,H,W), z hợp lệ +
    occl info). extra_shift (H,W,2) đơn vị pixel cộng vào lưới sample."""
    H, W = depth_t.shape
    fx, fy, cx, cy = K_np[0, 0], K_np[1, 1], K_np[0, 2], K_np[1, 2]
    vv, uu = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    x = (uu - cx) / fx * depth_t
    y = (vv - cy) / fy * depth_t
    Xc = torch.stack([x, y, depth_t], -1).reshape(-1, 3)
    c2w = torch.from_numpy(np.linalg.inv(m_t.w2c())).float().to(device)
    Xw = Xc @ c2w[:3, :3].T + c2w[:3, 3]
    w2c_s = torch.from_numpy(m_s.w2c()).float().to(device)
    Xs = Xw @ w2c_s[:3, :3].T + w2c_s[:3, 3]
    z_s = Xs[:, 2].reshape(H, W)
    u_s = (Xs[:, 0] / Xs[:, 2].clamp(min=1e-6) * fx + cx).reshape(H, W)
    v_s = (Xs[:, 1] / Xs[:, 2].clamp(min=1e-6) * fy + cy).reshape(H, W)
    if extra_shift is not None:
        u_s = u_s + extra_shift[..., 0]
        v_s = v_s + extra_shift[..., 1]
    gx = 2 * u_s / (W - 1) - 1
    gy = 2 * v_s / (H - 1) - 1
    grid = torch.stack([gx, gy], -1)[None]
    W_s = F.grid_sample(img_s, grid, mode="bilinear", align_corners=True,
                        padding_mode="zeros")
    d_samp = F.grid_sample(depth_s[None, None], grid, mode="bilinear",
                           align_corners=True, padding_mode="zeros")[0, 0]
    inb = (z_s > 0.05) & (gx.abs() <= 1) & (gy.abs() <= 1)
    occ = ((z_s - d_samp).abs() / z_s.clamp(min=1e-6))
    return W_s, inb, occ


def tile_align(Wlf, Rlf, tile, max_shift, min_peak, sign):
    """Phase-correlation per-tile giữa LF(warp) và LF(render) -> trường shift
    (H,W,2) pixel. sign=±1 để tự kiểm chiều (smoke in frac_improved)."""
    H, W = Rlf.shape
    T = tile
    Hp, Wp = (H + T - 1) // T * T, (W + T - 1) // T * T
    pad = (0, Wp - W, 0, Hp - H)
    a = F.pad(Rlf[None, None], pad, mode="reflect")[0, 0]
    b = F.pad(Wlf[None, None], pad, mode="reflect")[0, 0]
    nH, nW = Hp // T, Wp // T
    at = a.reshape(nH, T, nW, T).permute(0, 2, 1, 3).reshape(-1, T, T)
    bt = b.reshape(nH, T, nW, T).permute(0, 2, 1, 3).reshape(-1, T, T)
    win = torch.hann_window(T, device=a.device)
    win2 = win[:, None] * win[None, :]
    at = (at - at.mean(dim=(-2, -1), keepdim=True)) * win2
    bt = (bt - bt.mean(dim=(-2, -1), keepdim=True)) * win2
    A = torch.fft.fft2(at)
    B = torch.fft.fft2(bt)
    Rc = A * B.conj()
    Rc = Rc / (Rc.abs() + 1e-8)
    r = torch.fft.ifft2(Rc).real
    r = torch.fft.fftshift(r, dim=(-2, -1))
    peak, idx = r.reshape(r.shape[0], -1).max(dim=1)
    dy = (idx // T).float() - T // 2
    dx = (idx % T).float() - T // 2
    bad = (peak < min_peak) | (dy.abs() > max_shift) | (dx.abs() > max_shift)
    dy[bad] = 0
    dx[bad] = 0
    field = torch.stack([dx, dy], -1).reshape(nH, nW, 2) * float(sign)
    field = F.avg_pool2d(F.pad(field.permute(2, 0, 1)[None], (1, 1, 1, 1),
                               mode="replicate"), 3, stride=1)[0].permute(1, 2, 0)
    full = F.interpolate(field.permute(2, 0, 1)[None], size=(H, W),
                         mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    return full


def tile_energy(x_hw, tile):
    """Năng lượng Laplacian trung bình per-tile, upsample về (H,W)."""
    lap_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                         dtype=torch.float32, device=x_hw.device).view(1, 1, 3, 3)
    lap = F.conv2d(x_hw[None, None], lap_k, padding=1) ** 2
    e = F.avg_pool2d(lap, tile, stride=tile, ceil_mode=True)
    return F.interpolate(e, size=x_hw.shape, mode="nearest")[0, 0]


# ---------------------------------------------------------------- selftest
def selftest():
    """Kiểm chiều shift của tile_align bằng ảnh tổng hợp dịch đã biết —
    không cần data. PASS = hiệu chỉnh (sign=-1) khôi phục đúng (dx,dy)."""
    torch.manual_seed(0)
    H, W, T = 512, 640, 128
    base = gauss_blur(torch.randn(1, 1, H, W), 3.0)[0, 0]
    ok_all = True
    for d in ((5, -3), (-4, 2)):
        b = torch.roll(base, shifts=(d[1], d[0]), dims=(0, 1))  # b(x)=a(x-d)
        field = tile_align(b, base, T, max_shift=8.0, min_peak=0.01, sign=-1)
        inner = field[T:-T, T:-T]
        dx, dy = inner[..., 0].mean().item(), inner[..., 1].mean().item()
        ok = abs(dx - d[0]) < 0.7 and abs(dy - d[1]) < 0.7
        ok_all &= ok
        print(f"SELFTEST d={d} đo=({dx:+.2f},{dy:+.2f}) "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    print("SELFTEST", "PASS" if ok_all else
          "FAIL — KHÔNG chạy fusion, kiểm lại chiều shift", flush=True)
    return ok_all



# ---------------------------------------------------------------- main (TEST poses, 26/08)
class _T:
    """adapter TestPose -> giao diện m_t (w2c(), center(), name) của fuse_ibr."""
    def __init__(self, tp): self.tp, self.name = tp, tp.image_name
    def w2c(self): return self.tp.w2c
    def center(self): return -self.tp.w2c[:3, :3].T @ self.tp.w2c[:3, 3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", required=True, help="thư mục có ckpt.pt (để render depth)")
    p.add_argument("--renders", required=True, help="thư mục render test (png)")
    p.add_argument("--scene_dir", required=True)
    p.add_argument("--out", required=True, help="thư mục ghi ảnh fused")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--angle_max", type=float, default=25.0)
    p.add_argument("--overlap_min", type=float, default=0.4)
    p.add_argument("--tau_occ", type=float, default=0.01)
    p.add_argument("--tau_c", type=float, default=0.03)
    p.add_argument("--tau_r", type=float, default=0.05)
    p.add_argument("--sharp_ratio", type=float, default=1.3)
    p.add_argument("--blur_sigma", type=float, default=2.0)
    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--max_shift", type=float, default=4.0)
    p.add_argument("--min_peak", type=float, default=0.03)
    p.add_argument("--shift_sign", type=int, default=-1)
    p.add_argument("--align", type=int, default=1)
    p.add_argument("--erode", type=int, default=2)
    p.add_argument("--feather", type=float, default=3.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--lf_from_warp", type=float, default=0.0,
                   help="0 = LF luôn của render (an toàn PSNR); a>0: LF = (1-a)*LF(render) + a*LF(warp) ở vùng M")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(); dev = args.device
    scene = SceneData(args.scene_dir, load_images=False, distorted=False, holdout_every=0)
    # 26/08: nạp ảnh nguồn theo nhu cầu (cache nhỏ) thay vì 404 ảnh 21 MP = 25 GB RAM (cgroup 256 GiB)
    _img_cache = {}
    def _src_img(si):
        if si not in _img_cache:
            if len(_img_cache) > 24: _img_cache.pop(next(iter(_img_cache)))
            _p = os.path.join(args.scene_dir, "train", "images", scene.train_metas[si].name)
            _img_cache[si] = cv2.cvtColor(cv2.imread(_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return _img_cache[si]
    K_np = scene.K.astype(np.float64); Hh, Wh = scene.height, scene.width
    splats, sh_degree = load_splats(os.path.join(args.result_dir, "ckpt.pt"), dev)
    print(f"[ckpt] n={splats['means'].shape[0]}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    tps = scene.test_poses[: args.limit] if args.limit else scene.test_poses
    depth_cache = {}; covs = []; nskip = 0
    for tp in tps:
        m_t = _T(tp)
        base = os.path.splitext(tp.image_name)[0]
        rp = [q for e in (".png", ".jpg", ".JPG") for q in [os.path.join(args.renders, base + e)] if os.path.exists(q)][0]
        r_u8 = cv2.imread(rp, cv2.IMREAD_COLOR)
        R_t = torch.from_numpy(cv2.cvtColor(r_u8, cv2.COLOR_BGR2RGB)).to(dev).float().div(255).permute(2, 0, 1)[None]
        depth_t, alpha_t = render_depth(splats, sh_degree, tp.w2c, K_np, Wh, Hh, dev)
        src_idx = select_sources(scene, m_t, depth_t, alpha_t, K_np, args)
        out = R_t
        if len(src_idx) >= 2:
            R_lf = gauss_blur(R_t, args.blur_sigma); R_gray_lf = R_lf.mean(dim=1)[0]
            warps = []
            for si in src_idx:
                if si not in depth_cache:
                    d_s, _ = render_depth(splats, sh_degree, scene.train_metas[si].w2c(), K_np, Wh, Hh, dev)
                    depth_cache[si] = d_s.cpu()
                d_s = depth_cache[si].to(dev)
                img_s = torch.from_numpy(_src_img(si)).to(dev).float().div(255).permute(2, 0, 1)[None]
                W0, inb, occ = warp_source(img_s, d_s, m_t, scene.train_metas[si], depth_t, K_np, dev)
                if args.align:
                    shift = tile_align(gauss_blur(W0, args.blur_sigma).mean(dim=1)[0], R_gray_lf, args.tile, args.max_shift, args.min_peak, args.shift_sign)
                    W1, inb1, occ1 = warp_source(img_s, d_s, m_t, scene.train_metas[si], depth_t, K_np, dev, extra_shift=shift)
                    e0 = F.avg_pool2d((gauss_blur(W0, args.blur_sigma) - R_lf).abs().mean(1, keepdim=True), args.tile, stride=args.tile, ceil_mode=True)
                    e1 = F.avg_pool2d((gauss_blur(W1, args.blur_sigma) - R_lf).abs().mean(1, keepdim=True), args.tile, stride=args.tile, ceil_mode=True)
                    better = F.interpolate((e1 < e0).float(), size=(Hh, Wh), mode="nearest")[0, 0]
                    W0 = W1 * better[None, None] + W0 * (1 - better[None, None])
                    inb = torch.where(better.bool(), inb1, inb); occ = torch.where(better.bool(), occ1, occ)
                warps.append((W0, inb, occ))
            Ws = torch.cat([w[0] for w in warps]); acc = torch.stack([w[1] & (w[2] < args.tau_occ) for w in warps])
            n_acc = acc.sum(0); accf = acc[:, None].float()
            Wmean = (Ws * accf).sum(0, keepdim=True) / (accf.sum(0, keepdim=True) + 1e-6)
            Wmean = torch.where((n_acc >= 1)[None, None], Wmean, R_t)
            Ws_lf = gauss_blur(Ws, args.blur_sigma); Wm_lf = gauss_blur(Wmean, args.blur_sigma)
            dev_lf = ((Ws_lf - Wm_lf).abs().mean(1) * acc.float()).sum(0) / n_acc.clamp(min=1)
            M = (n_acc >= 2) & (dev_lf < args.tau_c) & ((Wm_lf - R_lf).abs().mean(1)[0] < args.tau_r) \
                & (tile_energy(Wmean.mean(1)[0], args.tile) > args.sharp_ratio * tile_energy(R_t.mean(1)[0], args.tile)) & (alpha_t > 0.5)
            Mf = M.float()[None, None]
            if args.erode > 0: Mf = 1 - F.max_pool2d(1 - Mf, 2 * args.erode + 1, stride=1, padding=args.erode)
            Mf = gauss_blur(Mf, args.feather).clamp(0, 1)
            _a = float(args.lf_from_warp)
            fused_px = ((1 - _a) * R_lf + _a * Wm_lf) + (Wmean - Wm_lf)
            out = (R_t * (1 - Mf) + fused_px * Mf).clamp(0, 1)
            covs.append(M.float().mean().item())
        else:
            nskip += 1
        u8 = (out[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        cv2.imwrite(os.path.join(args.out, base + ".png"), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
        print(f"[fuse] {base[-8:]} src={len(src_idx)} cov={covs[-1] if covs and len(src_idx)>=2 else 0:.3f}", flush=True)
    print(f"FUSE_DONE n={len(tps)} skip={nskip} cov_mean={np.mean(covs) if covs else 0:.3f}", flush=True)


if __name__ == "__main__":
    main()
