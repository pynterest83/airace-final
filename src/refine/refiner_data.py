"""Sinh dữ liệu cho refiner có điều kiện láng giềng (26/08, IBGS/GADA distilled):
mỗi target: render R (từ ckpt), K warp W_i (ảnh train láng giềng warp qua depth render, đã tile-align),
mask M_i (in-bounds & occ<tau), GT (chỉ với holdout). Lưu PNG uint8 trong --dump/<name>/.
  --targets holdout --holdout_every 4 : target = view train bị giữ lại (có GT) → tập train refiner
  --targets test                       : target = pose test (không GT) → suy luận
"""
import argparse, time, os, sys, json
import numpy as np, torch, cv2, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa
# (path bootstrap handled by _paths)
from dataset import SceneData  # noqa
import fuse_test as FT  # noqa

def fwd_splat(img_s, d_s, w2c_s, w2c_t, K_np, H, W, dev):
    """Forward-warp ảnh nguồn sang target bằng DEPTH CỦA NGUỒN (đúng ở view train) + z-buffer.
    -> (3,H,W) màu, (H,W) mask 1 = có pixel nguồn rơi vào."""
    fx, fy, cx, cy = K_np[0, 0], K_np[1, 1], K_np[0, 2], K_np[1, 2]
    vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    Xc = torch.stack([(uu - cx) / fx * d_s, (vv - cy) / fy * d_s, d_s], -1).reshape(-1, 3)
    c2w_s = torch.from_numpy(np.linalg.inv(w2c_s)).float().to(dev); Xw = Xc @ c2w_s[:3, :3].T + c2w_s[:3, 3]
    Tt = torch.from_numpy(w2c_t).float().to(dev); Xt = Xw @ Tt[:3, :3].T + Tt[:3, 3]
    z = Xt[:, 2]; u = (Xt[:, 0] / z.clamp_min(1e-6) * fx + cx).round().long(); v = (Xt[:, 1] / z.clamp_min(1e-6) * fy + cy).round().long()
    ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (d_s.reshape(-1) > 1e-6)
    idx = (v * W + u)[ok]; zz = z[ok]; col = img_s[0].permute(1, 2, 0).reshape(-1, 3)[ok]
    zbuf = torch.full((H * W,), float("inf"), device=dev).scatter_reduce_(0, idx, zz, reduce="amin")
    sel = zz <= zbuf[idx] * 1.002
    out = torch.zeros(H * W, 3, device=dev); out[idx[sel]] = col[sel]
    m = torch.zeros(H * W, device=dev); m[idx[sel]] = 1.0
    out = out.reshape(H, W, 3).permute(2, 0, 1); m = m.reshape(H, W)
    # vá lỗ nhỏ (splat thưa do zoom): dilate 3x3 hai lần
    for _ in range(2):
        mm = F.max_pool2d(m[None, None], 3, 1, 1)[0, 0]; om = F.max_pool2d(out[None], 3, 1, 1)[0]
        out = torch.where(m[None] > 0, out, om); m = mm
    return out, m


_RAFT = None
def flow_align(W0, R_t, dev, fmax=24.0, scale=0.25):
    """30/08: căn warp W0 (1,3,H,W) về render R_t bằng RAFT ở 1/4 res: flow(render→warp) rồi lấy mẫu W0 tại p+flow.
    -> W1 (1,3,H,W), grid (1,H,W,2) để lấy mẫu inb/occ y hệt."""
    global _RAFT
    if _RAFT is None:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        _RAFT = raft_large(weights=Raft_Large_Weights.DEFAULT).to(dev).eval()
    H, W = R_t.shape[-2:]; h, w = int(H * scale) // 8 * 8, int(W * scale) // 8 * 8
    a = F.interpolate(R_t, size=(h, w), mode="area") * 2 - 1; b = F.interpolate(W0, size=(h, w), mode="area") * 2 - 1
    with torch.no_grad(): fl = _RAFT(a, b)[-1]                                  # (1,2,h,w): a(p) ~ b(p+fl)
    fl = F.interpolate(fl, size=(H, W), mode="bilinear", align_corners=False); fl[:, 0] *= W / w; fl[:, 1] *= H / h
    fl = fl.clamp(-fmax, fmax)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    gx = (xx + fl[0, 0]) / (W - 1) * 2 - 1; gy = (yy + fl[0, 1]) / (H - 1) * 2 - 1; grid = torch.stack([gx, gy], -1)[None]
    W1 = F.grid_sample(W0, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return W1, grid, fl

def flow_raft(A, B, dev, fmax=32.0, scale=0.25):
    """flow fl: A(p) ~ B(p+fl). Dùng cho cặp ẢNH THẬT (warp↔warp), không dính render."""
    global _RAFT
    if _RAFT is None:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        _RAFT = raft_large(weights=Raft_Large_Weights.DEFAULT).to(dev).eval()
    H, W = A.shape[-2:]; h, w = int(H * scale) // 8 * 8, int(W * scale) // 8 * 8
    aa = F.interpolate(A, size=(h, w), mode="area") * 2 - 1
    bb = F.interpolate(B, size=(h, w), mode="area") * 2 - 1
    with torch.no_grad(): fl = _RAFT(aa, bb)[-1]
    fl = F.interpolate(fl, size=(H, W), mode="bilinear", align_corners=False)
    fl[:, 0] *= W / w; fl[:, 1] *= H / h
    return fl.clamp(-fmax, fmax)


def aniso_smooth(v, w, guide, iters=40, lam=0.22, sig=0.04):
    """Làm trơn trường vô hướng v với TRỌNG SỐ w (thông tin Fisher |dk|^2) và TÔN TRỌNG BIÊN ảnh.
    Khuếch tán dị hướng (Perona-Malik / Anisotropic Huber-L1 flow, Werlberger 2009): hệ số dẫn
    c = exp(-|grad I|/sig) → Δz KHÔNG bị bôi qua mép nhà, đúng chỗ sai số depth lớn nhất.
    Khuếch tán ĐỒNG THỜI tử (v·w) và mẫu (w) = tích chập chuẩn hoá → vùng thiếu tin cậy tự mượn
    thông tin từ hàng xóm cùng bề mặt thay vì bị kéo về 0."""
    g = guide.mean(0) if guide.dim() == 3 else guide
    cx = torch.exp(-(g[:, 1:] - g[:, :-1]).abs() / sig)
    cy = torch.exp(-(g[1:] - g[:-1]).abs() / sig)
    a, b = v * w, w.clone()
    for _ in range(iters):
        for arr in (a, b):
            f = torch.zeros_like(arr)
            dx = (arr[:, 1:] - arr[:, :-1]) * cx
            f[:, :-1] += dx; f[:, 1:] -= dx
            dy = (arr[1:] - arr[:-1]) * cy
            f[:-1] += dy; f[1:] -= dy
            arr += lam * f
    return a / b.clamp_min(1e-8)


def dz_jacobian(m_t, m_s, depth_t, K_np, dev):
    """∂(u_s,v_s)/∂z tại từng pixel target — đúng phép chiếu của FT.warp_source.
    u_s = fx(A_x z + B_x)/(A_z z + B_z) + cx  ⇒  ∂u/∂z = fx(A_x B_z − A_z B_x)/(A_z z + B_z)²."""
    H, W = depth_t.shape
    fx, fy, cx, cy = K_np[0, 0], K_np[1, 1], K_np[0, 2], K_np[1, 2]
    vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    ray = torch.stack([(uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu)], 0).reshape(3, -1)
    c2w = torch.from_numpy(np.linalg.inv(m_t.w2c())).float().to(dev)
    w2c_s = torch.from_numpy(m_s.w2c()).float().to(dev)
    A = (w2c_s[:3, :3] @ c2w[:3, :3]) @ ray
    B = (w2c_s[:3, :3] @ c2w[:3, 3] + w2c_s[:3, 3])[:, None]
    Z = A[2:3] * depth_t.reshape(1, -1) + B[2:3]
    Z = torch.where(Z.abs() < 1e-6, torch.full_like(Z, 1e-6), Z)
    du = float(fx) * (A[0:1] * B[2:3] - A[2:3] * B[0:1]) / (Z ** 2)
    dv = float(fy) * (A[1:2] * B[2:3] - A[2:3] * B[1:2]) / (Z ** 2)
    return torch.cat([du, dv], 0).reshape(2, H, W)


def fit_ground_plane(pts, iters=300, thr_frac=0.02, seed=0):
    """RANSAC mặt phẳng qua points3D (đa số nằm ở đất/mái). -> (n, d) với n·X = d, n đơn vị, n hướng lên phía camera."""
    rng = np.random.default_rng(seed); P = pts[np.isfinite(pts).all(1)]
    if P.shape[0] > 200000: P = P[rng.choice(P.shape[0], 200000, replace=False)]
    ext = np.linalg.norm(P.max(0) - P.min(0)); thr = thr_frac * ext; best = (0, None, None)
    for _ in range(iters):
        i = rng.choice(P.shape[0], 3, replace=False); n = np.cross(P[i[1]] - P[i[0]], P[i[2]] - P[i[0]]); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n = n / nn; d = n @ P[i[0]]; k = int((np.abs(P @ n - d) < thr).sum())
        if k > best[0]: best = (k, n, d)
    k, n, d = best; inl = P[np.abs(P @ n - d) < thr]
    c = inl.mean(0); u, sv, vt = np.linalg.svd(inl - c, full_matrices=False); n = vt[2]; d = n @ c
    return n, d, k / P.shape[0]


def plane_depth(n, d, w2c, K_np, H, W, dev):
    """Depth (theo trục z camera) của tia mỗi pixel giao mặt phẳng n·X=d. (H,W), inf nếu không cắt phía trước."""
    fx, fy, cx, cy = K_np[0, 0], K_np[1, 1], K_np[0, 2], K_np[1, 2]
    c2w = np.linalg.inv(w2c); Rc, C = c2w[:3, :3], c2w[:3, 3]
    vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    dirs = torch.stack([(uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu)], -1)  # z=1
    Rt = torch.from_numpy(Rc).float().to(dev); nt = torch.from_numpy(n).float().to(dev)
    dw = dirs.reshape(-1, 3) @ Rt.T; denom = dw @ nt
    t = (float(d) - float(n @ C)) / denom
    t = torch.where(denom.abs() > 1e-6, t, torch.full_like(t, float("inf")))
    t = torch.where(t > 0, t, torch.full_like(t, float("inf")))
    return t.reshape(H, W)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", required=True); p.add_argument("--scene_dir", required=True)
    p.add_argument("--dump", required=True); p.add_argument("--targets", default="holdout", choices=["holdout", "test"])
    p.add_argument("--holdout_every", type=int, default=4); p.add_argument("--K", type=int, default=3)
    p.add_argument("--angle_max", type=float, default=25.0); p.add_argument("--overlap_min", type=float, default=0.4)
    p.add_argument("--tau_occ", type=float, default=0.2); p.add_argument("--blur_sigma", type=float, default=2.0)
    p.add_argument("--tile", type=int, default=128); p.add_argument("--max_shift", type=float, default=4.0)
    p.add_argument("--min_peak", type=float, default=0.03); p.add_argument("--shift_sign", type=int, default=-1)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--device", default="cuda"); p.add_argument("--rank_lf", type=int, default=0, help="1: warp K_cand ứng viên, xếp theo đồng thuận LF với render, giữ K tốt nhất"); p.add_argument("--K_cand", type=int, default=6); p.add_argument("--rank_mode", default="render", choices=["render", "median"]); p.add_argument("--plane_fix", type=float, default=0.0, help=">0: pixel có depth render < plane_fix × depth mặt đất (floater trước mặt) hoặc alpha<0.5 → thay bằng depth mặt đất"); p.add_argument("--fwd", type=int, default=0, help="1 = forward-warp theo depth nguồn (z-buffer) thay vì backward qua depth target")
    p.add_argument("--render_dir", default="", help="dùng ảnh ngoài (vd ensemble) làm render thay vì rasterize (depth vẫn từ ckpt)")
    p.add_argument("--flow", type=int, default=0, help="1: căn warp bằng optical flow dày (RAFT, 1/4 res) thay tile phase-correlation; giữ per-tile cái tốt hơn")
    p.add_argument("--flow_max", type=float, default=24.0, help="giới hạn |flow| (px full-res)")
    p.add_argument("--mono_fwd", type=int, default=0, help="31/08: 1 = forward-splat theo depth MONO (DA-v2) của ảnh NGUỒN, căn scale-shift (không gian disparity) theo depth splat ở view nguồn — hình học độc lập với render")
    p.add_argument("--mono_dir", default="", help="thư mục .npy mono depth (theo tên ảnh train, không đuôi)")
    p.add_argument("--depth_fix", type=int, default=0, help="01/09: sửa depth target từ bất đồng 2 warp (GT-free) rồi warp lại")
    p.add_argument("--dfix_max", type=float, default=32.0, help="giới hạn |flow| warp0↔warp1 (px)")
    p.add_argument("--dfix_blur", type=float, default=8.0, help="sigma làm trơn trường Δz")
    p.add_argument("--dfix_rel", type=float, default=0.15, help="giới hạn |Δz|/z")
    p.add_argument("--src_depth_dir", default="", help="02/09: thư mục .npy depth NGUỒN đã hiệu chỉnh (depth_prior.py) — mask che khuất hiện dùng depth nguồn cũng sai 3-6%")
    p.add_argument("--dfix_wval", type=int, default=0, help="1 = nhân trọng số mỗi cặp với vùng CẢ HAI warp hợp lệ")
    p.add_argument("--dfix_nsrc", type=int, default=2, help="số nguồn tham gia GIẢI Δz: mỗi cặp cho một ước lượng độc lập của CÙNG một đại lượng vô hướng → gộp theo trọng số thông tin")
    p.add_argument("--dfix_scale", type=float, default=0.25, help="tỉ lệ chạy RAFT cho bài toán giải depth (nghi phạm: 1/4 res chưa đủ chính xác dưới pixel)")
    p.add_argument("--dfix_edge", type=int, default=0, help="1 = làm trơn Δz dị hướng theo biên ảnh + trọng số thông tin |dk|^2 (thay Gauss)")
    p.add_argument("--dfix_hyp", type=float, default=0.0, help=">0: mỗi nguồn xuất THÊM một warp ở depth z + hyp*std[z] (depth như PHÂN BỐ, không phải 1 số) → dùng với --K 4")
    p.add_argument("--dfix_damp", type=float, default=1.0, help="hệ số giảm bước mỗi vòng (tránh phân kỳ ở vùng flow không tin được)")
    p.add_argument("--dfix_iters", type=int, default=1, help="số vòng lặp giải Δz (tuyến tính hoá → lặp bắt sai số lớn)")
    p.add_argument("--src_skip", type=int, default=0, help="31/08: BỎ QUA n nguồn đầu (đã xếp hạng) — cascade đa dạng nguồn: tầng 1 dùng nguồn 1-2, tầng 2 dùng nguồn 3-4 → tổng 4 nguồn thật mà mỗi mạng chỉ trả giá 2")
    p.add_argument("--render_aa", type=int, default=0, help="07/09: 1 = kênh render rasterize antialiased (khớp render_test_views/audit_padded khi base train --antialiased 1)"); p.add_argument("--pad", type=int, default=0, help="07/09: 1 = canvas ĐỆM (nội hoá audit_padded_06sep): render/warp/GT trong camera pinhole mở rộng để redistort không thiếu góc")
    p.add_argument("--apply_ckpt", default="", help="07/09: áp UNet NGAY trong tiến trình (không cần dump PNG) → --apply_out; ảnh ra đã redistort về khung GT nếu scene có méo")
    p.add_argument("--apply_out", default=""); p.add_argument("--apply_ch", default="48,96,192,384"); p.add_argument("--apply_fp16", type=int, default=1); p.add_argument("--apply_tile", type=int, default=1024); p.add_argument("--apply_overlap", type=int, default=64)
    p.add_argument("--apply_fallback", default="", help="thư mục render thô (khung GT) lấp góc khi KHÔNG đệm (như y39_redistort --fallback)"); p.add_argument("--no_dump", type=int, default=0, help="1 = không ghi PNG dump (dùng với --apply_ckpt)")
    a = p.parse_args(); dev = a.device
    # 28/08 BUG: trước đây không truyền holdout_offset → luôn residue mặc định (=2) cho MỌI model; với ho4o1/ho4o3
    # dump lấy nhầm VIEW TRAIN (render overfit). Nay đọc holdout_offset từ config.json của model.
    _cfg = json.load(open(os.path.join(a.result_dir, "config.json"))); _off = _cfg.get("holdout_offset", None)
    scene = SceneData(a.scene_dir, load_images=False, distorted=False,
                      holdout_every=(a.holdout_every if a.targets == "holdout" else 0), holdout_offset=_off)
    if a.targets == "holdout": print(f"[holdout] model {os.path.basename(a.result_dir)} offset={_off} -> residue {(a.holdout_every//2 if _off is None else int(_off)%a.holdout_every)}", flush=True)
    K_np = scene.K.astype(np.float64); H, W = scene.height, scene.width
    print("[dbg] load splats", flush=True); splats, sh_degree = FT.load_splats(os.path.join(a.result_dir, "ckpt.pt"), dev); print("[dbg] splats ok", flush=True)
    _pad = 0; _W0, _H0 = W, H; _K0 = K_np.copy(); _tpK = {tp.image_name: (tp.K, tp.width, tp.height) for tp in scene.test_poses}
    if a.pad:   # 07/09: canvas đệm — mọi thứ (raster, depth nguồn, warp, GT) trong camera pinhole mở rộng K' = K + pad; nguồn remap thẳng ảnh gốc vào K'
        _mx0, _my0 = scene.redistort_map(_K0, _W0, _H0)
        _pad = int(np.ceil((max(-_mx0.min(), _mx0.max() - _W0, -_my0.min(), _my0.max() - _H0) + 32) / 64) * 64)
        K_np = _K0.copy(); K_np[:2, 2] += _pad; W, H = _W0 + 2 * _pad, _H0 + 2 * _pad
        _ux, _uy = cv2.initUndistortRectifyMap(_K0, np.asarray(scene.dist, dtype=np.float64), None, K_np, (W, H), cv2.CV_32FC1)
        _validm = ((_ux >= 1) & (_ux <= _W0 - 2) & (_uy >= 1) & (_uy <= _H0 - 2)).astype(np.float32)
        scene.width, scene.height, scene.K = W, H, K_np.copy()
        _warp_orig = FT.warp_source; _vt = torch.from_numpy(_validm).to(dev)[None, None]
        def _warp_valid(img, *args, **kw):   # pixel nguồn ngoài ảnh gốc (vùng đệm) = không hợp lệ
            wr, inb, occ = _warp_orig(torch.cat([img, _vt], 1), *args, **kw); return wr[:, :3], inb & (wr[0, 3] > .999), occ
        FT.warp_source = _warp_valid
        print(f"[pad] đệm {_pad} px → canvas {W}×{H}, phủ hợp lệ {100 * _validm.mean():.1f} %", flush=True)
    _net = None
    if a.apply_ckpt:
        import refiner_train as RT
        RT.set_K(a.K, 2); _net = RT.UNet(ch=tuple(int(c) for c in a.apply_ch.split(","))).to(dev); _net.load_state_dict(torch.load(a.apply_ckpt, map_location=dev)); _net.eval()
        if a.apply_fp16: _net = _net.half()
        os.makedirs(a.apply_out, exist_ok=True); _rmaps = {}; _T, _O = a.apply_tile, a.apply_overlap
        _win = torch.hann_window(_T, periodic=False, device=dev); _w2 = (_win[:, None] * _win[None, :]).clamp_min(1e-3); _t_apply = dict(unet=0.0, n=0)
        print(f"[apply] UNet {a.apply_ckpt} K={a.K} fp16={a.apply_fp16} → {a.apply_out}", flush=True)
    _warps, _masks = [], []; nW = 0
    def _emit(w3, M, od):   # ghi warp/mask (trừ --no_dump) và giữ tensor cho apply trong tiến trình
        nonlocal nW
        if not a.no_dump:
            cv2.imwrite(os.path.join(od, f"warp{nW}.png"), cv2.cvtColor((w3.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(od, f"mask{nW}.png"), (M.cpu().numpy() * 255).astype(np.uint8))
        if _net is not None: _warps.append(w3.clamp(0, 1)); _masks.append(M)
        nW += 1
    img_dir = os.path.join(a.scene_dir, "train", "images")
    cache = {}
    def src_depth(si):
        """Depth của view NGUỒN — ưu tiên bản đã hiệu chỉnh (depth_prior.py) nếu có.
        02/09 BUG: trước đây override chỉ nằm ở vòng lặp nguồn, nhưng khối depth_fix chạy TRƯỚC và
        đã nạp cache bằng depth chưa sửa ⇒ override thành code chết (x22 ra đúng bằng mốc 4 chữ số)."""
        if si not in depth_cache:
            d, _ = FT.render_depth(splats, sh_degree, scene.train_metas[si].w2c(), K_np, W, H, dev)
            if a.src_depth_dir:
                _pp = os.path.join(a.src_depth_dir, os.path.splitext(scene.train_metas[si].name)[0] + ".npy")
                if os.path.exists(_pp):
                    _dp = torch.from_numpy(np.load(_pp).astype(np.float32)).to(dev)[None, None]
                    if _dp.numel() > 64:
                        d = F.interpolate(_dp, size=d.shape, mode="bilinear", align_corners=False)[0, 0]
            depth_cache[si] = d.cpu()
        return depth_cache[si].to(dev)

    # 05/09: scene có méo (loader undistort khi train) → nguồn warp và GT holdout cũng phải ở khung UNDISTORT (cùng K pinhole).
    _und = bool(getattr(scene, "need_undistort", False)); _Ku = np.asarray(scene.K, dtype=np.float64); _du = np.asarray(getattr(scene, "dist", np.zeros(4)), dtype=np.float64)
    if _und: print(f"[refdata] camera có méo {np.round(_du, 5).tolist()} → undistort ảnh nguồn + GT holdout", flush=True)
    def _rd_und(path):
        im = cv2.imread(path)
        if _pad: return cv2.remap(im, _ux, _uy, cv2.INTER_LINEAR)
        return cv2.undistort(im, _Ku, _du, None, _Ku) if _und else im
    def src_img(si):
        if si not in cache:
            if len(cache) > 24: cache.pop(next(iter(cache)))
            cache[si] = cv2.cvtColor(_rd_und(os.path.join(img_dir, scene.train_metas[si].name)), cv2.COLOR_BGR2RGB)
        return cache[si]
    if a.targets == "holdout":
        targets = [(m.name, m.w2c(), m) for m in scene.holdout_metas]
    else:
        targets = [(tp.image_name, tp.w2c, FT._T(tp)) for tp in scene.test_poses]
    if a.limit: targets = targets[: a.limit]
    plane = None
    if a.plane_fix > 0:
        pts = np.asarray(scene.points.xyz, dtype=np.float64)
        n, d, frac = fit_ground_plane(np.asarray(pts, dtype=np.float64)); plane = (n, d)
        print(f"[plane] n={np.round(n,3)} d={d:.3f} inlier {100*frac:.1f}%", flush=True)
    os.makedirs(a.dump, exist_ok=True); meta = []
    depth_cache = {}
    mono_cache = {}
    def mono_aligned(si, d_s):
        """Depth mono (DA-v2, disparity thô thấp phân giải) của ảnh nguồn si, căn a*x+b về 1/d_splat (LS + trim outlier 2 vòng).
        Trả depth metric (H,W); fallback d_s nếu thiếu file / fit suy biến."""
        name0 = os.path.splitext(scene.train_metas[si].name)[0]
        mp = os.path.join(a.mono_dir, name0 + ".npy")
        if not os.path.exists(mp):
            print(f"[mono] THIEU {name0}.npy -> fallback depth splat", flush=True); return d_s
        dm = torch.from_numpy(np.load(mp).astype(np.float32))[None, None].to(dev)
        dm = F.interpolate(dm, size=d_s.shape, mode="bilinear", align_corners=False)[0, 0]
        ok = d_s > 1e-6
        x = dm[ok].float(); y = 1.0 / d_s[ok].clamp_min(1e-6)
        if x.numel() > 200000:
            ii = torch.randperm(x.numel(), device=dev)[:200000]; x, y = x[ii], y[ii]
        a1 = None
        for _ in range(3):
            vx = x - x.mean(); vy = y - y.mean()
            a1 = (vx * vy).sum() / vx.pow(2).sum().clamp_min(1e-12); b1 = y.mean() - a1 * x.mean()
            r = (a1 * x + b1 - y).abs(); thr = 2.5 * r.std() + 1e-12; keep = r < thr
            if int(keep.sum()) < 1000: break
            x, y = x[keep], y[keep]
        if a1 is None or float(a1) <= 0:
            print(f"[mono] fit suy bien a={float(a1) if a1 is not None else 0:.3g} ({name0}) -> fallback", flush=True); return d_s
        inv = (a1 * dm + b1)
        dmin = float(d_s[ok].min().clamp_min(1e-6)); dmax = float(d_s[ok].max())
        inv = inv.clamp(min=0.5 / dmax, max=2.0 / dmin)
        res = float((a1 * x + b1 - y).abs().median())
        print(f"[mono] {name0[-8:]} a={float(a1):.4g} b={float(b1):.4g} res_med={res:.3e} (1/d) valid={100*float((dm>1e-6).float().mean()):.1f}%", flush=True)
        return torch.where(dm > 1e-6, 1.0 / inv, d_s)   # 31/08: pixel nguồn không hợp lệ (MVS lỗ) -> depth splat
    for name, w2c, m_t in targets:
        base = os.path.splitext(name)[0]; od = os.path.join(a.dump, base); os.makedirs(od, exist_ok=True)
        with torch.no_grad():
            from gsplat.rendering import rasterization
            vm = torch.from_numpy(w2c).float().to(dev)[None]; Kt = torch.from_numpy(K_np).float().to(dev)[None]
            rend, alpha, _ = rasterization(means=splats["means"], quats=splats["quats"], scales=splats["scales"],
                opacities=splats["opacities"], colors=splats["colors"], viewmats=vm, Ks=Kt, width=W, height=H,
                sh_degree=sh_degree, render_mode="RGB+ED", rasterize_mode="classic", near_plane=0.01, far_plane=1e10, packed=False)
            R = rend[0, ..., :3].clamp(0, 1); depth_t = rend[0, ..., 3]; alpha_t = alpha[0, ..., 0]
            if a.render_aa:   # 07/09: kênh render theo rasterize_mode antialiased (như render_test_views / audit_padded); depth+alpha vẫn từ lượt classic
                _rgb_aa, _, _ = rasterization(means=splats["means"], quats=splats["quats"], scales=splats["scales"], opacities=splats["opacities"], colors=splats["colors"],
                    viewmats=vm, Ks=Kt, width=W, height=H, sh_degree=sh_degree, render_mode="RGB", rasterize_mode="antialiased", near_plane=0.01, far_plane=1e10, packed=False)
                R = _rgb_aa[0, ..., :3].clamp(0, 1)
            if plane is not None:
                print("[dbg] plane_depth", flush=True); dp = plane_depth(plane[0], plane[1], w2c, K_np, H, W, dev); print("[dbg] plane_depth ok", flush=True)
                bad = (depth_t < a.plane_fix * dp) | (alpha_t < 0.5)
                bad &= torch.isfinite(dp)
                depth_t = torch.where(bad, dp, depth_t); alpha_t = torch.where(bad, torch.ones_like(alpha_t), alpha_t)
                print(f"[plane] {name[-11:-4]} thay {100*bad.float().mean():.1f}% pixel", flush=True)
        if a.render_dir:
            _rp = [q for e in (".png", ".jpg", ".JPG") for q in [os.path.join(a.render_dir, base + e)] if os.path.exists(q)][0]
            R = torch.from_numpy(cv2.cvtColor(cv2.imread(_rp), cv2.COLOR_BGR2RGB)).to(dev).float().div(255)
        R_t = R.permute(2, 0, 1)[None]
        a.K, _Kbak = (a.K_cand if a.rank_lf else max(a.K + a.src_skip, a.dfix_nsrc if a.depth_fix else 0)), a.K
        src_idx = FT.select_sources(scene, m_t, depth_t, alpha_t, K_np, a); a.K = _Kbak
        if a.src_skip:
            _n0 = len(src_idx); src_idx = src_idx[a.src_skip:]
            print(f"[skip] {os.path.splitext(name)[0][-8:]} bỏ {a.src_skip} nguồn đầu: {_n0} -> {len(src_idx)}", flush=True)
        if not a.no_dump: cv2.imwrite(os.path.join(od, "render.png"), cv2.cvtColor((R.cpu().numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
        # 28/08: kênh phụ cho refiner — depth (uint16, chuẩn theo 4×median) + alpha
        _dm = float(depth_t[alpha_t > 0.5].median()) if bool((alpha_t > 0.5).any()) else 1.0
        if not a.no_dump: cv2.imwrite(os.path.join(od, "depth.png"), (depth_t / (4 * _dm)).clamp(0, 1).mul(65535).cpu().numpy().astype(np.uint16))
        if not a.no_dump: cv2.imwrite(os.path.join(od, "alpha.png"), (alpha_t.clamp(0, 1) * 255).cpu().numpy().astype(np.uint8))
        R_lf = FT.gauss_blur(R_t, a.blur_sigma); R_gray_lf = R_lf.mean(dim=1)[0]
        if a.depth_fix and len(src_idx) >= 2:
            # 01/09 — SỬA ĐỘ SÂU TỪ BẤT ĐỒNG GIỮA HAI ẢNH THẬT (GT-free, hợp lệ khi nộp).
            # Đo 01/09: hai warp lệch nhau median 34 px (tile-align chỉ ±4). Căn 2D về nhau ăn +0,49 dB
            # nhưng chỉ khử phần lệch TƯƠNG ĐỐI. Ở đây giải thẳng sai số độ sâu:
            #     f01 = (k0 − k1)·Δz,  k_s = ∂u_s/∂z giải tích  ⇒  Δz = ((k0−k1)·f01)/|k0−k1|²
            # rồi warp lại CẢ HAI bằng depth đã sửa → về đúng chỗ, khử được cả phần chung.
            _z0 = depth_t.clone()
            for _it in range(max(1, a.dfix_iters)):   # phép giải là TUYẾN TÍNH HOÁ → lặp để bắt sai số lớn
                _W, _J, _M = [], [], []
                for si in src_idx[: max(2, a.dfix_nsrc)]:
                    _ds0 = src_depth(si)
                    _im0 = torch.from_numpy(src_img(si)).to(dev).float().div(255).permute(2, 0, 1)[None]
                    _w0, _inb0, _occ0 = FT.warp_source(_im0, _ds0, m_t, scene.train_metas[si], depth_t, K_np, dev)
                    _ok0 = (_inb0 & (_occ0 < a.tau_occ) & (alpha_t > 0.5))
                    _W.append(torch.where(_ok0[None, None], _w0, R_t))     # lấp vùng hỏng bằng render: RAFT khỏi loạn
                    _M.append(_ok0.float())
                    _J.append(dz_jacobian(m_t, scene.train_metas[si], depth_t, K_np, dev))
                # Mỗi CẶP nguồn cho một ước lượng độc lập của CÙNG một số vô hướng Δz:
                #     Δz_ij = (dk_ij · f_ij)/|dk_ij|²   với dk_ij = J_i − J_j
                # Gộp theo hợp lý cực đại (nhiễu flow đẳng hướng) = trọng số |dk_ij|²:
                #     Δz = Σ (dk_ij · f_ij) / Σ |dk_ij|²
                # NS=2 rút về đúng công thức cũ. Khác hẳn "consensus 3 nguồn" (dịch 2D) từng thua:
                # ở đây nguồn tệ chỉ đóng góp trọng số thấp thay vì kéo lệch phép dịch.
                _num = 0; den = 0; _nf = []
                for _i in range(len(_W)):
                    for _j in range(_i + 1, len(_W)):
                        _f = flow_raft(_W[_i], _W[_j], dev, a.dfix_max, a.dfix_scale)[0]
                        _dkij = _J[_i] - _J[_j]
                        # 02/09 VÁ: cặp chỉ đáng tin ở pixel CẢ HAI warp đều hợp lệ. Chỗ một bên hỏng đã bị
                        # lấp bằng render ⇒ flow ở đó đo lệch warp↔RENDER = đúng cái đã thất bại (−0,58).
                        # nsrc=3 không có trọng số này = −0,45. Trọng số = |dk|² × (mask_i ∧ mask_j).
                        _wij = (_M[_i] * _M[_j]) if a.dfix_wval else 1.0
                        _num = _num + _wij * (_dkij * _f).sum(0)
                        den = den + _wij * (_dkij ** 2).sum(0)
                        _nf.append(float(_f.norm(dim=0).median()))
                f01 = _f; dk = _dkij
                lim = a.dfix_rel * _z0                                      # biên tính theo depth GỐC, không trôi theo vòng lặp
                ok = (den.sqrt() * lim) > 1.0                               # đủ nhạy để giải (không suy biến)
                dz = torch.where(ok, _num / den.clamp_min(1e-12), torch.zeros_like(den))
                if a.dfix_edge:
                    dz = aniso_smooth(dz, den * ok.float(), R_t[0]) * a.dfix_damp
                else:
                    dz = FT.gauss_blur(dz[None, None], a.dfix_blur)[0, 0] * a.dfix_damp
                depth_t = (depth_t - dz).clamp(_z0 - lim, _z0 + lim).clamp_min(1e-4)
                _rel = ((depth_t - _z0).abs() / _z0.clamp_min(1e-6))
                print(f"[dfix] {base[-8:]} it{_it} |Δz|/z med {float(_rel.median()):.4f} "
                      f"p90 {float(_rel.flatten().kthvalue(int(0.9 * _rel.numel())).values):.4f} "
                      f"|f| med {np.mean(_nf):.1f}px ({len(_nf)} cặp) giải được {100 * float(ok.float().mean()):.0f}%", flush=True)
            _dm = float(depth_t[alpha_t > 0.5].median()) if bool((alpha_t > 0.5).any()) else 1.0
            if not a.no_dump: cv2.imwrite(os.path.join(od, "depth.png"), (depth_t / (4 * _dm)).clamp(0, 1).mul(65535).cpu().numpy().astype(np.uint16))
        sigz = None
        if a.dfix_hyp > 0:
            _ez, sigz, _al = FT.render_depth_moments(splats, (m_t.w2c() if hasattr(m_t, "w2c") else w2c), K_np, W, H, dev)
            print(f"[hyp] {base[-8:]} std[z]/z med {float((sigz / depth_t.clamp_min(1e-6)).median()):.4f} "
                  f"p90 {float((sigz / depth_t.clamp_min(1e-6)).flatten().kthvalue(int(0.9 * sigz.numel())).values):.4f}", flush=True)
        nW = 0; _warps.clear(); _masks.clear()
        if a.rank_lf and len(src_idx) > a.K:
            # 28/08: nguồn gần theo baseline nhưng lệch trục quang (oblique, gimbal quay) cho warp rác → chấm từng ứng viên
            cand = []
            for si in src_idx[: a.K_cand]:
                if si not in depth_cache:
                    d_s0, _ = FT.render_depth(splats, sh_degree, scene.train_metas[si].w2c(), K_np, W, H, dev); depth_cache[si] = d_s0.cpu()
                d_s0 = depth_cache[si].to(dev)
                img0 = torch.from_numpy(src_img(si)).to(dev).float().div(255).permute(2, 0, 1)[None]
                Wc, inbc, occc = FT.warp_source(img0, d_s0, m_t, scene.train_metas[si], depth_t, K_np, dev)
                vm = (inbc & (occc < a.tau_occ)).float()
                cand.append([None, si, FT.gauss_blur(Wc, a.blur_sigma), vm])
            if a.rank_mode == "median":
                # 28/08: render có thể sai (sương) → chuẩn = median LF của các ứng viên (nguồn rác lệch với đa số)
                stack = torch.stack([c[2][0] for c in cand]); vms = torch.stack([c[3] for c in cand])
                med = torch.nanmedian(torch.where(vms[:, None] > 0, stack, torch.full_like(stack, float("nan"))), dim=0).values
                med = torch.nan_to_num(med, nan=0.0)
                for c in cand:
                    vm = c[3]; agree = ((c[2][0] - med).abs().mean(0) * vm).sum() / vm.sum().clamp_min(1)
                    c[0] = float(agree) + 0.5 * (1 - float(vm.mean()))
            else:
                for c in cand:
                    vm = c[3]; agree = ((c[2] - R_lf).abs().mean(1)[0] * vm).sum() / vm.sum().clamp_min(1)
                    c[0] = float(agree) + 0.5 * (1 - float(vm.mean()))
            cand.sort(key=lambda c: c[0]); src_idx = [c[1] for c in cand]
            cand = [(c[0], c[1]) for c in cand]
            print(f"[rank] {base[-8:]} chọn {[scene.train_metas[si].name[-9:-4] for si in src_idx[:a.K]]} score {[round(c[0],3) for c in cand[:a.K]]}", flush=True)
        for si in src_idx[: (max(1, a.K // 2) if a.dfix_hyp > 0 else a.K)]:
            d_s = src_depth(si)
            img_s = torch.from_numpy(src_img(si)).to(dev).float().div(255).permute(2, 0, 1)[None]
            if a.mono_fwd:
                if si not in mono_cache:
                    if len(mono_cache) > 24: mono_cache.pop(next(iter(mono_cache)))
                    mono_cache[si] = mono_aligned(si, d_s).cpu()
                d_m = mono_cache[si].to(dev)
                Wf, Mf = fwd_splat(img_s, d_m, scene.train_metas[si].w2c(), (m_t.w2c() if hasattr(m_t, "w2c") else w2c), K_np, H, W, dev)
                M = (Mf > 0).float() * (alpha_t > 0.5).float(); _emit(Wf, M, od)
                continue
            if a.fwd:
                Wf, Mf = fwd_splat(img_s, d_s, scene.train_metas[si].w2c(), (m_t.w2c() if hasattr(m_t, "w2c") else w2c), K_np, H, W, dev)
                M = (Mf > 0).float() * (alpha_t > 0.5).float(); _emit(Wf, M, od)
                continue
            W0, inb, occ = FT.warp_source(img_s, d_s, m_t, scene.train_metas[si], depth_t, K_np, dev)
            if a.flow:
                W1, grid, fl = flow_align(W0, R_t, dev, a.flow_max)
                inb1 = F.grid_sample(inb.float()[None, None], grid, mode="nearest", padding_mode="zeros", align_corners=True)[0, 0] > 0.5
                occ1 = F.grid_sample(occ[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]
                print(f"[flow] {base[-8:]} src {scene.train_metas[si].name[-9:-4]} |flow| median {float(fl.norm(dim=1).median()):.2f} px p90 {float(fl.norm(dim=1).flatten().kthvalue(int(0.9*fl[0,0].numel())).values):.2f}", flush=True)
            else:
                shift = FT.tile_align(FT.gauss_blur(W0, a.blur_sigma).mean(dim=1)[0], R_gray_lf, a.tile, a.max_shift, a.min_peak, a.shift_sign)
                W1, inb1, occ1 = FT.warp_source(img_s, d_s, m_t, scene.train_metas[si], depth_t, K_np, dev, extra_shift=shift)
            e0 = F.avg_pool2d((FT.gauss_blur(W0, a.blur_sigma) - R_lf).abs().mean(1, keepdim=True), a.tile, stride=a.tile, ceil_mode=True)
            e1 = F.avg_pool2d((FT.gauss_blur(W1, a.blur_sigma) - R_lf).abs().mean(1, keepdim=True), a.tile, stride=a.tile, ceil_mode=True)
            better = F.interpolate((e1 < e0).float(), size=(H, W), mode="nearest")[0, 0]
            Wf = W1 * better[None, None] + W0 * (1 - better[None, None])
            inb = torch.where(better.bool(), inb1, inb); occ = torch.where(better.bool(), occ1, occ)
            M = (inb & (occ < a.tau_occ) & (alpha_t > 0.5)).float(); _emit(Wf[0], M, od)
            if a.dfix_hyp > 0 and sigz is not None and nW < a.K:
                # GIẢ THUYẾT ĐỘ SÂU THỨ HAI: depth là PHÂN BỐ dọc tia, không phải một số.
                # Ở mép nhà/sương, expected-depth rơi vào GIỮA hai mặt → warp lấy nhầm bề mặt.
                # Cấp thêm một warp ở z + hyp·std[z] và để MẠNG tự chọn (kiểu cost-volume của MVS).
                _d2 = (depth_t + a.dfix_hyp * sigz).clamp_min(1e-4)
                W2, inb2, occ2 = FT.warp_source(img_s, d_s, m_t, scene.train_metas[si], _d2, K_np, dev)
                M2 = (inb2 & (occ2 < a.tau_occ) & (alpha_t > 0.5)).float(); _emit(W2[0], M2, od)
        while nW < a.K:   # thiếu nguồn: warp = render, mask = 0
            _emit(R.permute(2, 0, 1), torch.zeros(H, W, device=dev), od)
        if _net is not None:   # 07/09: áp UNet ngay (kênh lượng tử 8 bit y như đọc PNG) → redistort về khung GT → ghi 1 ảnh
            torch.cuda.synchronize(); _ta = time.time(); _q = lambda t: (t * 255).round().div(255)
            _dep8 = ((depth_t / (4 * _dm)).clamp(0, 1).mul(65535).floor() / 256).floor().div(255)
            _x = torch.cat([_q(R_t[0])] + [_q(w) for w in _warps[: a.K]] + [_q(m)[None] for m in _masks[: a.K]] + [_dep8[None], _q(alpha_t.clamp(0, 1))[None]], 0)[None]
            _xt = _x.half() if a.apply_fp16 else _x; _acc = torch.zeros(1, 3, H, W, device=dev); _wacc = torch.zeros(1, 1, H, W, device=dev)
            _ys = list(range(0, max(H - _T, 0) + 1, _T - _O)); _xs = list(range(0, max(W - _T, 0) + 1, _T - _O))
            if _ys[-1] != H - _T: _ys.append(H - _T)
            if _xs[-1] != W - _T: _xs.append(W - _T)
            with torch.no_grad():
                for _y0 in _ys:
                    for _x0 in _xs:
                        _pr = _net(_xt[..., _y0:_y0 + _T, _x0:_x0 + _T]); _acc[..., _y0:_y0 + _T, _x0:_x0 + _T] += _pr.float() * _w2; _wacc[..., _y0:_y0 + _T, _x0:_x0 + _T] += _w2
            _img = cv2.cvtColor(((_acc / _wacc)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
            torch.cuda.synchronize(); _t_apply["unet"] += time.time() - _ta; _t_apply["n"] += 1
            if _und:   # về khung GT (méo): remap theo K GỐC của pose (+pad nếu canvas đệm)
                _Kr, _Wr, _Hr = _tpK.get(name, (_K0, _W0, _H0)); _key = (tuple(np.round(np.asarray(_Kr, dtype=np.float64).ravel(), 3)), _Wr, _Hr)
                if _key not in _rmaps: _rmaps[_key] = scene.redistort_map(np.asarray(_Kr, dtype=np.float64), _Wr, _Hr)
                _mx, _my = _rmaps[_key]; _out = cv2.remap(_img, _mx + _pad, _my + _pad, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                if a.apply_fallback and not _pad:
                    _fb = [q for e in (".png", ".jpg", ".JPG") for q in [os.path.join(a.apply_fallback, base + e)] if os.path.exists(q)]
                    if _fb:
                        _F = cv2.imread(_fb[0]); _ins = (_mx >= 1) & (_mx <= _img.shape[1] - 2) & (_my >= 1) & (_my <= _img.shape[0] - 2)
                        _m = cv2.GaussianBlur(_ins.astype(np.float32), (0, 0), 3)[..., None]; _out = (_out.astype(np.float32) * _m + _F.astype(np.float32) * (1 - _m)).round().clip(0, 255).astype(np.uint8)
            else: _out = _img
            cv2.imwrite(os.path.join(a.apply_out, base + ".png"), _out)
        gt = os.path.join(img_dir, name) if a.targets == "holdout" else ""
        if gt and _und and not a.no_dump:  # GT holdout ở khung undistort, lưu cạnh dump
            cv2.imwrite(os.path.join(od, "gt_und.png"), _rd_und(gt)); gt = os.path.join(od, "gt_und.png")
        meta.append(dict(name=base, gt=gt, n_src=len(src_idx))); print(f"[refdata] {base[-8:]} src={len(src_idx)}", flush=True)
    json.dump(meta, open(os.path.join(a.dump, "meta.json"), "w"), indent=1)
    if _net is not None: print(f"APPLY_INPROC n={_t_apply['n']} unet_s/view={_t_apply['unet'] / max(_t_apply['n'], 1):.3f} pad={_pad} K={a.K} -> {a.apply_out}", flush=True)
    print(f"REFDATA_DONE n={len(meta)} -> {a.dump}")

if __name__ == "__main__":
    main()
