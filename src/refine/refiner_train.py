"""Refiner có điều kiện láng giềng (26/08): UNet nhỏ, vào = [render, warp×K, mask×K] (3+3K+K kênh),
ra = residual (gated) cộng vào render. Train trên crop 512² từ dump holdout (refiner_data.py --targets holdout),
loss = 0,6·L1 + 0,3·(1−SSIM) + 0,4·LPIPS_vgg (theo thước). Suy luận tile 1024 chồng 64, cửa sổ Hann.
  train:  python refiner_train.py train --data DUMP_HO --out CKPT.pt [--val_frac 0.15] [--iters 6000]
  apply:  python refiner_train.py apply --data DUMP_TEST --ckpt CKPT.pt --out DIR_PNG
"""
import argparse, time, os, json, random, sys, math, re, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa

K = 3
EXTRA = 0
COND = 0
def set_K(k, extra=0, cond=0):
    global K, EXTRA, COND; K = k; EXTRA = extra + 3 * cond; COND = cond
def conv(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GELU())
class PatchD(nn.Module):   # 30/08: PatchGAN nhỏ (70×70) cho fine-tune adversarial nhẹ
    def __init__(self, c=3, ch=64):
        super().__init__()
        def blk(i, o, s): return nn.Sequential(nn.Conv2d(i, o, 4, s, 1), nn.LeakyReLU(0.2, True))
        self.net = nn.Sequential(blk(c, ch, 2), blk(ch, ch * 2, 2), blk(ch * 2, ch * 4, 2), blk(ch * 4, ch * 8, 1), nn.Conv2d(ch * 8, 1, 4, 1, 1))
    def forward(self, x): return self.net(x * 2 - 1)

class MDTA(nn.Module):
    """Restormer (CVPR22): attention theo KÊNH thay vì theo pixel — chi phí TUYẾN TÍNH theo
    số pixel nên chạy được ở ảnh 21 MP. Ma trận attention là C×C (không phải HW×HW)."""
    def __init__(self, c, heads=4):
        super().__init__(); self.h = heads
        self.t = nn.Parameter(torch.ones(heads, 1, 1))
        self.n = nn.GroupNorm(1, c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.dw = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3)
        self.pj = nn.Conv2d(c, c, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.dw(self.qkv(self.n(x))).chunk(3, 1)
        q = F.normalize(q.reshape(B, self.h, C // self.h, H * W), dim=-1)
        k = F.normalize(k.reshape(B, self.h, C // self.h, H * W), dim=-1)
        v = v.reshape(B, self.h, C // self.h, H * W)
        a = torch.softmax((q @ k.transpose(-2, -1)) * self.t, -1)
        return x + self.pj((a @ v).reshape(B, C, H, W))

class GDFN(nn.Module):
    """Restormer gated-Dconv FFN: cổng nhân từng phần tử → lọc thông tin theo vị trí."""
    def __init__(self, c, e=2.0):
        super().__init__(); h = int(c * e)
        self.n = nn.GroupNorm(1, c)
        self.i = nn.Conv2d(c, h * 2, 1)
        self.dw = nn.Conv2d(h * 2, h * 2, 3, padding=1, groups=h * 2)
        self.o = nn.Conv2d(h, c, 1)
    def forward(self, x):
        a, b = self.dw(self.i(self.n(x))).chunk(2, 1)
        return x + self.o(F.gelu(a) * b)

def tstack(c, n):
    return nn.Sequential(*[m for _ in range(n) for m in (MDTA(c), GDFN(c))]) if n > 0 else nn.Identity()

class UNet(nn.Module):
    def __init__(self, cin=None, ch=(32, 64, 128, 256), tblocks=0):
        super().__init__(); cin = cin or (3 + 4 * K + EXTRA)
        self.tb = tstack(ch[-1], tblocks)
        self.e = nn.ModuleList(); c = cin
        for o in ch: self.e.append(conv(c, o)); c = o
        self.d = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1): self.d.append(conv(ch[i] + ch[i - 1], ch[i - 1]))
        self.out = nn.Conv2d(ch[0], 4, 3, padding=1)   # 3 residual + 1 gate
    def forward(self, x):
        R = x[:, :3]; fs = []; h = x
        for i, e in enumerate(self.e):
            h = e(h); fs.append(h)
            if i < len(self.e) - 1: h = F.max_pool2d(h, 2)
        h = self.tb(h)
        for d, f in zip(self.d, fs[-2::-1]):
            h = F.interpolate(h, size=f.shape[-2:], mode="bilinear", align_corners=False); h = d(torch.cat([h, f], 1))
        o = self.out(h); res, g = o[:, :3], torch.sigmoid(o[:, 3:4])
        return (R + g * res).clamp(0, 1)

class UNetBlend(nn.Module):
    """02/09: CHỌN thay vì TỔNG HỢP LẠI.  out = Σ_j α_j·S_j + gate·tanh(res)·max_res,
    với S = [render, warp_0 … warp_{K−1}] và α = softmax có che theo mask nguồn.

    Chẩn đoán 02/09 (hf_audit): warp là ẢNH TRAIN THẬT nắn về pose test nên mang ~100 %
    năng lượng HF, còn đầu ra cuối chỉ giữ ~10 %. Kiến trúc cũ `render + gate·res` bắt một
    UNet 4,4 M TÁI TẠO texture qua conv dưới loss L1 — dưới bất định căn chỉnh, nghiệm tối ưu
    của L1 là TRUNG BÌNH = làm mờ (hồi quy về trung vị). Ở đây HF đi thẳng từ pixel thật;
    mạng chỉ phải quyết định lấy từ đâu, còn residual bị chặn biên độ nên không mờ hoá được.
    Cùng họ với Deep Blending (Hedman 2018) và nhánh soft-selection của RefSR."""
    def __init__(self, cin=None, ch=(32, 64, 128, 256), max_res=0.15, tblocks=0):
        super().__init__(); cin = cin or (3 + 4 * K + EXTRA); self.max_res = max_res
        self.tb = tstack(ch[-1], tblocks)
        self.e = nn.ModuleList(); c = cin
        for o in ch: self.e.append(conv(c, o)); c = o
        self.d = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1): self.d.append(conv(ch[i] + ch[i - 1], ch[i - 1]))
        self.out = nn.Conv2d(ch[0], (K + 1) + 4, 3, padding=1)   # K+1 logit chọn + 3 res + 1 gate
    def forward(self, x):
        R = x[:, :3]
        S = [R] + [x[:, 3 + 3 * i:6 + 3 * i] for i in range(K)]
        Msk = torch.cat([torch.ones_like(R[:, :1])]
                        + [x[:, 3 + 3 * K + i:4 + 3 * K + i] for i in range(K)], 1)
        fs = []; h = x
        for i, e in enumerate(self.e):
            h = e(h); fs.append(h)
            if i < len(self.e) - 1: h = F.max_pool2d(h, 2)
        h = self.tb(h)
        for d, f in zip(self.d, fs[-2::-1]):
            h = F.interpolate(h, size=f.shape[-2:], mode="bilinear", align_corners=False); h = d(torch.cat([h, f], 1))
        o = self.out(h)
        lg = o[:, :K + 1].masked_fill(Msk < 0.5, -1e4)
        w = torch.softmax(lg, 1)
        base = sum(w[:, j:j + 1] * S[j] for j in range(K + 1))
        g = torch.sigmoid(o[:, K + 4:K + 5])
        return (base + g * torch.tanh(o[:, K + 1:K + 4]) * self.max_res).clamp(0, 1)

class CorrAlign(nn.Module):
    """02/09: căn warp bằng TÌM KIẾM tháp 2 tầng (khối tương quan) chứ không bằng HỒI QUY.

    AlignNet (31/08) hồi quy flow thẳng từ đặc trưng ghép và thất bại (−0,011) — đúng kiểu
    FlowNetS thua PWC-Net. Ở đây mạng THẤY điểm giống của từng dịch chuyển.

    Bán kính chọn theo số đo align_gap 02/09 (flow oracle warp0→GT): median 17,7 px,
    p90 50,0 px — bản r=6 @1/4 res (±24 px) THIẾU. Tháp:
        tầng 1  1/8 res, r=8  → ±64 px full-res   (bắt phần lớn lệch)
        tầng 2  1/4 res, r=3  → ±12 px residual   (tinh, dưới pixel)
    Cùng align_gap: trường flow làm trơn theo ô 64 px giữ được gần trọn oracle
    (rho 0,289 vs 0,291) ⇒ KHÔNG cần flow dày từng pixel, ước lượng thô là đủ."""
    def __init__(self, ch=32, r1=8, r2=3):
        super().__init__(); self.r1, self.r2 = r1, r2
        self.enc = nn.Sequential(nn.Conv2d(3, ch, 5, 2, 2), nn.GELU(),
                                 nn.Conv2d(ch, ch, 3, 2, 1), nn.GELU(),
                                 nn.Conv2d(ch, ch, 3, padding=1))
        n1, n2 = (2 * r1 + 1) ** 2, (2 * r2 + 1) ** 2
        self.ref1 = nn.Sequential(nn.Conv2d(n1, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, n1, 3, padding=1))
        self.ref2 = nn.Sequential(nn.Conv2d(n2, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, n2, 3, padding=1))
        self.tau1 = nn.Parameter(torch.tensor(2.0)); self.tau2 = nn.Parameter(torch.tensor(2.0))
        self.step = nn.Parameter(torch.tensor(0.5))   # 0,5 = đi ĐIỂM GIỮA (đo 02/09: mid 0,141 > toWarp1 0,116)
    @staticmethod
    def _grid(x, fl):
        H, W = x.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(H, device=x.device, dtype=x.dtype),
                                torch.arange(W, device=x.device, dtype=x.dtype), indexing="ij")
        return torch.stack([(xx[None] + fl[:, 0]) / max(W - 1, 1) * 2 - 1,
                            (yy[None] + fl[:, 1]) / max(H - 1, 1) * 2 - 1], -1)
    def _corr(self, fa, fb, r, ref, tau):
        # unfold + einsum: MỘT matmul thay cho (2r+1)² lát nhỏ (nhanh ~2x)
        B, C, h, w = fa.shape
        n = (2 * r + 1) ** 2
        u = F.unfold(fb, 2 * r + 1, padding=r).view(B, C, n, h * w)
        cost = torch.einsum("bcl,bcnl->bnl", fa.reshape(B, C, h * w), u).view(B, n, h, w)
        cost = cost + ref(cost)
        p = torch.softmax(cost * tau.clamp(0.2, 20.0), 1)
        dv = torch.arange(-r, r + 1, device=fa.device, dtype=fa.dtype)
        dy = dv.repeat_interleave(2 * r + 1); dx = dv.repeat(2 * r + 1)
        return torch.stack([(p * dx.view(1, -1, 1, 1)).sum(1), (p * dy.view(1, -1, 1, 1)).sum(1)], 1)
    def forward(self, R, W, M, T=None):
        if T is None: T = R
        f4r = F.normalize(self.enc(T), dim=1); f4w = F.normalize(self.enc(W), dim=1)
        f8r, f8w = F.avg_pool2d(f4r, 2), F.avg_pool2d(f4w, 2)
        fl8 = self._corr(f8r, f8w, self.r1, self.ref1, self.tau1)            # đơn vị pixel 1/8
        fl4 = F.interpolate(fl8, size=f4r.shape[-2:], mode="bilinear", align_corners=False) * 2
        f4wa = F.grid_sample(f4w, self._grid(f4w, fl4), mode="bilinear",
                             padding_mode="border", align_corners=True)
        fl4 = fl4 + self._corr(f4r, f4wa, self.r2, self.ref2, self.tau2)     # đơn vị pixel 1/4
        fl = F.interpolate(fl4 * 4.0, size=R.shape[-2:], mode="bilinear", align_corners=False) * self.step.clamp(0.0, 1.0)
        g = self._grid(W, fl)
        W2 = F.grid_sample(W, g, mode="bilinear", padding_mode="border", align_corners=True)
        M2 = (F.grid_sample(M, g, mode="bilinear", padding_mode="zeros", align_corners=True) > 0.5).to(M.dtype)
        return W2, M2

class UNetCorr(nn.Module):
    """CorrAlign (chung trọng số cho K nguồn) + UNetBlend. Kênh vào/ra không đổi."""
    def __init__(self, ch=(32, 64, 128, 256), r=8, blend=1, tblocks=0, mutual=0):
        super().__init__(); self.align = CorrAlign(r1=r, r2=3); self.mutual = mutual
        self.body = UNetBlend(ch=ch, tblocks=tblocks) if blend else UNet(ch=ch, tblocks=tblocks)
    def forward(self, x):
        R = x[:, :3]; parts = [R]; ms = []
        Ws = [x[:, 3 + 3 * i:6 + 3 * i] for i in range(K)]
        for i in range(K):
            M = x[:, 3 + 3 * K + i:4 + 3 * K + i]
            # mutual: căn warp_i về warp_{j} (ảnh THẬT), KHÔNG về render.
            # Đo 02/09: rho(warp0) 0,102 · căn về render 0,057 (HẠI) · về warp1 0,116 · điểm giữa 0,141.
            T = Ws[(i + 1) % K] if (self.mutual and K > 1) else R
            W2, M2 = self.align(R, Ws[i], M, T); parts.append(W2); ms.append(M2)
        return self.body(torch.cat(parts + ms + [x[:, 3 + 4 * K:]], 1))

class AlignNet(nn.Module):
    """31/08: học CĂN WARP có giám sát GT (EDVR/FDAN-style) — 2 thang: flow thô ±32 px @1/8 res + tinh ±4 px full-res.
    Khác flow RAFT render→warp (−0,58: căn về hình học SAI của render): ở đây flow do loss ảnh với GT dạy, mạng học
    "warp thường lệch kiểu gì" (parallax cục bộ) thay vì khớp mù với render."""
    def __init__(self, ch=24, max_lr=32.0, max_hr=4.0):
        super().__init__(); self.max_lr, self.max_hr = max_lr, max_hr
        self.enc = nn.Sequential(conv(7, ch), nn.MaxPool2d(2), conv(ch, ch * 2), nn.MaxPool2d(2), conv(ch * 2, ch * 2), nn.MaxPool2d(2), conv(ch * 2, ch * 2))
        self.head = nn.Conv2d(ch * 2, 2, 3, padding=1)
        self.ref = nn.Sequential(nn.Conv2d(7, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, 2, 3, padding=1))
    @staticmethod
    def _warp(W, M, flow):
        B, _, H, Wd = W.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=W.device, dtype=W.dtype), torch.arange(Wd, device=W.device, dtype=W.dtype), indexing="ij")
        gx = (xx[None] + flow[:, 0]) / max(Wd - 1, 1) * 2 - 1; gy = (yy[None] + flow[:, 1]) / max(H - 1, 1) * 2 - 1
        g = torch.stack([gx, gy], -1)
        W2 = F.grid_sample(W, g, mode="bilinear", padding_mode="border", align_corners=True)
        M2 = (F.grid_sample(M, g, mode="bilinear", padding_mode="zeros", align_corners=True) > 0.5).to(M.dtype)
        return W2, M2
    def forward(self, R, W, M):
        x = torch.cat([R, W, M], 1)
        fl = torch.tanh(self.head(self.enc(x))) * self.max_lr
        fl = F.interpolate(fl, size=R.shape[-2:], mode="bilinear", align_corners=False)
        W1, M1 = self._warp(W, M, fl)
        d = torch.tanh(self.ref(torch.cat([R, W1, M1], 1))) * self.max_hr
        return self._warp(W, M, fl + d)

class UNetAligned(nn.Module):
    """AlignNet (chung trọng số cho K nguồn) + UNet chuẩn. Kênh vào/ra không đổi."""
    def __init__(self, ch=(32, 64, 128, 256)):
        super().__init__(); self.align = AlignNet(); self.unet = UNet(ch=ch)
    def forward(self, x):
        R = x[:, :3]; parts = [R]
        ms = []
        for i in range(K):
            W = x[:, 3 + 3 * i:6 + 3 * i]; M = x[:, 3 + 3 * K + i:4 + 3 * K + i]
            W2, M2 = self.align(R, W, M); parts.append(W2); ms.append(M2)
        rest = x[:, 3 + 4 * K:]
        return self.unet(torch.cat(parts + ms + [rest], 1))

def load_view(d, with_gt):
    r = cv2.cvtColor(cv2.imread(os.path.join(d, "render.png")), cv2.COLOR_BGR2RGB)
    ws = [cv2.cvtColor(cv2.imread(os.path.join(d, f"warp{i}.png")), cv2.COLOR_BGR2RGB) for i in range(K)]
    ms = [cv2.imread(os.path.join(d, f"mask{i}.png"), 0)[..., None] for i in range(K)]
    ex = []
    if EXTRA:   # depth (uint16 → uint8 theo 8 bit cao) + alpha
        dep = cv2.imread(os.path.join(d, "depth.png"), cv2.IMREAD_UNCHANGED); ex.append((dep >> 8).astype(np.uint8)[..., None])
        ex.append(cv2.imread(os.path.join(d, "alpha.png"), 0)[..., None])
    if COND:    # 30/08: đầu ra refiner THÔ (cond.png) làm kênh điều kiện cho refiner mịn (coarse-to-fine)
        ex.append(cv2.cvtColor(cv2.imread(os.path.join(d, "cond.png")), cv2.COLOR_BGR2RGB))
    x = np.concatenate([r] + ws + ms + ex, -1)     # H,W,3+3K+K(+2) uint8
    return x, (None if not with_gt else None)

def to_t(x): return torch.from_numpy(x).permute(2, 0, 1).float().div(255)

def ssim_t(a, b):
    from trainer import ssim_torch  # noqa
    return ssim_torch(a, b)

def make_net(a, dev):
    """02/09: chọn kiến trúc — unet (gốc) · blend (chọn nguồn) · corr (căn bằng tương quan + chọn)."""
    ch = tuple(int(c) for c in a.ch.split(","))
    arch = getattr(a, "arch", "unet")
    if a.align: return UNetAligned(ch=ch).to(dev)
    if arch == "blend": return UNetBlend(ch=ch, max_res=a.max_res, tblocks=getattr(a, "tblocks", 0)).to(dev)
    if arch == "corr":  return UNetCorr(ch=ch, r=a.corr_r, blend=1, tblocks=getattr(a, "tblocks", 0)).to(dev)
    if arch == "corrm": return UNetCorr(ch=ch, r=a.corr_r, blend=0, tblocks=getattr(a, "tblocks", 0), mutual=1).to(dev)
    if arch == "corru": return UNetCorr(ch=ch, r=a.corr_r, blend=0).to(dev)
    return UNet(ch=ch, tblocks=getattr(a, "tblocks", 0)).to(dev)

def train(a):
    torch.set_num_threads(8)
    dev = "cuda"; meta = []
    for _d in a.data.split(","):   # nhiều dump (vd 2 model holdout) — ghi kèm đường dẫn
        for _m in json.load(open(os.path.join(_d, "meta.json"))): _m["dir"] = _d; meta.append(_m)
    random.seed(0); random.shuffle(meta); torch.manual_seed(a.seed); random.seed(a.seed)
    if a.view_re: meta = [m for m in meta if re.search(a.view_re, m["name"])]        # 29/08: lọc view (test generalization vùng)
    if a.view_re_neg: meta = [m for m in meta if not re.search(a.view_re_neg, m["name"])]
    if a.max_views > 0: meta = meta[: a.max_views]   # giới hạn RAM (~315 MB/view ở 21 MP)
    nv = max(1, int(len(meta) * a.val_frac)); val, tr = meta[:nv], meta[nv:]
    print(f"[refiner] train {len(tr)} / val {len(val)} view", flush=True)
    def load_all(lst):
        out = []
        for m in lst:
            x, _ = load_view(os.path.join(m.get("dir", a.data), m["name"]), True)
            g = cv2.cvtColor(cv2.imread(m["gt"]), cv2.COLOR_BGR2RGB); out.append((x, g))
        return out
    TR, VA = load_all(tr), load_all(val)
    if a.gpu_data:   # 27/08: đưa dataset uint8 lên VRAM, cắt crop bằng torch → hết CPU/RAM (2 refiner từng ăn 54 GB + 14 core mỗi cái)
        TR = [(torch.from_numpy(x).to(dev), torch.from_numpy(g).to(dev)) for x, g in TR]
        VA = [(torch.from_numpy(x).to(dev), torch.from_numpy(g).to(dev)) for x, g in VA]
        print(f"[refiner] dataset trên GPU: {sum(x.numel()+g.numel() for x,g in TR+VA)/2**30:.1f} GB", flush=True)
    import lpips; lp = lpips.LPIPS(net="vgg").to(dev).eval()
    for q in lp.parameters(): q.requires_grad_(False)
    net = make_net(a, dev)
    if a.init: net.load_state_dict(torch.load(a.init, map_location=dev)); print(f"[refiner] init từ {a.init}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    if a.w_gan > 0:   # hinge GAN, D nhỏ, lr D = lr
        D = PatchD().to(dev); optD = torch.optim.AdamW(D.parameters(), lr=a.lr, betas=(0.5, 0.99), weight_decay=1e-4)
    # 01/09: hard-example mining — crop lấy mẫu theo BẢN ĐỒ KHÓ thay vì đều.
    # Lý do (chẩn đoán 01/09): vùng không nguồn 8,7 % diện tích gánh 14 % lỗi, vùng texture dày
    # 31 % gánh 47 % lỗi; crop đều ⇒ mạng thấy chủ yếu vùng dễ. Đây là trục DATA (chưa ai thử ở đây),
    # khác mọi thứ đã đóng (kiến trúc/loss/ngữ cảnh/kênh điều kiện).
    HARD = []
    if a.hard_mine > 0:
        S = 32
        def sl(t, i0, i1):   # lát kênh [i0:i1) → (1,C,h,w) float trên dev, không nhân bản cả stack
            u = t[..., i0:i1] if torch.is_tensor(t) else torch.from_numpy(np.ascontiguousarray(t[..., i0:i1]))
            return u.to(dev).permute(2, 0, 1)[None].float()
        def nz(t): return (t - t.min()) / (t.max() - t.min() + 1e-6)
        for x, g in TR:
            lf_err = (F.avg_pool2d(sl(x, 0, 3), S) - F.avg_pool2d(sl(g, 0, 3), S))[0].abs().mean(0)
            nosrc = 1 - F.avg_pool2d(sl(x, 3 + 3 * K, 3 + 4 * K) / 255.0, S)[0].amax(0)
            h = 0.5 * nz(lf_err) + 0.5 * nz(nosrc) + 1e-3
            HARD.append((h / h.sum()).flatten().cpu())
        print(f"[refiner] hard-mine bật: {len(HARD)} bản đồ khó {tuple(HARD[0].shape)}, tỉ lệ crop khó = {a.hard_mine}", flush=True)

    def batch(pool, bs, cs):
        xs, gs = [], []
        for _ in range(bs):
            i = random.randrange(len(pool)); x, g = pool[i]; H, W = g.shape[:2]; cs = min(cs, H, W); cs -= cs % 8
            if HARD and pool is TR and random.random() < a.hard_mine:
                S = 32; hw = W // S
                c = int(torch.multinomial(HARD[i], 1).item())
                cy, cx = (c // hw) * S + S // 2, (c % hw) * S + S // 2
                y0 = min(max(cy - cs // 2 + random.randint(-cs // 4, cs // 4), 0), H - cs)
                x0 = min(max(cx - cs // 2 + random.randint(-cs // 4, cs // 4), 0), W - cs)
            else:
                y0 = random.randint(0, H - cs); x0 = random.randint(0, W - cs)
            if a.gpu_data:
                xs.append(x[y0:y0 + cs, x0:x0 + cs].permute(2, 0, 1).float().div(255)); gs.append(g[y0:y0 + cs, x0:x0 + cs].permute(2, 0, 1).float().div(255))
            else:
                xs.append(to_t(x[y0:y0 + cs, x0:x0 + cs])); gs.append(to_t(g[y0:y0 + cs, x0:x0 + cs]))
        X, G_ = torch.stack(xs).to(dev), torch.stack(gs).to(dev)
        if a.src_drop > 0:   # 30/08: source dropout — bỏ ngẫu nhiên 1 nguồn (warp=render, mask=0) như khi thiếu láng giềng
            for b in range(X.shape[0]):
                if random.random() < a.src_drop:
                    i = random.randrange(K); X[b, 3 + 3 * i:6 + 3 * i] = X[b, :3]; X[b, 3 + 3 * K + i] = 0
        if a.scale != 1.0:   # 28/08: refiner đa tỉ lệ — crop lớn hạ về scale (context ×1/scale)
            X = F.interpolate(X, scale_factor=a.scale, mode="area"); G_ = F.interpolate(G_, scale_factor=a.scale, mode="area")
        return X, G_
    def score(pr, gt):
        with torch.no_grad():
            mse = F.mse_loss(pr, gt).clamp_min(1e-10); psnr = 10 * torch.log10(1 / mse)
            s = ssim_t(pr, gt); l = lp(pr, gt, normalize=False).mean()
        return float(100 * (0.4 * (1 - l) + 0.3 * s + 0.3 * min(psnr / 50, 1))), float(psnr), float(s), float(l)
    EMA = None
    if a.ema > 0:
        EMA = {k: v.detach().clone().float() for k, v in net.state_dict().items()}
        net_ema = make_net(a, dev)
        print(f"[refiner] EMA bật, decay={a.ema}", flush=True)
    best = -1
    for it in range(1, a.iters + 1):
        if it == 1: _t_loop = time.time()
        net.train(); x, g = batch(TR, a.bs, int(a.crop / a.scale))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.amp)):   # 07/09: --amp 1 = bf16 autocast cho UNet (loss tính fp32 bên ngoài)
            pr = net(x)
        pr = pr.float()
        loss = 0.6 * F.l1_loss(pr, g) + 0.3 * (1 - ssim_t(pr, g)) + a.w_lpips * lp(pr, g, normalize=False).mean()
        if a.w_lf > 0:   # 30/08: loss băng thấp (avg-pool 16) nhắm 58 % lỗi LF
            loss = loss + a.w_lf * F.mse_loss(F.avg_pool2d(pr, 16), F.avg_pool2d(g, 16)).sqrt()
        if a.w_gan > 0:
            loss = loss + a.w_gan * (-D(pr).mean())
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 2000 == 0: print(f"[refiner] {it} iter, {it / (time.time() - _t_loop):.2f} it/s (amp={a.amp})", flush=True)
        if EMA is not None:
            with torch.no_grad():
                d_ = min(a.ema, (1 + it) / (10 + it))      # warmup: tránh EMA bám trọng số khởi tạo
                for k, v in net.state_dict().items():
                    EMA[k].mul_(d_).add_(v.detach().float(), alpha=1 - d_)
        if a.w_gan > 0:
            dl = F.relu(1 - D(g)).mean() + F.relu(1 + D(pr.detach())).mean(); optD.zero_grad(); dl.backward(); optD.step()
        if it % a.eval_every == 0 or it == a.iters:
            net.eval(); torch.manual_seed(0); random.seed(1)
            if EMA is not None:
                net_ema.load_state_dict({k: v.to(dtype=net.state_dict()[k].dtype) for k, v in EMA.items()}); net_ema.eval()
            _net = net_ema if EMA is not None else net
            sb, sr = [], []
            for _ in range(24):
                x, g = batch(VA, 1, int(1024 / a.scale)); pr = _net(x)
                sb.append(score(x[:, :3], g)); sr.append(score(pr, g))
            sb, sr = np.array(sb).mean(0), np.array(sr).mean(0)
            print(f"[{it:5d}] loss={loss.item():.4f} VAL base {sb[0]:.2f} (psnr {sb[1]:.2f} ssim {sb[2]:.3f} lp {sb[3]:.3f})"
                  f" -> refined {sr[0]:.2f} (psnr {sr[1]:.2f} ssim {sr[2]:.3f} lp {sr[3]:.3f}) Δ={sr[0]-sb[0]:+.2f}", flush=True)
            if sr[0] > best: best = sr[0]; torch.save((net_ema if EMA is not None else net).state_dict(), a.out)
            random.seed(it)
    print(f"REFINER_DONE best_val={best:.2f} -> {a.out}")

@torch.no_grad()
def apply(a):
    dev = "cuda"; net = make_net(a, dev); net.load_state_dict(torch.load(a.ckpt, map_location=dev)); net.eval()
    meta = json.load(open(os.path.join(a.data, "meta.json"))); os.makedirs(a.out, exist_ok=True)
    T0, O = a.tile, a.overlap
    if a.fp16: net = net.half()
    torch.cuda.reset_peak_memory_stats(); t_all = time.time(); t_net = 0.0
    for m in meta:
        x, _ = load_view(os.path.join(a.data, m["name"]), False); xt = to_t(x).to(dev)[None]
        if a.fp16: xt = xt.half()
        torch.cuda.synchronize(); t0 = time.time()
        H0, W0 = xt.shape[-2:]
        if a.scale != 1.0: xt = F.interpolate(xt, scale_factor=a.scale, mode="area")
        _, C, H, W = xt.shape
        T = min(T0, H, W); T -= T % 8; win = torch.hann_window(T, periodic=False, device=dev); w2 = (win[:, None] * win[None, :]).clamp_min(1e-3)   # tile không vượt ảnh (scale<1)
        acc = torch.zeros(1, 3, H, W, device=dev); wacc = torch.zeros(1, 1, H, W, device=dev)
        ys = list(range(0, max(H - T, 0) + 1, T - O)); xs = list(range(0, max(W - T, 0) + 1, T - O))
        if ys[-1] != H - T: ys.append(H - T)
        if xs[-1] != W - T: xs.append(W - T)
        for y0 in ys:
            for x0 in xs:
                xt_ = xt[..., y0:y0 + T, x0:x0 + T]; pr = net(xt_)
                if a.tta:   # lật ngang/dọc rồi lật lại, trung bình (TTA)
                    pr = (pr + net(xt_.flip(-1)).flip(-1) + net(xt_.flip(-2)).flip(-2)) / 3
                acc[..., y0:y0 + T, x0:x0 + T] += pr.float() * w2; wacc[..., y0:y0 + T, x0:x0 + T] += w2
        outt = (acc / wacc).float()
        torch.cuda.synchronize(); t_net += time.time() - t0
        if a.scale != 1.0: outt = F.interpolate(outt, size=(H0, W0), mode="bicubic", align_corners=False)
        out = outt[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy(); img8 = cv2.cvtColor((out * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(a.out, m["name"] + ".png"), img8)
        if a.write_cond: cv2.imwrite(os.path.join(a.data, m["name"], "cond.png"), img8)   # ghi làm kênh điều kiện cho refiner mịn
        print(f"[apply] {m['name'][-8:]}", flush=True)
    n = len(meta); print(f"APPLY_TIME n={n} unet_s/view={t_net/n:.3f} wall_s/view={(time.time()-t_all)/n:.3f} "
          f"peakVRAM_GB={torch.cuda.max_memory_allocated()/2**30:.2f} fp16={a.fp16} tile={a.tile} K={K} params={sum(p.numel() for p in net.parameters())}")
    print(f"APPLY_DONE n={len(meta)} -> {a.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=["train", "apply"]); p.add_argument("--data", required=True)
    p.add_argument("--out", required=True); p.add_argument("--ckpt", default=""); p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--bs", type=int, default=4); p.add_argument("--crop", type=int, default=512); p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val_frac", type=float, default=0.15); p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--tile", type=int, default=1024); p.add_argument("--overlap", type=int, default=64); p.add_argument("--tta", type=int, default=0); p.add_argument("--ch", default="32,64,128,256"); p.add_argument("--max_views", type=int, default=0); p.add_argument("--w_lpips", type=float, default=0.4); p.add_argument("--gpu_data", type=int, default=0); p.add_argument("--K", type=int, default=3); p.add_argument("--scale", type=float, default=1.0); p.add_argument("--seed", type=int, default=0); p.add_argument("--init", default=""); p.add_argument("--extra", type=int, default=0, help="1: thêm kênh depth+alpha (2 kênh)")
    p.add_argument("--align", type=int, default=0, help="31/08: 1 = AlignNet căn warp học được (2 thang) trước UNet"); p.add_argument("--arch", default="unet", choices=["unet", "blend", "corr", "corru", "corrm"], help="02/09: blend = softmax chọn giữa render/warp (HF từ pixel thật); corr = CorrAlign + blend; corru = CorrAlign + UNet gốc"); p.add_argument("--max_res", type=float, default=0.15, help="biên độ residual tối đa của nhánh blend"); p.add_argument("--tblocks", type=int, default=0, help="02/09: N khối transformer Restormer (MDTA+GDFN) ở đáy UNet — attention theo kênh, tuyến tính theo pixel"); p.add_argument("--corr_r", type=int, default=8, help="bán kính tầng thô @1/8 res (r=8 → ±64 px full-res); tầng tinh cố định r=3 @1/4 res (±12 px)"); p.add_argument("--fp16", type=int, default=0); p.add_argument("--src_drop", type=float, default=0.0); p.add_argument("--w_gan", type=float, default=0.0); p.add_argument("--w_lf", type=float, default=0.0); p.add_argument("--cond", type=int, default=0); p.add_argument("--write_cond", type=int, default=0); p.add_argument("--view_re", default=""); p.add_argument("--view_re_neg", default="")
    p.add_argument("--amp", type=int, default=0, help="07/09: 1 = bf16 autocast khi train (tốc độ)"); p.add_argument("--ema", type=float, default=0.0, help="02/09: trung bình trượt trọng số (EMA/SWA) — kỹ thuật chuẩn ngoài 3DGS, dự án chưa dùng; 0 = tắt, thường 0.999")
    p.add_argument("--hard_mine", type=float, default=0.0, help="01/09: tỉ lệ crop lấy theo bản đồ khó (LF-err + không-nguồn); 0 = đều như cũ")
    a = p.parse_args(); set_K(a.K, 2 if a.extra else 0, a.cond); train(a) if a.mode == "train" else apply(a)
