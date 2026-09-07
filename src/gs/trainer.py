"""3DGS trainer for the VAI NVS challenge.

Recipe: gsplat rasterization + MCMC densification strategy + antialiasing +
per-image bilateral grid (exposure compensation) + sparse-depth regularization
from COLMAP points + optional LPIPS fine-tuning phase.

Trains on ALL train images (no holdout — public scenes have real GT in
test/images). After training, renders every pose in test_poses.csv, with the
bilateral grid of the nearest train view(s), and (optionally) re-distorts the
pinhole render back to the SIMPLE_RADIAL frame of the original images.
"""
import argparse
import copy
import json
import math
import os
import re
import time

import cv2
# 05/09 y67/y68: nội suy khi méo lại render — lanczos hơn cubic trên ảnh hoàn hảo nhưng ±0 trên render thô (53,457 vs 53,456) và ÂM trên output refiner (60,13 vs 60,19) → giữ cubic; VT_REDISTORT_INTERP=lanczos để thử
_REDISTORT_INTERP = {"cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4, "linear": cv2.INTER_LINEAR}[os.environ.get("VT_REDISTORT_INTERP", "cubic")]
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa
from dataset import SceneData, TestPose
from lib_bilagrid import BilateralGrid, slice as bg_slice, total_variation_loss

C0 = 0.28209479177387814


def rgb_to_sh(rgb):
    return (rgb - 0.5) / C0


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


# --------------------------------------------------------------------------
# SSIM (pure torch, gaussian window 11, matches Wang et al. 2004)
# --------------------------------------------------------------------------
def _gaussian_window(size=11, sigma=1.5, channels=3, device="cuda"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(0)
    w = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return w.expand(channels, 1, size, size).contiguous()


_ssim_win = None


def ssim_torch(img1, img2):  # NCHW in [0,1]
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
    C1, C2 = 0.01**2, 0.03**2
    m = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return m.mean()


# --------------------------------------------------------------------------
def create_splats_and_optimizers(scene: SceneData, cfg, device, init_state=None):
    """init_state: splats state_dict from a checkpoint -> fine-tune mode
    (skips point-cloud init, keeps gaussian count fixed)."""
    if init_state is not None:
        splats = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v.clone().float()) for k, v in init_state.items()}
        ).to(device)
        lrs = {"means": cfg.means_lr * scene.scene_scale, "scales": 5e-3,
               "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
        params = [(n, splats[n], lrs[n]) for n in splats.keys()]
    else:
        xyz, rgb = scene.init_points()
        points = torch.from_numpy(xyz)
        rgbs = torch.from_numpy(rgb)

        tree = cKDTree(xyz)
        d, _ = tree.query(xyz, k=4)
        dist_avg = np.sqrt((d[:, 1:] ** 2).mean(axis=1))
        scales = torch.log(torch.from_numpy(dist_avg).float() * cfg.init_scale + 1e-10)
        scales = scales.unsqueeze(-1).repeat(1, 3)

        N = points.shape[0]
        quats = torch.rand((N, 4))
        opacities = torch.logit(torch.full((N,), cfg.init_opacity))
        colors = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))
        colors[:, 0, :] = rgb_to_sh(rgbs)

        params = [
            ("means", torch.nn.Parameter(points.float()), cfg.means_lr * scene.scene_scale),
            ("scales", torch.nn.Parameter(scales), 5e-3),
            ("quats", torch.nn.Parameter(quats), 1e-3),
            ("opacities", torch.nn.Parameter(opacities), 5e-2),
            ("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3),
            ("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20),
        ]
        splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    lr_scale = getattr(cfg, "lr_scale", 1.0)
    optimizers = {
        n: torch.optim.Adam([{"params": splats[n], "lr": lr * lr_scale, "name": n}],
                            eps=1e-15)
        for n, _, lr in params
    }
    return splats, optimizers


@torch.no_grad()
def compute_mip_filter(means, viewmats, K, W, H, factor=0.2):
    """Mip-Splatting 3D smoothing filter (Yu et al., CVPR 2024): per-Gaussian
    world-space sigma = factor * (nearest-view depth / focal) = the size of one
    pixel at that Gaussian in the view that samples it most finely. Added in
    quadrature to the scales at render time so no primitive is sub-pixel; kills
    the aliasing floaters/speckle that thin high-contrast BTS structure bakes
    into tiny Gaussians. One intrinsic per scene here, so focal is constant.
    Returns (N,) detached tensor (a render-time buffer, recomputed as means
    move; never a learnable parameter)."""
    N = means.shape[0]
    device = means.device
    fx, cx = float(K[0, 0]), float(K[0, 2])
    fy, cy = float(K[1, 1]), float(K[1, 2])
    homo = torch.cat([means, torch.ones(N, 1, device=device)], dim=1)  # (N,4)
    dist = torch.full((N,), 1e10, device=device)
    valid = torch.zeros(N, dtype=torch.bool, device=device)
    for w2c in viewmats:
        cam = homo @ w2c.T          # (N,4) world->cam
        z = cam[:, 2]
        x = cam[:, 0] / z * fx + cx
        y = cam[:, 1] / z * fy + cy
        ok = (z > 0.2) & (x >= -0.15 * W) & (x <= 1.15 * W) \
            & (y >= -0.15 * H) & (y <= 1.15 * H)
        dist = torch.where(ok, torch.minimum(dist, z), dist)
        valid |= ok
    if valid.any():
        dist[~valid] = dist[valid].max()
    else:
        dist[:] = 1.0
    return (dist / fx * factor).contiguous()   # (N,) world-space sigma


_MIP_RATIO_FLOOR = 1e-12


def _mip_opacity_ratio(s2, new_s2):
    """sqrt(det Σ / det(Σ+σ²I)) — bù khối lượng cho bộ lọc 3D Mip-Splatting.

    KHÔNG được viết là ``sqrt(s2.prod(1) / new_s2.prod(1))``. Đó chính là lỗi
    giết c3_mip (j209) và e1_96m_mip (j211) ngày 15/08, cả hai chết ở refine
    ~#12 với ``binarySearchForMultinomial: cumdist[size-1] > 0 failed``:

    1. MCMC relocate đẻ ra Gaussian rất nhỏ. scale 1e-7 ⇒ s2 = 1e-14 ⇒
       ``s2.prod(dim=1)`` = 1e-42, **dưới ngưỡng số chuẩn nhỏ nhất của float32
       (1.18e-38) ⇒ tràn xuống 0**. Mẫu số thì không tràn (luôn ≥ σ⁶ > 0).
    2. Tỉ số thành đúng 0, và ``sqrt`` tại 0 có đạo hàm **vô cực** ⇒ gradient
       inf ⇒ sau một bước optimizer toàn bộ opacity thành NaN.
    3. torch.multinomial trong MCMC nhận vector xác suất NaN ⇒ device-side
       assert, và assert CUDA là bất khả hồi: cả tiến trình phải chết.

    Cách viết dưới nhân **tỉ số theo từng chiều** — mỗi thừa số nằm trong (0,1]
    nên không có phép nhân nào tràn — rồi kẹp sàn để đạo hàm sqrt hữu hạn.
    """
    ratio = (s2 / new_s2.clamp(min=torch.finfo(s2.dtype).tiny)).prod(dim=1)
    return torch.sqrt(ratio.clamp(min=_MIP_RATIO_FLOOR))


def rasterize(splats, viewmats, Ks, width, height, sh_degree, cfg,
              render_depth=False, radial_coeffs=None,
              means_override=None, quats_override=None, absgrad=False,
              scale_mult=1.0, scale_comp="none"):
    from gsplat.rendering import rasterization

    # Mip-Splatting 3D filter: dilate scales in quadrature by the per-Gaussian
    # sampling-limit sigma and compensate opacity to preserve integrated mass.
    # Detached buffer stashed on `splats` (train() keeps it fresh); absent =>
    # exact original behaviour. Pure scale/opacity reparam, so UT-compatible.
    scales_r = torch.exp(splats["scales"])
    opacities_r = torch.sigmoid(splats["opacities"])
    mip = getattr(splats, "_mip_filter", None)
    if mip is not None:
        s2 = scales_r * scales_r
        new_s2 = s2 + (mip * mip).unsqueeze(-1)          # (N,3)
        opacities_r = opacities_r * _mip_opacity_ratio(s2, new_s2)
        scales_r = torch.sqrt(new_s2)

    # Pure render-time footprint probe.  A model can cover a high-frequency
    # region with Gaussians whose projected footprints are slightly too broad;
    # shrinking them at evaluation sharpens the *3D representation* rather
    # than hallucinating pixels after rendering.  Optional compensation keeps
    # either projected 2D mass (~s^2) or volumetric mass (~s^3) approximately
    # constant.  Defaults are exact historical behaviour.
    sm = float(scale_mult)
    if abs(sm - 1.0) > 1e-8:
        scales_r = scales_r * sm
        if scale_comp == "area":
            opacities_r = opacities_r / (sm * sm)
        elif scale_comp == "volume":
            opacities_r = opacities_r / (sm * sm * sm)
        elif scale_comp != "none":
            raise ValueError(f"unknown scale compensation {scale_comp!r}")
        opacities_r = opacities_r.clamp(max=0.999)

    extra = {}
    depth_mode = "RGB+ED"
    if radial_coeffs is not None:
        # 3DGUT path: models OpenCV radial distortion directly (no undistort)
        extra = dict(with_ut=True, with_eval3d=True, radial_coeffs=radial_coeffs)
        mode = "classic"  # antialiased not supported with UT
        # NOTE: UT aborts (core dump, gsplat 1.5.3) on BOTH RGB+ED and RGB+D —
        # depth rendering is unsupported; callers must keep depth_weight=0.
    else:
        mode = "antialiased" if cfg.antialiased else "classic"
    renders, alphas, info = rasterization(
        means=splats["means"] if means_override is None else means_override,
        quats=splats["quats"] if quats_override is None else quats_override,
        scales=scales_r,
        opacities=opacities_r,
        colors=torch.cat([splats["sh0"], splats["shN"]], 1),
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=depth_mode if render_depth else "RGB",
        rasterize_mode=mode,
        near_plane=0.01,
        far_plane=1e10,
        packed=False,
        absgrad=absgrad,
        **extra,
    )
    return renders, alphas, info


def rasterize_projected_ut_absgrad(splats, viewmats, Ks, width, height,
                                   sh_degree, radial_coeffs=None,
                                   means_override=None, quats_override=None):
    """Render an isolated projected-UT graph for an AbsGS topology signal.

    Production distorted training uses ``with_ut=True, with_eval3d=True``.
    gsplat 1.5.3 does not expose ``means2d.absgrad`` from that Eval3D pixel
    kernel.  J122-D therefore makes a *second*, projected-2D UT render with
    ``with_eval3d=False``.  Every splat tensor is detached, and a disposable SH
    leaf is the only differentiable model input.  Backpropagating to that leaf
    runs gsplat's pixel backward (which attaches ``means2d.absgrad``) without
    adding any gradient to the production parameters.

    This function is deliberately separate from :func:`rasterize`: changing
    it cannot silently switch the renderer or objective used by the real loss.
    """
    from gsplat.rendering import rasterization

    means = (splats["means"] if means_override is None
             else means_override).detach()
    quats = (splats["quats"] if quats_override is None
             else quats_override).detach()
    scales = torch.exp(splats["scales"].detach())
    opacities = torch.sigmoid(splats["opacities"].detach())

    # Match the production render-time Mip reparameterization if it is active.
    mip = getattr(splats, "_mip_filter", None)
    if mip is not None:
        mip = mip.detach()
        s2 = scales * scales
        new_s2 = s2 + (mip * mip).unsqueeze(-1)
        opacities = opacities * _mip_opacity_ratio(s2, new_s2)
        scales = torch.sqrt(new_s2)

    # At least one raster input must require grad for the custom CUDA backward
    # to run.  This leaf is intentionally thrown away after autograd.grad().
    sh_leaf = torch.cat(
        [splats["sh0"].detach(), splats["shN"].detach()], dim=1
    ).requires_grad_(True)
    extra = dict(with_ut=True, with_eval3d=False)
    if radial_coeffs is not None:
        extra["radial_coeffs"] = radial_coeffs.detach()
    renders, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh_leaf,
        viewmats=viewmats.detach(),
        Ks=Ks.detach(),
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode="RGB",
        rasterize_mode="classic",
        near_plane=0.01,
        far_plane=1e10,
        packed=False,
        absgrad=True,
        **extra,
    )
    return renders, alphas, info, sh_leaf


def _errguided_mcmc_class():
    """MCMC that biases relocation/growth toward high-error Gaussians.

    Plain MCMCStrategy samples teleport/add targets purely ∝ opacity mass, so
    budget flows to dense high-opacity regions (roofs) regardless of where the
    reconstruction error lives. This subclass multiplies the sampling prob by a
    per-Gaussian weight (1 + λ·w) where w = accumulated screen-space |grad| of
    the 2D means (AbsGS signal; falls back to signed .grad if absgrad absent
    under the UT path), normalized to mean 1. Everything else — compute_relocation,
    optimizer-state surgery, noise injection — reuses gsplat's own ops verbatim
    (only the `probs` line changes), so it is bit-faithful outside the weighting.
    λ=0 ⇒ identical to MCMCStrategy. See ROOTCAUSE_2607 radial deficit + AbsGS/
    ConeGS (error-guided densification, 2026).
    """
    from gsplat.strategy import MCMCStrategy
    from gsplat.strategy import ops as gsops
    from gsplat.relocation import compute_relocation
    from gsplat.strategy.ops import inject_noise_to_position

    class ErrGuidedMCMC(MCMCStrategy):
        def __init__(self, err_lambda=1.0, **kw):
            super().__init__(**kw)
            self.err_lambda = float(err_lambda)

        def initialize_state(self):
            st = super().initialize_state()
            st["err"] = None
            st["count"] = None
            return st

        def step_pre_backward(self, params, optimizers, state, step, info):
            # 3DGUT/UT exposes no differentiable means2d (retain_grad fails), so
            # instead of the AbsGS positional gradient we use the per-pixel
            # render-vs-GT error sampled at each Gaussian's screen center — a
            # UT-safe, ConeGS-style error signal. The loop stashes info["_pix_err"].
            pass

        @torch.no_grad()
        def _update_state(self, params, state, info):
            err = info.get("_pix_err", None)          # [1,H,W] render-vs-GT L1
            if err is None:
                return
            m2d = info["means2d"]                      # [C,N,2] pixel coords
            radii = info["radii"]
            H, W = int(err.shape[-2]), int(err.shape[-1])
            m = m2d[0] if m2d.dim() == 3 else m2d      # [N,2] (single train view)
            rv = radii[0] if (radii.dim() >= 2 and radii.shape[0] == 1) else radii
            sel = (rv > 0).all(dim=-1) if rv.dim() == 2 else (rv > 0)   # [N]
            N = params["means"].shape[0]
            if state["err"] is None:
                state["err"] = torch.zeros(N, device=m2d.device)
                state["count"] = torch.zeros(N, device=m2d.device)
            gs_ids = torch.where(sel)[0]
            xs = m[gs_ids, 0].round().long().clamp(0, W - 1)
            ys = m[gs_ids, 1].round().long().clamp(0, H - 1)
            e_per = err.reshape(H, W)[ys, xs]
            state["err"].index_add_(0, gs_ids, e_per)
            state["count"].index_add_(0, gs_ids, torch.ones_like(e_per))

        @staticmethod
        def _safe_probs(probs, fallback, where):
            """gsplat lấy mẫu bằng np.random.choice khi N > 2^24 (đúng vùng cap
            ≥ 24M ta đang chạy), và hàm đó CHIA cho tổng ⇒ tổng 0 thành NaN, còn
            NaN lọt vào thì nó ném thẳng ValueError. j209/c2_err chết ở đúng đây
            sau ~155 lần refine. Không được để một nhánh thí nghiệm mất 25 phút
            GPU vì một phần tử hỏng: khử NaN, và nếu vẫn không dùng được thì rơi
            về đúng MCMC gốc (chỉ theo opacity) cho lần refine đó."""
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            if not bool(torch.isfinite(probs).all()) or float(probs.sum()) <= 0:
                print(f"[errguided] CẢNH BÁO {where}: probs hỏng, rơi về MCMC gốc",
                      flush=True)
                probs = torch.nan_to_num(fallback, nan=0.0, posinf=0.0, neginf=0.0)
                if float(probs.sum()) <= 0:
                    probs = torch.ones_like(probs)
            return probs

        @torch.no_grad()
        def _relocate_weighted(self, params, optimizers, binoms, weight):
            opacities = torch.sigmoid(params["opacities"])
            # NaN so sánh luôn cho False ⇒ Gaussian có opacity NaN sẽ lọt vào
            # `alive` và đầu độc probs. Loại nó ra như thể đã chết.
            flat = opacities.flatten()
            bad = ~torch.isfinite(flat)
            mask = (flat <= self.min_opacity) | bad
            dead = mask.nonzero(as_tuple=True)[0]
            alive = (~mask).nonzero(as_tuple=True)[0]
            n = len(dead)
            if n == 0 or len(alive) == 0:
                return 0
            eps = torch.finfo(torch.float32).eps
            probs = flat[alive] * weight[alive].clamp(min=0)
            probs = self._safe_probs(probs, flat[alive], "relocate")
            sampled = gsops._multinomial_sample(probs, n, replacement=True)
            sampled = alive[sampled]
            new_op, new_sc = compute_relocation(
                opacities=opacities[sampled],
                scales=torch.exp(params["scales"])[sampled],
                ratios=torch.bincount(sampled)[sampled] + 1, binoms=binoms)
            new_op = torch.clamp(new_op, max=1.0 - eps, min=self.min_opacity)

            def param_fn(name, p):
                if name == "opacities":
                    p[sampled] = torch.logit(new_op)
                elif name == "scales":
                    p[sampled] = torch.log(new_sc)
                p[dead] = p[sampled]
                return torch.nn.Parameter(p, requires_grad=p.requires_grad)

            def opt_fn(key, v):
                v[sampled] = 0
                return v

            gsops._update_param_with_optimizer(param_fn, opt_fn, params, optimizers)
            return n

        @torch.no_grad()
        def _add_weighted(self, params, optimizers, binoms, weight):
            cur = params["means"].shape[0]
            n = max(0, min(self.cap_max, int(1.05 * cur)) - cur)
            if n == 0:
                return 0
            opacities = torch.sigmoid(params["opacities"])
            eps = torch.finfo(torch.float32).eps
            flat = torch.nan_to_num(opacities.flatten(), nan=0.0,
                                    posinf=0.0, neginf=0.0)
            probs = flat * weight.clamp(min=0)
            probs = self._safe_probs(probs, flat, "add")
            sampled = gsops._multinomial_sample(probs, n, replacement=True)
            new_op, new_sc = compute_relocation(
                opacities=opacities[sampled],
                scales=torch.exp(params["scales"])[sampled],
                ratios=torch.bincount(sampled)[sampled] + 1, binoms=binoms)
            new_op = torch.clamp(new_op, max=1.0 - eps, min=self.min_opacity)

            def param_fn(name, p):
                if name == "opacities":
                    p[sampled] = torch.logit(new_op)
                elif name == "scales":
                    p[sampled] = torch.log(new_sc)
                return torch.nn.Parameter(torch.cat([p, p[sampled]]),
                                          requires_grad=p.requires_grad)

            def opt_fn(key, v):
                return torch.cat([v, torch.zeros((len(sampled), *v.shape[1:]),
                                                 device=v.device)])

            gsops._update_param_with_optimizer(param_fn, opt_fn, params, optimizers)
            return n

        def step_post_backward(self, params, optimizers, state, step, info,
                               lr, packed=False):
            self._update_state(params, state, info)
            if (self.refine_start_iter < step < self.refine_stop_iter
                    and step % self.refine_every == 0):
                binoms = state["binoms"]
                if not binoms.is_cuda:               # gsplat inits binoms on CPU;
                    binoms = binoms.to(params["means"].device)  # compute_relocation needs CUDA
                    state["binoms"] = binoms
                if state["err"] is not None:
                    w = state["err"] / state["count"].clamp(min=1.0)
                    nbad = int((~torch.isfinite(w)).sum())
                    if nbad:
                        # in ra để lần sau biết nguồn NaN nằm ở tín hiệu lỗi hay
                        # ở chính opacity — j209/c2_err chết mà không có manh mối
                        print(f"[errguided] step {step}: {nbad} phần tử err "
                              f"không hữu hạn, đã khử", flush=True)
                        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                    mw = float(w.mean())
                    w = w / mw if (mw == mw and mw > 0) else torch.zeros_like(w)
                else:
                    w = torch.ones(params["means"].shape[0],
                                   device=params["means"].device)
                weight = 1.0 + self.err_lambda * w
                if step <= self.refine_start_iter + self.refine_every:
                    print(f"[errguided] step {step} weight std={w.std().item():.3f} "
                          f"max={w.max().item():.2f} (uniform=>degenerate signal)",
                          flush=True)
                self._relocate_weighted(params, optimizers, binoms, weight)
                self._add_weighted(params, optimizers, binoms, weight)
                state["err"] = None
                state["count"] = None
                torch.cuda.empty_cache()
            inject_noise_to_position(params=params, optimizers=optimizers,
                                     state={}, scaler=lr * self.noise_lr)

    return ErrGuidedMCMC


def _absgrad_las_mcmc_class():
    """MCMC with a capped quota of AbsGrad-selected long-axis splits.

    J122-D keeps dead-Gaussian relocation, stochastic MCMC growth, SGLD noise,
    the production Eval3D loss and the 18M cap intact.  At each normal MCMC
    growth event, only ``las_quota`` of the available *net-new* slots is
    assigned to geometry-preserving long-axis splits.  Splitting one parent
    into two children consumes exactly one net slot; the remaining slots use
    gsplat's unmodified ``sample_add``.  Consequently the post-event count is
    exactly ``min(cap_max, floor(1.05 * count_before))`` and can never exceed
    the cap.
    """
    from gsplat.strategy import MCMCStrategy
    from gsplat.strategy import ops as gsops
    from gsplat.strategy.ops import inject_noise_to_position
    from gsplat.utils import normalized_quat_to_rotmat

    class AbsGradLASMCMC(MCMCStrategy):
        def __init__(self, las_quota=0.25, las_distance=0.45,
                     las_opacity_reduction=0.6, las_min_radius_px=2.0,
                     las_min_anisotropy=1.0, las_preflight_calls=1,
                     las_signal_every=10, **kw):
            super().__init__(**kw)
            if not (0.0 < float(las_quota) <= 1.0):
                raise ValueError("las_quota must be in (0, 1]")
            if not (0.0 < float(las_distance) < 1.0):
                raise ValueError("las_distance must be in (0, 1)")
            if not (0.0 < float(las_opacity_reduction) <= 1.0):
                raise ValueError("las_opacity_reduction must be in (0, 1]")
            self.las_quota = float(las_quota)
            self.las_distance = float(las_distance)
            self.las_opacity_reduction = float(las_opacity_reduction)
            self.las_min_radius_px = float(las_min_radius_px)
            self.las_min_anisotropy = float(las_min_anisotropy)
            self.las_preflight_calls = max(1, int(las_preflight_calls))
            self.las_signal_every = int(las_signal_every)
            if self.las_signal_every <= 0:
                raise ValueError("las_signal_every must be positive")

        def initialize_state(self):
            st = super().initialize_state()
            st.update(grad2d=None, count=None, radii=None, aux_calls=0,
                      las_total=0, mcmc_total=0)
            return st

        def step_pre_backward(self, params, optimizers, state, step, info):
            # The production Eval3D graph intentionally has no means2d
            # retain_grad.  record_absgrad() receives the isolated auxiliary
            # projected-UT result after the production optimizer step.
            pass

        @torch.no_grad()
        def record_absgrad(self, params, state, info, step):
            required = ("means2d", "radii", "width", "height", "n_cameras")
            missing = [key for key in required if key not in info]
            if missing:
                raise RuntimeError(
                    f"J122D_ABSGRAD_PREFLIGHT_FAIL step={step} missing={missing}")
            means2d = info["means2d"]
            if not hasattr(means2d, "absgrad"):
                raise RuntimeError(
                    "J122D_ABSGRAD_PREFLIGHT_FAIL "
                    f"step={step} means2d.absgrad is missing; auxiliary render "
                    "must use with_ut=True, with_eval3d=False, absgrad=True")
            grads = means2d.absgrad.detach().clone()
            if grads.shape != means2d.shape or grads.shape[-1] != 2:
                raise RuntimeError(
                    "J122D_ABSGRAD_PREFLIGHT_FAIL "
                    f"step={step} bad_shape={tuple(grads.shape)} "
                    f"means2d={tuple(means2d.shape)}")
            if not torch.isfinite(grads).all():
                n_bad = int((~torch.isfinite(grads)).sum().item())
                raise RuntimeError(
                    "J122D_ABSGRAD_PREFLIGHT_FAIL "
                    f"step={step} nonfinite={n_bad}")

            radii = info["radii"]
            visible = (radii > 0).all(dim=-1)
            if not visible.any():
                raise RuntimeError(
                    f"J122D_ABSGRAD_PREFLIGHT_FAIL step={step} visible=0")
            # Match gsplat DefaultStrategy's pixel -> normalized-screen
            # conversion before taking the homodirectional gradient norm.
            grads[..., 0] *= float(info["width"]) / 2.0 * int(info["n_cameras"])
            grads[..., 1] *= float(info["height"]) / 2.0 * int(info["n_cameras"])
            visible_norms = grads[visible].norm(dim=-1)
            signal_sum = float(visible_norms.sum().item())
            next_call = int(state["aux_calls"]) + 1
            if (not math.isfinite(signal_sum)
                    or (next_call <= self.las_preflight_calls
                        and signal_sum <= torch.finfo(grads.dtype).eps)):
                raise RuntimeError(
                    "J122D_ABSGRAD_PREFLIGHT_FAIL "
                    f"step={step} call={next_call} signal_sum={signal_sum}")
            if signal_sum <= torch.finfo(grads.dtype).eps:
                print(f"[j122d] zero AbsGrad view skipped step={step}", flush=True)
                return

            n = int(params["means"].shape[0])
            if state["grad2d"] is None or state["grad2d"].numel() != n:
                state["grad2d"] = torch.zeros(n, device=grads.device)
                state["count"] = torch.zeros(n, device=grads.device)
                state["radii"] = torch.zeros(n, device=grads.device)
            # The trainer renders one camera per iteration.  Keep the indexing
            # general for dense [C,N,*] outputs nevertheless.
            gs_ids = torch.where(visible)[-1]
            state["grad2d"].index_add_(0, gs_ids, visible_norms)
            state["count"].index_add_(
                0, gs_ids, torch.ones_like(visible_norms, dtype=torch.float32))
            radius_px = radii[visible].max(dim=-1).values.float()
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids], radius_px)
            state["aux_calls"] = next_call
            if next_call <= self.las_preflight_calls:
                pos = visible_norms[visible_norms > 0]
                print(
                    "J122D_ABSGRAD_PREFLIGHT_OK "
                    f"step={step} call={next_call}/{self.las_preflight_calls} "
                    f"visible={int(visible.sum())}/{n} "
                    f"positive={int(pos.numel())} sum={signal_sum:.6g} "
                    f"mean={float(visible_norms.mean()):.6g} "
                    f"max={float(visible_norms.max()):.6g}", flush=True)

        @torch.no_grad()
        def _long_axis_split(self, params, optimizers, selected):
            """Replace K parents by 2K ImprovedGS LAS children (net +K)."""
            k = int(selected.numel())
            if k == 0:
                return 0
            n = int(params["means"].shape[0])
            keep_mask = torch.ones(n, dtype=torch.bool,
                                   device=params["means"].device)
            keep_mask[selected] = False
            rest = torch.where(keep_mask)[0]

            stds = torch.exp(params["scales"].detach()[selected])
            max_values, max_indices = torch.max(stds, dim=1, keepdim=True)
            axis_mask = torch.zeros_like(stds).scatter_(1, max_indices, 1.0)
            axis_offsets = stds * axis_mask * (3.0 * self.las_distance)
            axis_offsets = torch.cat([axis_offsets, -axis_offsets], dim=0)
            rotmats = normalized_quat_to_rotmat(
                F.normalize(params["quats"].detach()[selected], dim=-1)
            ).repeat(2, 1, 1)
            parent_means = params["means"].detach()[selected].repeat(2, 1)
            child_means = (torch.bmm(rotmats, axis_offsets.unsqueeze(-1))
                           .squeeze(-1) + parent_means)

            d2 = self.las_distance * self.las_distance
            rate_h = math.sqrt(max(1.0 - d2, 1e-6))
            rate_w = max(1.0 - self.las_distance, 1e-6)
            child_scales = stds.scatter(
                1, max_indices, max_values * rate_w / rate_h
            ).repeat(2, 1) * rate_h
            eps = torch.finfo(params["opacities"].dtype).eps
            child_opacities = (
                torch.sigmoid(params["opacities"].detach()[selected])
                * self.las_opacity_reduction
            ).clamp(min=eps, max=1.0 - eps).repeat(2)

            def param_fn(name, p):
                repeats = [2] + [1] * (p.dim() - 1)
                if name == "means":
                    children = child_means
                elif name == "scales":
                    children = torch.log(child_scales)
                elif name == "opacities":
                    children = torch.logit(child_opacities)
                else:
                    children = p.detach()[selected].repeat(repeats)
                return torch.nn.Parameter(
                    torch.cat([p.detach()[rest], children], dim=0),
                    requires_grad=p.requires_grad)

            def opt_fn(key, value):
                child_zeros = torch.zeros(
                    (2 * k, *value.shape[1:]), device=value.device,
                    dtype=value.dtype)
                return torch.cat([value[rest], child_zeros], dim=0)

            gsops._update_param_with_optimizer(
                param_fn, opt_fn, params, optimizers)
            return k

        def step_post_backward(self, params, optimizers, state, step, info,
                               lr, packed=False):
            if int(state["aux_calls"]) < self.las_preflight_calls:
                # Do not wait hundreds of steps to discover a dead signal.
                deadline = ((self.las_preflight_calls - 1)
                            * self.las_signal_every)
                if step >= deadline:
                    raise RuntimeError(
                        "J122D_ABSGRAD_PREFLIGHT_FAIL "
                        f"step={step} successful_calls={state['aux_calls']} "
                        f"required={self.las_preflight_calls}")

            state["binoms"] = state["binoms"].to(params["means"].device)
            if (self.refine_start_iter < step < self.refine_stop_iter
                    and step % self.refine_every == 0):
                n_before = int(params["means"].shape[0])
                dead_mask = (torch.sigmoid(params["opacities"].flatten())
                             <= self.min_opacity)
                n_relocated = super()._relocate_gs(
                    params, optimizers, state["binoms"])
                if state["grad2d"] is not None and dead_mask.any():
                    # Relocated slots no longer describe the Gaussian whose
                    # auxiliary signal occupied this index.
                    state["grad2d"][dead_mask] = 0
                    state["count"][dead_mask] = 0
                    state["radii"][dead_mask] = 0

                growth_target = min(self.cap_max, int(1.05 * n_before))
                n_new = max(0, growth_target - n_before)
                n_las_wanted = int(math.floor(n_new * self.las_quota + 0.5))
                selected = torch.empty(
                    0, dtype=torch.long, device=params["means"].device)
                score_mean = score_max = 0.0
                eligible_count = 0
                if n_las_wanted > 0:
                    if state["grad2d"] is None:
                        raise RuntimeError(
                            f"J122D_ABSGRAD_MISSING_AT_REFINE step={step}")
                    scores = state["grad2d"] / state["count"].clamp_min(1.0)
                    scales = torch.exp(params["scales"].detach())
                    anisotropy = (scales.max(dim=-1).values
                                  / scales.min(dim=-1).values.clamp_min(1e-12))
                    eligible = ((state["count"] > 0) & torch.isfinite(scores)
                                & (scores > 0)
                                & (state["radii"] >= self.las_min_radius_px)
                                & (anisotropy >= self.las_min_anisotropy))
                    candidate_ids = torch.where(eligible)[0]
                    eligible_count = int(candidate_ids.numel())
                    n_select = min(n_las_wanted, eligible_count)
                    if n_select > 0:
                        candidate_scores = scores[candidate_ids]
                        top = torch.topk(candidate_scores, n_select,
                                         largest=True, sorted=False).indices
                        selected = candidate_ids[top]
                        score_mean = float(scores[selected].mean().item())
                        score_max = float(scores[selected].max().item())

                n_las = self._long_axis_split(params, optimizers, selected)
                # Any LAS deficit (including no eligible candidates) falls back
                # to vanilla MCMC so the count trajectory stays exactly matched.
                n_mcmc = n_new - n_las
                if n_mcmc > 0:
                    gsops.sample_add(
                        params=params, optimizers=optimizers, state={},
                        n=n_mcmc, binoms=state["binoms"],
                        min_opacity=self.min_opacity)
                n_after = int(params["means"].shape[0])
                if n_after != growth_target or n_after > self.cap_max:
                    raise RuntimeError(
                        "J122D_CAP_INVARIANT_FAIL "
                        f"step={step} before={n_before} target={growth_target} "
                        f"after={n_after} cap={self.cap_max} las={n_las} "
                        f"mcmc={n_mcmc}")
                state["las_total"] += n_las
                state["mcmc_total"] += n_mcmc
                print(
                    "J122D_REFINE "
                    f"step={step} before={n_before} relocated={n_relocated} "
                    f"slots={n_new} las_wanted={n_las_wanted} "
                    f"eligible={eligible_count} las={n_las} mcmc={n_mcmc} "
                    f"after={n_after} cap={self.cap_max} "
                    f"score_mean={score_mean:.6g} score_max={score_max:.6g} "
                    f"las_total={state['las_total']} "
                    f"mcmc_total={state['mcmc_total']}", flush=True)
                state["grad2d"] = torch.zeros(
                    n_after, device=params["means"].device)
                state["count"] = torch.zeros(
                    n_after, device=params["means"].device)
                state["radii"] = torch.zeros(
                    n_after, device=params["means"].device)
                torch.cuda.empty_cache()

            inject_noise_to_position(
                params=params, optimizers=optimizers, state={},
                scaler=lr * self.noise_lr)

    return AbsGradLASMCMC


def _capped_default_class():
    """gsplat DefaultStrategy (= adaptive density control của 3DGS gốc) + TRẦN CỨNG.

    Vì sao cần lớp này. Cả chiến dịch vòng 2 chỉ đo trong họ MCMC; ADC — cách
    chọn/nhân Gaussian của chính bài 3DGS 2023, và là nền của AbsGS lẫn "Revising
    Densification" — chưa từng chạy một lần nào. Không có nó thì mọi câu "MCMC
    tốt hơn" đều là so MCMC với MCMC.

    Ba baseline nằm gọn trong MỘT lớp của gsplat, bật bằng cờ:
      absgrad=False, revised_opacity=False  -> 3DGS gốc (Kerbl 2023)
      absgrad=True                          -> AbsGS / Gradient-tuyệt-đối (2404.10484)
      revised_opacity=True                  -> Revising Densification (Bulò 2404.06109)

    TRẦN CỨNG: ADC không có cap_max — nó mọc tới khi nào ngưỡng gradient hết bắt.
    Trên cảnh lớn 643 ảnh, mọc tự do là canh bạc VRAM, mà một lần OOM là mất cả
    đêm. Khi n chạm trần ta chỉ TẮT MỌC, vẫn cho tỉa/reset chạy tiếp — như vậy
    trần là một giới hạn tài nguyên, không phải một luật động lực học mới.

    CHẶN SỚM, không chặn đúng vạch. Kiểm tra "n >= trần" xảy ra TRƯỚC một lượt
    mọc, nên nếu để tới sát vạch thì lượt cuối vẫn vọt qua. Khói thử 15/08 đo
    được mức mọc mỗi lần refine trên chính cảnh này: +17%, +23%, +18%. Tệ hơn
    nữa, `_update_param_with_optimizer` dựng tensor mới trong khi tensor cũ còn
    sống ⇒ đỉnh VRAM tức thời gấp ~2 lần trạng thái ổn định. Cộng hai điều đó,
    vọt 25% ở mức 48M là 60M ổn định ≈ 72 GB, tức ~144 GB lúc realloc — chết
    ngay trên card 143 GB. Nên đóng cửa ở trần/1.5, còn dư chỗ cho trọn một lượt
    vọt.
    """
    from gsplat.strategy import DefaultStrategy

    #: hệ số an toàn giữa "ngưỡng đóng cửa" và trần thật (xem docstring)
    _CAP_MARGIN = 1.5

    class CappedDefault(DefaultStrategy):
        _cap = 0
        _cap_hit = False
        #: Pixel-GS (2403.15530) — trọng số theo diện tích chiếu
        _pixel_gs = False
        #: Taming-3DGS (2406.15643) — ngân sách cuối cố định (0 = tắt)
        _budget = 0
        _budget_n0 = None
        _budget_said = False

        def _update_state(self, params, state, info, packed=False):
            """Bản Pixel-GS. Tắt cờ ⇒ gọi thẳng bản gsplat, không lệch một bit.

            3DGS gốc lấy TRUNG BÌNH ĐỀU gradient qua các view thấy được
            Gaussian: mỗi view một phiếu, bất kể nó chiếm 4 pixel hay 4000.
            Pixel-GS lập luận rằng đó chính là chỗ Gaussian nền lớn-và-mờ bị bỏ
            sót — chúng phủ rất nhiều pixel nên đóng góp nhiều sai số, nhưng
            phiếu của chúng bị pha loãng bởi những view nhìn nghiêng chỉ thấy
            một mẩu. Ở đây đổi sang TRUNG BÌNH CÓ TRỌNG SỐ theo diện tích chiếu
            (∝ bán kính²), tức mỗi view bỏ phiếu theo số pixel nó thật sự vẽ.
            """
            if not self._pixel_gs:
                return super()._update_state(params, state, info, packed=packed)

            if self.absgrad:
                grads = info[self.key_for_gradient].absgrad.clone()
            else:
                grads = info[self.key_for_gradient].grad.clone()
            grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
            grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

            n_gaussian = len(list(params.values())[0])
            if state["grad2d"] is None:
                state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
            if state["count"] is None:
                state["count"] = torch.zeros(n_gaussian, device=grads.device)

            if packed:
                gs_ids = info["gaussian_ids"]
                radii = info["radii"].max(dim=-1).values
            else:
                sel = (info["radii"] > 0.0).all(dim=-1)
                gs_ids = torch.where(sel)[1]
                grads = grads[sel]
                radii = info["radii"][sel].max(dim=-1).values
            # trọng số = diện tích chiếu (bỏ hằng số π: chỉ tỉ lệ mới có nghĩa
            # vì grad2d và count chia cho nhau ngay sau đó). Sàn 1 để view nào
            # cũng còn ít nhất một phiếu.
            w = radii.float().clamp(min=1.0) ** 2
            state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1) * w)
            state["count"].index_add_(0, gs_ids, w)

        def _apply_budget(self, params, state, step):
            """Taming-3DGS: đổi NGƯỠNG mỗi lần refine để trúng ngân sách đã định.

            3DGS gốc cố định ngưỡng gradient rồi để số Gaussian rơi vào đâu thì
            rơi — trên cảnh lớn đó là canh bạc VRAM (`b1_adc` dừng ở 21,3M,
            `f1_adc` trên lfls phi qua 32M). Taming lật ngược: ngân sách cuối là
            thứ ta CHỌN, còn ngưỡng là thứ suy ra. Mỗi lần refine tính xem còn
            được đẻ bao nhiêu để đi đúng đường tới đích, rồi đặt ngưỡng bằng
            đúng phần tử thứ k lớn nhất ⇒ chọn ra top-k ứng viên tốt nhất.

            Cách này TÁI DÙNG NGUYÊN VẸN luật clone/split/prune của gsplat —
            chỉ một con số `grow_grad2d` bị thay — nên khác biệt đo được là
            khác biệt của LUẬT NGÂN SÁCH, không lẫn với khác biệt cài đặt.
            """
            n = params["means"].shape[0]
            if self._budget_n0 is None:
                self._budget_n0 = n
            span = max(1, self.refine_stop_iter - self.refine_start_iter)
            frac = min(1.0, max(0.0, (step - self.refine_start_iter) / span))
            target = self._budget_n0 + (self._budget - self._budget_n0) * frac
            k = int(target) - n
            if k <= 0:
                self.grow_grad2d = float("inf")     # đủ rồi, lượt này không đẻ
                return
            score = state["grad2d"] / state["count"].clamp_min(1)
            score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
            # clone tạo 1 con, split tạo 1 con ⇒ k ứng viên ≈ k Gaussian mới
            k = min(k, score.numel() - 1)
            thr = torch.topk(score, k, largest=True).values[-1]
            self.grow_grad2d = float(thr.item())
            if not self._budget_said:
                print(f"[densify] Taming: ngân sách {self._budget}, bước {step} "
                      f"n={n} -> đích {int(target)}, chọn top-{k}, "
                      f"ngưỡng suy ra={self.grow_grad2d:.3e}", flush=True)
                self._budget_said = True

        def step_post_backward(self, params, optimizers, state, step, info,
                               lr=None, packed=False):
            # `lr` là của MCMC (biên độ nhiễu vị trí); ADC không dùng. Nuốt nó ở
            # đây để vòng lặp train gọi chung một chữ ký cho mọi strategy.
            n = params["means"].shape[0]
            # Taming đặt ngưỡng TRƯỚC, để trần VRAM bên dưới vẫn phủ quyết được
            if (self._budget > 0 and step > self.refine_start_iter
                    and step % self.refine_every == 0
                    and step < self.refine_stop_iter
                    and state.get("grad2d") is not None):
                self._apply_budget(params, state, step)
            gate = int(self._cap / _CAP_MARGIN) if self._cap > 0 else 0
            if gate > 0 and n >= gate:
                if not self._cap_hit:
                    print(f"[densify] ADC dừng mọc ở bước {step}: n={n} chạm "
                          f"ngưỡng {gate} (= trần {self._cap} / {_CAP_MARGIN}) "
                          f"— từ đây CHỈ TỈA. Baseline BỊ CẮT, đọc điểm phải "
                          f"kèm cảnh báo này.", flush=True)
                    self._cap_hit = True
                grow = self.grow_grad2d
                self.grow_grad2d = float("inf")     # không ứng viên nào vượt nổi
                try:
                    super().step_post_backward(params, optimizers, state, step,
                                               info, packed=packed)
                finally:
                    self.grow_grad2d = grow
                return
            super().step_post_backward(params, optimizers, state, step, info,
                                       packed=packed)

    return CappedDefault


def _so3_exp(w):
    """Axis-angle (3,) -> rotation matrix (3,3), safe near zero."""
    theta = w.norm() + 1e-8
    k = w / theta
    K = torch.stack([
        torch.stack([torch.zeros_like(k[0]), -k[2], k[1]]),
        torch.stack([k[2], torch.zeros_like(k[0]), -k[0]]),
        torch.stack([-k[1], k[0], torch.zeros_like(k[0])]),
    ])
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def _rotmat_to_quat(R):
    """(3,3) -> quat wxyz; trace branch only (fine for near-identity)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    qw = torch.sqrt(torch.clamp(1 + tr, min=1e-8)) / 2
    return torch.stack([qw,
                        (R[2, 1] - R[1, 2]) / (4 * qw),
                        (R[0, 2] - R[2, 0]) / (4 * qw),
                        (R[1, 0] - R[0, 1]) / (4 * qw)])


def _qmul(r, q):
    """Left-multiply unit quat r (4,) onto quats q (N,4), wxyz."""
    rw, rx, ry, rz = r
    qw, qx, qy, qz = q.unbind(-1)
    return torch.stack([
        rw * qw - rx * qx - ry * qy - rz * qz,
        rw * qx + rx * qw + ry * qz - rz * qy,
        rw * qy - rx * qz + ry * qw + rz * qx,
        rw * qz + rx * qy - ry * qx + rz * qw,
    ], dim=-1)


class PoseOpt(torch.nn.Module):
    """Per-train-view SE(3) residual on w2c, init zero. gsplat's UT backward
    has no viewmat gradients, so the residual is applied as the EQUIVALENT
    world-space transform M = w2c^-1 @ dT @ w2c on means/quats — those DO get
    gradients. Test/holdout poses are never adjusted (organizers' frame);
    an L2 prior keeps residuals a trust region."""

    def __init__(self, n):
        super().__init__()
        self.res = torch.nn.Parameter(torch.zeros(n, 6))

    def world_correction(self, viewmat, idx):  # viewmat (4,4) w2c, no grad
        xi = self.res[idx]
        R = _so3_exp(xi[:3])
        bot = torch.tensor([[0.0, 0.0, 0.0, 1.0]],
                           device=viewmat.device, dtype=viewmat.dtype)
        dT = torch.cat([torch.cat([R, xi[3:].unsqueeze(-1)], dim=1), bot], dim=0)
        Rv, tv = viewmat[:3, :3], viewmat[:3, 3]
        inv = torch.cat([torch.cat([Rv.T, (-Rv.T @ tv).unsqueeze(-1)], dim=1),
                         bot], dim=0)  # w2c^-1 (rigid)
        return inv @ dT @ viewmat


def make_radial(scene, cfg, device):
    """Radial-coeff tensor for UT rendering, with optional k1/k2 override
    (COLMAP's SIMPLE_RADIAL k1 may be imperfect; sweep via --k1_scale/--k2).

    Returning None keeps the classic EWA path (local affine approximation of the
    projection).  --force_ut returns an all-zero coeff tensor instead, which
    switches gsplat to with_ut/with_eval3d on a *distortion-free* camera: same
    pixels, but the projection is no longer linearised.  That isolates the
    projection error, which the round-2 diagnosis measured as the single largest
    remaining defect (frame corners carry 1.6-2.5x the MSE of the frame centre
    at equal image texture, in both nadir and oblique views)."""
    forced = int(getattr(cfg, "force_ut", 0))
    if np.abs(scene.dist).max() <= 1e-12:
        return torch.zeros(1, 6, device=device) if forced else None
    if not getattr(scene, "distorted", False):
        # 05/09: scene có méo nhưng train ở khung UNDISTORT (loader đã undistort ảnh) → render pinhole, KHÔNG áp méo UT
        # (render_test sẽ redistort có đệm sau). Trước đây trả k1 ở đây → render bị méo hai lần (lệch ~40 px ở r=0,5).
        return torch.zeros(1, 6, device=device) if forced else None
    radial = torch.zeros(1, 6, device=device)
    radial[0, 0] = float(scene.dist[0]) * getattr(cfg, "k1_scale", 1.0)
    radial[0, 1] = getattr(cfg, "k2", 0.0)
    return radial


def apply_bilagrid(bil_grids, colors, image_ids, meshgrid_cache={}):
    B, H, W, _ = colors.shape
    key = (H, W, colors.device)
    if key not in meshgrid_cache:
        gy, gx = torch.meshgrid(
            (torch.arange(H, device=colors.device) + 0.5) / H,
            (torch.arange(W, device=colors.device) + 0.5) / W,
            indexing="ij",
        )
        meshgrid_cache[key] = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    xy = meshgrid_cache[key].expand(B, -1, -1, -1)
    return bg_slice(bil_grids, xy, colors, image_ids)["rgb"]


def _orbit_center(viewmats):
    """Least-squares intersection of the camera optical axes = the 3D point the
    orbital rig is looking at (the subject, e.g. the BTS tower). OpenCV convention
    (+z forward). viewmats: (N,4,4) world->cam. Point p minimizing the summed
    squared distance to every optical-axis ray."""
    dev = viewmats.device
    I3 = torch.eye(3, device=dev, dtype=torch.float64)
    fwd = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=torch.float64)
    A = torch.zeros(3, 3, device=dev, dtype=torch.float64)
    b = torch.zeros(3, device=dev, dtype=torch.float64)
    for i in range(viewmats.shape[0]):
        R = viewmats[i, :3, :3].double()
        t = viewmats[i, :3, 3].double()
        C = -R.T @ t                       # camera centre (world)
        d = R.T @ fwd
        d = d / (d.norm() + 1e-12)         # optical axis (world)
        P = I3 - torch.outer(d, d)         # projector orthogonal to the ray
        A += P
        b += P @ C
    p = torch.linalg.solve(A + 1e-6 * I3, b)
    return p.float()


def _scene_depth_median(scene, q=50.0):
    """Độ sâu (phân vị q, mặc định trung vị) của cảnh nhìn từ camera train — quy px ↔ đơn vị thế giới."""
    _zs = [u[:, 2] for u in scene.sparse_uvd if len(u)]
    if _zs:
        return float(np.percentile(np.concatenate(_zs), q)), "track SfM"
    # GauU-Scene: COLMAP do ContextCapture xuất, points3D.txt RỖNG nên
    # images.txt không có tương ứng 2D-3D -> sparse_uvd rỗng ở mọi ảnh.
    # Chiếu thẳng đám mây khởi tạo vào một mẫu camera để lấy độ sâu.
    rng = np.random.default_rng(0)
    P = scene.points.xyz
    P = P[rng.choice(len(P), size=min(20000, len(P)), replace=False)]
    zs = []
    for m in [scene.train_metas[i] for i in
              rng.choice(len(scene.train_metas), size=min(32, len(scene.train_metas)),
                         replace=False)]:
        z = ((m.R() @ P.T).T + m.tvec)[:, 2]
        zs.append(z[z > 1e-6])
    zs = np.concatenate(zs) if zs else np.array([])
    return (float(np.median(zs)) if zs.size else 0.0), "chiếu đám mây init (không có track)"


def _install_mcmc_opacity_probe(min_opacity, log_every=5, invis_gsd=0.0):
    """Chặn crash "alive rỗng" của gsplat MCMC + in phân bố opacity mỗi lần refine.

    invis_gsd > 0 (--relocate_invisible, 02/09): coi là CHẾT cả Gaussian "vô hình" —
    ở chế độ antialiased, alpha đỉnh = opacity × ab/√((a²+0.3)(b²+0.3)) (a,b = std 2D
    theo px); raster bỏ qua splat có alpha < 1/255. Gaussian đã sụp dưới ~0,1 px
    (scale_reg kéo xuống, không còn gradient render kéo lên) thì không render, không
    có gradient, opacity vẫn > min_opacity nên MCMC gốc không bao giờ relocate: đo
    trên f1_r_c12_s15k được 42 % của 12M nằm ở trạng thái này. Footprint lấy trường
    hợp lớn nhất (hai trục lớn) quy ra px bằng GSD ⇒ chỉ đánh dấu cái vô hình ở MỌI
    hướng nhìn.

    `gsplat/strategy/ops.py::relocate` lấy `probs = opacities[~dead]`. Nếu MỌI
    Gaussian đều <= min_opacity thì probs rỗng, torch.multinomial launch kernel với
    block dim = 0 -> CUDA "invalid configuration argument". Đây là cách j200 chết ở
    step ~7600 trên SMBU: KHÔNG phải lỗi hạ tầng mà là opacity sụp toàn cục.

    Paper của chính bộ data (GauU-Scene V2, arXiv 2404.04880, Fig.6) đo được 2/3
    Gaussian nằm quanh logit -5 — sát đúng ngưỡng min_opacity=0.005 (logit -5.29).
    Vanilla 3DGS chỉ prune chúng nên vô hại; MCMC gọi relocate nên chết.

    Bản vá KHÔNG đổi kết quả khi mọi thứ bình thường: chỉ bỏ qua đúng lần refine
    không còn Gaussian sống, và in thống kê. Đường error-guided/LAS trong file này
    đã tự có guard (`_relocate_weighted`), đây là vá cho đường MCMCStrategy gốc.
    """
    import gsplat.strategy.mcmc as _mcmc
    import gsplat.strategy.ops as _ops
    if getattr(_ops, "_vt_probe_installed", False):
        return
    _orig = _ops.relocate
    ctr = {"n": 0}

    def relocate(params, optimizers, state, mask, binoms, min_opacity=min_opacity):
        op = torch.sigmoid(params["opacities"].detach().flatten().float())
        n_tot = op.numel()
        if invis_gsd > 0:
            s3, _ = torch.sort(torch.exp(params["scales"].detach().float()) / invis_gsd, dim=-1)
            a, b = s3[:, 2], s3[:, 1]
            comp = a * b / torch.sqrt((a * a + 0.3) * (b * b + 0.3))
            invis = (op * comp) < (1.0 / 255.0)
            n_inv = int((invis & ~mask).sum())
            mask = mask | invis
            if log_every <= 0 or ctr["n"] % log_every == 0:
                print("[opacity] relocate_invisible: +%d (%.1f%%) Gaussian vô hình"
                      % (n_inv, 100.0 * n_inv / max(n_tot, 1)), flush=True)
        n_dead = int(mask.sum())
        n_alive = n_tot - n_dead
        ctr["n"] += 1
        if n_alive == 0 or log_every <= 0 or ctr["n"] % log_every == 1:
            # torch.quantile giới hạn số phần tử -> lấy mẫu khi Gaussian đã đông
            s = op if n_tot <= 1_000_000 else op[
                torch.randint(0, n_tot, (1_000_000,), device=op.device)]
            q = torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99], device=op.device)
            p = torch.quantile(s, q).tolist()
            print("[opacity] refine#%d n=%d dead=%d (%.1f%%) alive=%d | "
                  "p01=%.4f p10=%.4f p50=%.4f p90=%.4f p99=%.4f"
                  % (ctr["n"], n_tot, n_dead, 100.0 * n_dead / max(n_tot, 1),
                     n_alive, *p), flush=True)
        if n_alive == 0:
            print("[opacity] ALIVE=0 — không còn Gaussian nào để nhân bản; BỎ QUA "
                  "lần relocate này thay vì chết. Opacity đã sụp toàn cục.", flush=True)
            return
        return _orig(params=params, optimizers=optimizers, state=state,
                     mask=mask, binoms=binoms, min_opacity=min_opacity)

    # mcmc.py làm `from .ops import relocate` nên phải vá CẢ HAI namespace.
    _ops.relocate = relocate
    _mcmc.relocate = relocate
    _ops._vt_probe_installed = True
    print("[opacity] probe MCMC đã cài (guard alive=0, thống kê mỗi %d lần refine)"
          % log_every, flush=True)


# --------------------------------------------------------------------------
def train(cfg):
    device = "cuda"
    set_seed(cfg.seed)
    t0 = time.time()

    scene = SceneData(cfg.scene_dir, distorted=bool(getattr(cfg, "distorted", 0)),
                      mask_dir=getattr(cfg, "mask_dir", "") or None,
                      holdout_every=getattr(cfg, "holdout_every", 0),
                      holdout_offset=getattr(cfg, "holdout_offset", None),
                      multi_camera=getattr(cfg, "multi_camera", "auto"),
                      camera_tol_px=getattr(cfg, "camera_tol_px", None),
                      train_list=getattr(cfg, "train_list", "") or None)
    n_imgs = len(scene.train_metas)
    W, H = scene.width, scene.height
    if float(getattr(cfg, "scene_scale", 0.0)) > 0:
        # train_list ít view → scene_scale (bán kính cụm camera) sụp về 0 ⇒ lr means & noise MCMC = 0;
        # ép bằng giá trị của cảnh đầy đủ để recipe so sánh được (x53, 02/09)
        scene.scene_scale = float(cfg.scene_scale)
    print(f"[scene] {cfg.scene_dir}: {n_imgs} train imgs, {len(scene.test_poses)} test poses, "
          f"{scene.points.xyz.shape[0]} pts, scale={scene.scene_scale:.2f}, dist k={scene.dist[0]:.4f}")

    os.makedirs(cfg.result_dir, exist_ok=True)
    with open(os.path.join(cfg.result_dir, "config.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2)

    # data tensors (images stay on CPU as uint8)
    # 26/08: KHÔNG torch.stack — stack là bản SAO thứ hai của 404 ảnh 21 MP (+25 GB RSS);
    # from_numpy dùng chung bộ nhớ với scene.images. Với cgroup 256 GiB trên H200 đây là
    # khác biệt giữa 4 và 6 trainer f1 chạy song song. Mọi chỗ dùng chỉ index/iterate.
    images_u8 = [torch.from_numpy(im) for im in scene.images]  # list N × (H,W,3) uint8, zero-copy
    masks_t = [torch.from_numpy(m) if m is not None else None for m in scene.masks]
    viewmats = torch.from_numpy(scene.w2c).float().to(device)
    K = torch.from_numpy(scene.K).float().to(device)
    Ks = K[None]
    uvds = [torch.from_numpy(x).to(device) for x in scene.sparse_uvd]
    # --depth_flat_thr (26/08): bỏ điểm track COLMAP nằm ở vùng PHẲNG của ảnh (nước):
    # dw01 đo được +0,15 tổng nhưng worst-10 −1,14 — track trên mặt nước là rác.
    _dft = float(getattr(cfg, "depth_flat_thr", 0.0))
    if _dft > 0 and cfg.depth_weight > 0:
        import cv2 as _cv2
        _kept = _tot = 0
        for _i in range(len(uvds)):
            if uvds[_i].shape[0] == 0:
                continue
            _g = _cv2.cvtColor(images_u8[_i].numpy(), _cv2.COLOR_RGB2GRAY).astype(np.float32)
            _s = max(1, min(_g.shape) // 660)          # ~ /4 so với f2
            _g = _cv2.resize(_g, (_g.shape[1] // _s, _g.shape[0] // _s), interpolation=_cv2.INTER_AREA)
            _mag = _cv2.GaussianBlur(np.abs(_cv2.Sobel(_g, _cv2.CV_32F, 1, 0))
                                     + np.abs(_cv2.Sobel(_g, _cv2.CV_32F, 0, 1)), (0, 0), 4)
            _uv = uvds[_i][:, :2].cpu().numpy() / _s
            _x = np.clip(_uv[:, 0].astype(int), 0, _mag.shape[1] - 1)
            _y = np.clip(_uv[:, 1].astype(int), 0, _mag.shape[0] - 1)
            _ok = torch.from_numpy(_mag[_y, _x] >= _dft).to(device)
            _tot += int(uvds[_i].shape[0]); _kept += int(_ok.sum())
            uvds[_i] = uvds[_i][_ok]
        print(f"[depth_flat] thr={_dft}: giữ {_kept}/{_tot} điểm track ({100*_kept/max(_tot,1):.1f}%)", flush=True)

    # --vpc_weight (26/08, "virtual-pose photometric consistency", PGSR/HBSplat-style, KHÔNG cần GT):
    # render ở pose ẢO giữa 2 frame train kề (đúng regime pose test = nửa bước bay), warp 2 ảnh train
    # kề vào pose ảo qua depth render, phạt |render_v − mean(warp)| ở vùng 2 warp đồng thuận. Gradient đi
    # qua cả màu lẫn depth (grid_sample khả vi theo lưới) ⇒ ép hình học đúng GIỮA các view.
    _vpc_w = float(getattr(cfg, "vpc_weight", 0.0))
    if _vpc_w > 0:
        from dataset import frame_index as _fidx
        _fr = [_fidx(m.name) for m in scene.train_metas]
        _ordr = sorted(range(len(_fr)), key=lambda i: (_fr[i] if _fr[i] is not None else i))
        _pos = {v: k for k, v in enumerate(_ordr)}
        def _vpc_nbr(i):
            k = _pos[i]
            return _ordr[k + 1] if k + 1 < len(_ordr) else _ordr[k - 1]
        _vpc_s = float(getattr(cfg, "vpc_scale", 0.5))
        _vpc_tau = float(getattr(cfg, "vpc_tau", 0.08))
        print(f"[vpc] weight={_vpc_w} scale={_vpc_s} tau={_vpc_tau} prob={getattr(cfg,'vpc_prob',1.0)}", flush=True)

        def _rot_mid(Ri, Rj):
            D = Ri.T @ Rj
            cos = ((D[0, 0] + D[1, 1] + D[2, 2] - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
            th = torch.acos(cos)
            ax = torch.stack([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]]) / (2 * torch.sin(th) + 1e-9)
            v = ax * (th / 2)
            Kx = torch.stack([torch.stack([0 * v[0], -v[2], v[1]]), torch.stack([v[2], 0 * v[0], -v[0]]), torch.stack([-v[1], v[0], 0 * v[0]])])
            return Ri @ torch.linalg.matrix_exp(Kx)

        def _vpc_loss(i, sh_deg):
            j = _vpc_nbr(i)
            Ri, Rj = viewmats[i, :3, :3], viewmats[j, :3, :3]
            ci = -Ri.T @ viewmats[i, :3, 3]; cj = -Rj.T @ viewmats[j, :3, 3]
            Rv = _rot_mid(Ri, Rj); cv_ = 0.5 * (ci + cj)
            w2c_v = torch.eye(4, device=device); w2c_v[:3, :3] = Rv; w2c_v[:3, 3] = -Rv @ cv_
            Wv, Hv = int(round(W * _vpc_s)), int(round(H * _vpc_s))
            Kv = Ks[:1].clone(); Kv[:, 0, :] *= Wv / float(W); Kv[:, 1, :] *= Hv / float(H)
            rend, _, _ = rasterize(splats, w2c_v[None], Kv, Wv, Hv, sh_deg, cfg, render_depth=True, radial_coeffs=radial)
            col_v, dep_v = rend[0, ..., :3], rend[0, ..., 3]
            fx, fy, cx, cy = Kv[0, 0, 0], Kv[0, 1, 1], Kv[0, 0, 2], Kv[0, 1, 2]
            vv, uu = torch.meshgrid(torch.arange(Hv, device=device, dtype=torch.float32),
                                    torch.arange(Wv, device=device, dtype=torch.float32), indexing="ij")
            Xc = torch.stack([(uu - cx) / fx * dep_v, (vv - cy) / fy * dep_v, dep_v], -1).reshape(-1, 3)
            Xw = Xc @ Rv + cv_          # c2w: R_v^T x + c
            warps, inbs = [], []
            for sidx in (i, j):
                Rs, ts = viewmats[sidx, :3, :3], viewmats[sidx, :3, 3]
                Xs = Xw @ Rs.T + ts
                z = Xs[:, 2].clamp_min(1e-6)
                gx = (Xs[:, 0] / z * fx + cx).reshape(Hv, Wv) * 2 / (Wv - 1) - 1
                gy = (Xs[:, 1] / z * fy + cy).reshape(Hv, Wv) * 2 / (Hv - 1) - 1
                grid = torch.stack([gx, gy], -1)[None]
                img = images_u8[sidx].to(device, non_blocking=True).permute(2, 0, 1)[None].float() / 255.0
                img = F.interpolate(img, size=(Hv, Wv), mode="area")
                warps.append(F.grid_sample(img, grid, mode="bilinear", align_corners=True, padding_mode="zeros")[0])
                inbs.append((Xs[:, 2].reshape(Hv, Wv) > 0.05) & (gx.abs() <= 1) & (gy.abs() <= 1))
            with torch.no_grad():
                agree = (warps[0] - warps[1]).abs().mean(0) < _vpc_tau
                m = (agree & inbs[0] & inbs[1]).float()
            if m.sum() < 1024:
                return None, 0.0
            tgt = 0.5 * (warps[0] + warps[1])
            l = ((col_v.permute(2, 0, 1) - tgt).abs().mean(0) * m).sum() / m.sum()
            return l, float(m.mean())

    # 03/09 (x58) — ba luật vật lý, xem help của --ray_var_weight / --manhattan_weight / --geo_consist_weight.
    # Mọi thứ ở đây chỉ chạy khi weight > 0; mặc định 0 giữ đường cũ bit-for-bit.
    _rv_w = float(getattr(cfg, "ray_var_weight", 0.0))
    _mh_w = float(getattr(cfg, "manhattan_weight", 0.0))
    _gc_w = float(getattr(cfg, "geo_consist_weight", 0.0))
    _phys_down = max(1, int(getattr(cfg, "ray_var_down", 4)))
    _g_vec = None
    if _mh_w > 0:
        # trọng lực = trục phương sai nhỏ nhất của đám mây SfM (cảnh bay = lát mỏng); dấu không quan trọng (|n·g|)
        _P = torch.from_numpy(np.asarray(scene.points.xyz, dtype=np.float32))
        _P = _P[torch.randperm(_P.shape[0])[: min(300000, _P.shape[0])]]
        _P = _P - _P.mean(0, keepdim=True)
        _evals, _evecs = torch.linalg.eigh(_P.T @ _P / _P.shape[0])          # (3,3) nhỏ → lớn
        _g_vec = _evecs[:, 0].to(device)
        # kiểm: góc với trục nhìn (z camera) trung bình — nadir nên nhỏ; > 45° = trục sai, dừng
        _cz = viewmats[:, 2, :3].mean(0); _cz = _cz / _cz.norm()
        _ang = float(torch.rad2deg(torch.acos((_cz @ _g_vec).abs().clamp(max=1.0))))
        print(f"[manhattan] g={_g_vec.tolist()} eig={_evals.tolist()} góc(g, trục nhìn TB)={_ang:.1f}°", flush=True)
        assert _ang < 45.0, "trục trọng lực từ PCA SfM lệch trục nhìn > 45° — kiểm lại"
    _nn_cam = None
    if _gc_w > 0:
        _C = torch.stack([-viewmats[i, :3, :3].T @ viewmats[i, :3, 3] for i in range(viewmats.shape[0])])
        _D = torch.cdist(_C, _C); _D.fill_diagonal_(float("inf"))
        _nn_cam = _D.argmin(1)                                               # camera train gần nhất theo tâm
        print(f"[geo_consist] weight={_gc_w} down={_phys_down} d_nn median={float(_D.min(1).values.median()):.3f}",
              flush=True)

    def _phys_K(down):
        Wd, Hd = W // down, H // down
        Kd = Ks[:1].clone(); Kd[:, 0, :] *= Wd / float(W); Kd[:, 1, :] *= Hd / float(H)
        return Kd, Wd, Hd

    def _ray_var_loss(vm, means_o=None, quats_o=None):
        """Var[z]/E[z]² dọc tia (alpha-weighted), render [z, z²] làm màu ở 1/down res."""
        from gsplat.rendering import rasterization
        Kd, Wd, Hd = _phys_K(_phys_down)
        _mu = splats["means"] if means_o is None else means_o
        _z = _mu @ vm[0, 2, :3] + vm[0, 2, 3]
        _col = torch.stack([_z, _z * _z], -1)
        _mip = getattr(splats, "_mip_filter", None)
        _sc = torch.exp(splats["scales"]); _op = torch.sigmoid(splats["opacities"])
        if _mip is not None:
            _s2 = _sc * _sc; _n2 = _s2 + (_mip * _mip).unsqueeze(-1)
            _op = _op * _mip_opacity_ratio(_s2, _n2); _sc = torch.sqrt(_n2)
        _r, _a, _ = rasterization(
            means=_mu, quats=splats["quats"] if quats_o is None else quats_o, scales=_sc, opacities=_op,
            colors=_col, viewmats=vm, Ks=Kd, width=Wd, height=Hd, sh_degree=None, render_mode="RGB",
            rasterize_mode="antialiased" if cfg.antialiased else "classic", near_plane=0.01, far_plane=1e10,
            packed=False)
        _a = _a[0, ..., 0].clamp_min(1e-6)
        _ez = _r[0, ..., 0] / _a; _ez2 = _r[0, ..., 1] / _a
        _rel = (_ez2 - _ez * _ez).clamp_min(0) / (_ez * _ez + 1e-6)
        _m = (_a > 0.5).detach()
        if int(_m.sum()) < 256:
            return None, 0.0
        return _rel[_m].mean(), float(_rel[_m].mean().sqrt())

    def _manhattan_loss():
        """op · min(|n·g|, 1−|n·g|) trên đĩa mỏng (smid/smin > 3), mẫu 1M Gaussian/bước."""
        N = splats["means"].shape[0]
        _ix = torch.randint(N, (min(N, 1_000_000),), device=device)
        _q = F.normalize(splats["quats"][_ix], dim=-1)
        _s = torch.exp(splats["scales"][_ix]).detach()
        _ss, _ = _s.sort(-1)                                                  # nhỏ → lớn
        _disk = (_ss[:, 1] / _ss[:, 0].clamp_min(1e-9) > 3.0)
        _w, _x, _y, _zq = _q.unbind(-1)
        # cột của R (trục Gaussian) — chọn trục ứng với scale NHỎ NHẤT = pháp tuyến
        _R = torch.stack([
            torch.stack([1 - 2 * (_y * _y + _zq * _zq), 2 * (_x * _y - _w * _zq), 2 * (_x * _zq + _w * _y)], -1),
            torch.stack([2 * (_x * _y + _w * _zq), 1 - 2 * (_x * _x + _zq * _zq), 2 * (_y * _zq - _w * _x)], -1),
            torch.stack([2 * (_x * _zq - _w * _y), 2 * (_y * _zq + _w * _x), 1 - 2 * (_x * _x + _y * _y)], -1),
        ], -2)                                                                # (M,3,3) hàng
        _ax = _s.argmin(-1)
        _n = _R[torch.arange(_R.shape[0], device=device), :, _ax]             # (M,3)
        _c = (_n @ _g_vec).abs()
        _pen = torch.minimum(_c, 1 - _c)
        _op = torch.sigmoid(splats["opacities"][_ix]).detach()
        _wm = _op * _disk.float()
        if float(_wm.sum()) < 1:
            return None, 0.0
        return (_pen * _wm).sum() / _wm.sum(), float((_pen * _wm).sum() / _wm.sum())

    def _geo_consist_loss(i, depth_i, means_o=None, quats_o=None):
        """depth render view i (full res, đã có) chiếu sang camera j gần nhất, so với depth render tại j (1/down)."""
        j = int(_nn_cam[i])
        Kd, Wd, Hd = _phys_K(_phys_down)
        _rj, _aj, _ = rasterize(splats, viewmats[j : j + 1], Kd, Wd, Hd, 0, cfg, render_depth=True,
                                radial_coeffs=radial, means_override=means_o, quats_override=quats_o)
        _dj = _rj[0, ..., 3]                                                  # (Hd,Wd) E[z] tại j
        # avg_pool2d chứ không interpolate(area): adaptive pool trên (1,1,3648,5472) nổ sharedMem (smoke 03/09)
        _di = F.avg_pool2d(depth_i.permute(0, 3, 1, 2), _phys_down)[0, 0, :Hd, :Wd]
        fx, fy, cx, cy = Kd[0, 0, 0], Kd[0, 1, 1], Kd[0, 0, 2], Kd[0, 1, 2]
        vv, uu = torch.meshgrid(torch.arange(Hd, device=device, dtype=torch.float32),
                                torch.arange(Wd, device=device, dtype=torch.float32), indexing="ij")
        Ri, ti = viewmats[i, :3, :3], viewmats[i, :3, 3]
        Xc = torch.stack([(uu - cx) / fx * _di, (vv - cy) / fy * _di, _di], -1).reshape(-1, 3)
        Xw = (Xc - ti) @ Ri                                                   # c2w: Rᵀ(x − t)
        Rj, tj = viewmats[j, :3, :3], viewmats[j, :3, 3]
        Xj = Xw @ Rj.T + tj
        zj = Xj[:, 2].clamp_min(1e-6)
        gx = (Xj[:, 0] / zj * fx + cx).reshape(Hd, Wd) * 2 / (Wd - 1) - 1
        gy = (Xj[:, 1] / zj * fy + cy).reshape(Hd, Wd) * 2 / (Hd - 1) - 1
        _samp = F.grid_sample(_dj[None, None], torch.stack([gx, gy], -1)[None], mode="bilinear",
                              align_corners=True, padding_mode="zeros")[0, 0]
        _res = (_samp - zj.reshape(Hd, Wd)).abs() / zj.reshape(Hd, Wd)
        with torch.no_grad():
            _m = (gx.abs() <= 1) & (gy.abs() <= 1) & (_di > 1e-3) & (_samp > 1e-3) & (_res < 0.1)
        if int(_m.sum()) < 256:
            return None, 0.0
        return _res[_m].mean(), float(_m.float().mean())

    # Dense mono-depth (DA-v2) regularization (§19.47 E3, sửa §19.50): images.bin
    # của GauU KHÔNG có track điểm 3D per-image (npts=0 cả 715 ảnh — vì thế
    # sparse-depth e2 là null-test), nên KHÔNG có neo SfM để fit scale/shift.
    # Thay bằng TỰ-CĂN-CHỈNH per-step: mỗi bước fit affine mono→disparity-render
    # trên mẫu pixel (2 ứng viên m và 1/m vì ngữ nghĩa DA-v2-hf không đảm bảo,
    # hệ số detach), rồi L1 kéo render về CẤU TRÚC của mono. Chỉ nạp map thô.
    mono_disp = None
    if getattr(cfg, "mono_depth_weight", 0.0) > 0 and getattr(cfg, "mono_depth_dir", ""):
        mono_disp = []
        n_load = 0
        for _m in scene.train_metas:
            _p = os.path.join(cfg.mono_depth_dir, os.path.splitext(_m.name)[0] + ".npy")
            if os.path.exists(_p):
                mono_disp.append(torch.from_numpy(np.load(_p).astype(np.float32)))
                n_load += 1
            else:
                mono_disp.append(None)
        print(f"[mono_depth] loaded {n_load}/{len(scene.train_metas)} map "
              f"(self-align per-step; dir={cfg.mono_depth_dir})")
        if n_load == 0:
            mono_disp = None

    # Handheld-video scenes (chair, bonsai) carry heavy per-frame motion blur:
    # sharpness varies up to 8x between adjacent frames of the same content.
    # Uniform loss weighting bakes that blur into the Gaussians, so discount
    # blurry frames by variance-of-Laplacian (the Instant-NGP culling metric,
    # used here as a soft weight instead of a hard drop to keep their coverage).
    sharp_w = None
    if getattr(cfg, "sharp_gamma", 0.0) > 0:
        lap_k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                             device=device).view(1, 1, 3, 3)
        lap = []
        for im in images_u8:
            g = im.to(device).float().mean(-1)[None, None] / 255.0
            lap.append(F.conv2d(F.avg_pool2d(g, 2), lap_k).var().item())
        lap = np.array(lap)
        rel = np.clip(lap / np.median(lap), 0.0, 1.0) ** cfg.sharp_gamma
        sharp_w = torch.from_numpy(
            np.maximum(rel, cfg.sharp_floor)).float().to(device)
        print(f"[sharp] mode={cfg.sharp_mode} gamma={cfg.sharp_gamma} "
              f"floor={cfg.sharp_floor} | w min={sharp_w.min():.3f} "
              f"med={sharp_w.median():.3f} | {(sharp_w < 0.9).sum().item()}/"
              f"{n_imgs} frames downweighted", flush=True)

    radial = None
    if scene.distorted:
        radial = make_radial(scene, cfg, device)
        if radial is not None and cfg.depth_weight > 0:
            print("[warn] depth rendering unsupported with UT/distorted mode "
                  "-> forcing depth_weight=0")
            cfg.depth_weight = 0.0

    ckpt0 = None
    if getattr(cfg, "init_ckpt", ""):
        ckpt0 = torch.load(cfg.init_ckpt, map_location="cpu")
        print(f"[finetune] loaded {cfg.init_ckpt} "
              f"({ckpt0['splats']['means'].shape[0]} gaussians, "
              f"{ckpt0.get('steps', '?')} steps)")
    splats, optimizers = create_splats_and_optimizers(
        scene, cfg, device, init_state=ckpt0["splats"] if ckpt0 else None)
    print(f"[init] {splats['means'].shape[0]} gaussians")

    # Mip-Splatting 3D filter buffer: recomputed from the (single) train
    # intrinsic as means move; stashed on `splats` so rasterize()/render/eval
    # all pick it up. Off => attribute absent => exact original behaviour.
    mip_on = float(getattr(cfg, "mip_filter", 0.0)) > 0

    def refresh_mip():
        splats._mip_filter = compute_mip_filter(
            splats["means"].detach(), viewmats, K, W, H, factor=cfg.mip_filter)

    if mip_on:
        refresh_mip()
        print(f"[mip] 3D filter ON factor={cfg.mip_filter} | sigma "
              f"med={splats._mip_filter.median():.4g} "
              f"max={splats._mip_filter.max():.4g}", flush=True)

    # fine-tune mode: topology frozen — no MCMC relocation/growth/noise
    strategy = None
    strategy_kind = "mcmc"
    if ckpt0 is None and str(getattr(cfg, "strategy", "mcmc")).lower() == "default":
        # ---- ADC (3DGS gốc / AbsGS / Revised-opacity) --------------------
        # Mặc định giữ ĐÚNG số của bài báo (dừng densify ở 50% số bước,
        # reset opacity mỗi 3000, ngưỡng 2e-4) chứ không mượn số của MCMC:
        # baseline phải là phương pháp NHƯ ĐÃ CÔNG BỐ, nếu không thì thắng/thua
        # lại lẫn với chuyện ta vặn núm hộ nó.
        strategy_kind = "default"
        _absgrad_on = bool(int(getattr(cfg, "adc_absgrad", 0)))
        if _absgrad_on and radial is not None:
            raise ValueError(
                "--adc_absgrad cần means2d.absgrad từ đường raster thường; "
                "đường UT/3DGUT (--force_ut / distorted) không sinh ra nó. "
                "Bỏ --force_ut hoặc bỏ --adc_absgrad.")
        strategy = _capped_default_class()(
            prune_opa=float(getattr(cfg, "adc_prune_opa", 0.005)),
            grow_grad2d=float(getattr(cfg, "adc_grow_grad2d", 0.0002)),
            grow_scale3d=float(getattr(cfg, "adc_grow_scale3d", 0.01)),
            prune_scale3d=float(getattr(cfg, "adc_prune_scale3d", 0.1)),
            refine_start_iter=500,
            refine_stop_iter=int(cfg.max_steps
                                 * float(getattr(cfg, "adc_stop_ratio", 0.5))),
            refine_every=int(getattr(cfg, "adc_refine_every", 100)),
            reset_every=int(getattr(cfg, "adc_reset_every", 3000)),
            absgrad=_absgrad_on,
            revised_opacity=bool(int(getattr(cfg, "adc_revised_opacity", 0))),
            verbose=True,
        )
        strategy._cap = int(getattr(cfg, "cap_max", 0) or 0)
        strategy._pixel_gs = bool(int(getattr(cfg, "adc_pixel_gs", 0)))
        strategy._budget = int(getattr(cfg, "adc_budget", 0) or 0)
        if strategy._pixel_gs:
            print("[densify] Pixel-GS BẬT: gradient lấy trung bình CÓ TRỌNG SỐ "
                  "theo diện tích chiếu (∝ bán kính²) thay vì mỗi view một phiếu",
                  flush=True)
        if strategy._budget > 0:
            print(f"[densify] Taming-3DGS BẬT: ngân sách cuối cố định "
                  f"{strategy._budget}, ngưỡng gradient được SUY RA mỗi lần refine",
                  flush=True)
        strategy.check_sanity(splats, optimizers)
        state = strategy.initialize_state(scene_scale=scene.scene_scale)
        print(f"[densify] ADC (DefaultStrategy) grow_grad2d="
              f"{strategy.grow_grad2d:g} absgrad={_absgrad_on} "
              f"revised_opacity={strategy.revised_opacity} "
              f"reset_every={strategy.reset_every} "
              f"stop={strategy.refine_stop_iter} trần={strategy._cap} "
              f"scene_scale={scene.scene_scale:.2f}", flush=True)
    elif ckpt0 is None:
        mcmc_kw = dict(
            cap_max=cfg.cap_max,
            noise_lr=getattr(cfg, "noise_lr", 5e5),
            refine_start_iter=500,
            refine_stop_iter=int(cfg.max_steps * getattr(cfg, "refine_stop_ratio", 25 / 30)),
            refine_every=getattr(cfg, "refine_every", 100),
            min_opacity=getattr(cfg, "min_opacity", 0.005),
        )
        _errlam = float(getattr(cfg, "err_guided", 0.0))
        _las_quota = float(getattr(cfg, "las_quota", 0.0))
        if _errlam > 0 and _las_quota > 0:
            raise ValueError("--err_guided and --las_quota are mutually exclusive")
        if _las_quota > 0:
            strategy = _absgrad_las_mcmc_class()(
                las_quota=_las_quota,
                las_distance=getattr(cfg, "las_distance", 0.45),
                las_opacity_reduction=getattr(
                    cfg, "las_opacity_reduction", 0.6),
                las_min_radius_px=getattr(cfg, "las_min_radius_px", 2.0),
                las_min_anisotropy=getattr(
                    cfg, "las_min_anisotropy", 1.0),
                las_preflight_calls=getattr(
                    cfg, "las_preflight_calls", 1),
                las_signal_every=getattr(cfg, "las_signal_every", 10),
                **mcmc_kw)
            print(
                "[densify] J122-D projected-UT AbsGrad + capped LAS/MCMC "
                f"quota={_las_quota:g} signal_every="
                f"{getattr(cfg, 'las_signal_every', 10)} "
                f"distance={getattr(cfg, 'las_distance', 0.45):g} "
                f"radius>={getattr(cfg, 'las_min_radius_px', 2.0):g}px; "
                "production loss remains 3DGUT Eval3D", flush=True)
        elif _errlam > 0:
            strategy = _errguided_mcmc_class()(err_lambda=_errlam, **mcmc_kw)
            print(f"[densify] error-guided MCMC (render-error-weighted) λ={_errlam}", flush=True)
        else:
            from gsplat.strategy import MCMCStrategy
            strategy = MCMCStrategy(**mcmc_kw)
        strategy.check_sanity(splats, optimizers)
        state = strategy.initialize_state()
        _invis_gsd = 0.0
        if int(getattr(cfg, "relocate_invisible", 0)) and cfg.antialiased:
            # p20 độ sâu (view gần nhất hay thấy) chứ không phải trung vị: chỉ đánh dấu
            # cái vô hình NGAY CẢ ở view nhìn gần, tránh giết Gaussian nadir còn dùng được.
            _zm, _zsrc = _scene_depth_median(scene, q=20.0)
            _invis_gsd = _zm / float(scene.K[0, 0]) if _zm > 0 else 0.0
            print(f"[opacity] relocate_invisible: GSD {_invis_gsd:.3e} đơn vị/px "
                  f"(depth p20 {_zm:.2f}, {_zsrc})", flush=True)
        if getattr(cfg, "opacity_probe_every", 5) >= 0:
            _install_mcmc_opacity_probe(
                getattr(cfg, "min_opacity", 0.005),
                int(getattr(cfg, "opacity_probe_every", 5)), invis_gsd=_invis_gsd)

    # Sàn log-scale, quy từ đơn vị TƯƠNG ĐỐI ra thế giới bằng scene_scale — một
    # hằng số tuyệt đối sẽ vô nghĩa khi scene_scale nhảy từ 10 (HCM0421) lên 662
    # (SMBU). Xem §19.24 và chỗ kẹp sau opt.step().
    _sf_rel = float(getattr(cfg, "scale_floor_rel", 0.0))
    _scale_floor_log = None
    if _sf_rel > 0:
        _scale_floor_log = float(np.log(_sf_rel * scene.scene_scale))
        print(f"[scale_floor] sàn tương đối {_sf_rel:g} × scene_scale "
              f"{scene.scene_scale:.2f} = {_sf_rel * scene.scene_scale:.3e} đơn vị "
              f"thế giới (log {_scale_floor_log:.3f})", flush=True)

    # PrismGS size floor: quy "size_floor_px pixel" ra đơn vị thế giới bằng GSD
    # THẬT của cảnh (độ sâu trung vị / tiêu cự), chứ không đặt một hằng số — cùng
    # một hằng số sẽ vô nghĩa khi scene_scale nhảy từ 10 (HCM0421) lên 662 (SMBU).
    size_floor = 0.0
    if float(getattr(cfg, "size_reg", 0.0)) > 0:
        zmed, src = _scene_depth_median(scene)
        if zmed > 0:
            gsd = zmed / float(scene.K[0, 0])
            size_floor = float(getattr(cfg, "size_floor_px", 1.0)) * gsd
            print(f"[prismgs] size floor = {getattr(cfg, 'size_floor_px', 1.0):g} px "
                  f"x GSD {gsd:.4f} (depth median {zmed:.1f} / f {scene.K[0, 0]:.1f}, "
                  f"nguồn: {src}) = {size_floor:.4f} đơn vị thế giới", flush=True)
        else:
            print("[prismgs] không suy được độ sâu -> size_reg bị tắt", flush=True)

    schedulers = [
        torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / max(cfg.max_steps, 1)))
    ]

    bil_grids, bil_opt = None, None
    if cfg.bilagrid and not getattr(cfg, "use_ppisp", 0):
        # grids are sized/indexed by the FULL view list (scene.train_orig_idx)
        # so they stay checkpoint-compatible regardless of holdout
        bil_grids = BilateralGrid(scene.n_total_views,
                                  grid_X=16, grid_Y=16, grid_W=8).to(device)
        if ckpt0 is not None and "bil_grids" in ckpt0:
            bil_grids.load_state_dict(ckpt0["bil_grids"])
        bil_opt = torch.optim.Adam(bil_grids.parameters(),
                                   lr=2e-3 * getattr(cfg, "lr_scale", 1.0), eps=1e-15)
    grid_ids = torch.from_numpy(scene.train_orig_idx).to(device)

    # PPISP: learned exposure/vignetting/color/CRF correction + test-time
    # controller (replaces bilagrid — mutually exclusive, see above).
    # Frames indexed by the FULL view list (like bil_grids) for
    # holdout-independent checkpoint compatibility.
    ppisp_module, ppisp_opts, ppisp_scheds = None, None, None
    if getattr(cfg, "use_ppisp", 0):
        from ppisp import PPISP, PPISPConfig

        # default controller_activation_ratio=0.8 assumes ~30-35k-iter runs
        # (paper's own Stage1+Stage2 budget); at our 100k-step base phase
        # that means a freshly-initialized, unwarmed controller suddenly
        # overrides converged per-frame color params at step 80k for every
        # image at once -> loss spike -> NaN splats -> MCMC relocate crash
        # (observed job41 v1). Fix: keep controller inactive through base
        # (ratio > 1), only let it activate+warm up during the short ft
        # phase (ratio matches ft's own much shorter schedule).
        ppisp_cfg = PPISPConfig(
            controller_activation_ratio=getattr(cfg, "ppisp_controller_ratio", 0.8))
        if ckpt0 is not None and "ppisp" in ckpt0:
            ppisp_module = PPISP.from_state_dict(ckpt0["ppisp"], config=ppisp_cfg).to(device)
        else:
            ppisp_module = PPISP(num_cameras=1, num_frames=scene.n_total_views,
                                 config=ppisp_cfg).to(device)
        ppisp_opts = ppisp_module.create_optimizers()
        ppisp_scheds = ppisp_module.create_schedulers(ppisp_opts, cfg.max_steps)

    pose_adj, pose_opt_optim = None, None
    if getattr(cfg, "pose_opt", 0):
        pose_adj = PoseOpt(n_imgs).to(device)
        pose_opt_optim = torch.optim.Adam(pose_adj.parameters(),
                                          lr=cfg.pose_opt_lr)
        # 03/09 (x74): --pose_opt_list <file tên ảnh> → CHỈ các camera đó được học residual (ảnh pose SfM yếu, <400 track);
        # camera mạnh giữ nguyên pose BTC = neo khung toạ độ cho test pose. Mask gradient theo hàng của res.
        _pl = getattr(cfg, "pose_opt_list", "") or ""
        if _pl:
            _names = set(ln.strip() for ln in open(_pl) if ln.strip())
            _mask = torch.tensor([1.0 if m.name in _names else 0.0 for m in scene.train_metas], device=device)
            pose_adj.res.register_hook(lambda g, _m=_mask: g * _m[:, None])
            pose_adj._mask = _mask
            print(f"[pose_opt] chỉ học pose cho {int(_mask.sum())}/{n_imgs} camera trong {os.path.basename(_pl)} (lr {cfg.pose_opt_lr}, reg {cfg.pose_opt_reg})", flush=True)

    lpips_fn = None
    if cfg.lpips_weight > 0:
        import lpips

        lpips_fn = lpips.LPIPS(net=cfg.lpips_net).to(device)
        for p in lpips_fn.parameters():
            p.requires_grad_(False)

    # WD-R (Apple 2603.23297, 26/08): Wasserstein Distortion trên thống kê cục bộ
    # (mean/std, pool Gaussian σ) của feature VGG16 — khớp *thống kê texture* thay
    # vì vị trí từng pixel, nên không kéo về mờ như L1/SSIM ở vùng foliage. Dùng
    # chung máy crop với LPIPS (crop trước khi vào VGG). wd_weight=0 = tắt hẳn.
    wd_fn = None
    if float(getattr(cfg, "wd_weight", 0.0)) > 0:
        import torchvision

        class _WD(torch.nn.Module):
            def __init__(self, sigma):
                super().__init__()
                vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features.eval()
                self.blocks = torch.nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16], vgg[16:23]])
                for p_ in self.parameters():
                    p_.requires_grad_(False)
                self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
                self.sigma = float(sigma)

            def _pool(self, x, s):
                s = max(s, 0.5)
                k = int(2 * round(3 * s) + 1)
                ax = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
                g = torch.exp(-0.5 * (ax / s) ** 2); g = g / g.sum()
                C = x.shape[1]
                x = F.conv2d(x, g.view(1, 1, 1, k).expand(C, 1, 1, k), padding=(0, k // 2), groups=C)
                x = F.conv2d(x, g.view(1, 1, k, 1).expand(C, 1, k, 1), padding=(k // 2, 0), groups=C)
                return x

            def forward(self, x, y):
                # x,y: (N,3,H,W) trong [0,1]. Tầng 0 = pixel; tầng 1..4 = VGG.
                x = (x - self.mean) / self.std
                y = (y - self.mean) / self.std
                tot, s = 0.0, self.sigma
                fx, fy = x, y
                for li in range(5):
                    if li > 0:
                        fx, fy = self.blocks[li - 1](fx), self.blocks[li - 1](fy)
                        if li > 1:
                            s = s / 2.0
                    mx, my = self._pool(fx, s), self._pool(fy, s)
                    vx = (self._pool(fx * fx, s) - mx * mx).clamp_min(1e-6).sqrt()
                    vy = (self._pool(fy * fy, s) - my * my).clamp_min(1e-6).sqrt()
                    tot = tot + ((mx - my) ** 2 + (vx - vy) ** 2).mean()
                return tot / 5.0

        wd_fn = _WD(getattr(cfg, "wd_sigma", 4.0)).to(device)
        print(f"[wd] WD-R bật: weight={cfg.wd_weight} sigma={cfg.wd_sigma} "
              f"start={cfg.lpips_start} crop={getattr(cfg, 'lpips_crop', 0)}", flush=True)

    perm = np.random.permutation(n_imgs)
    ptr = 0
    pbar_every = 500

    # --ss_train: render each training view at S x and box-average down to
    # native before any loss. Test-time supersampling LOST on ft models
    # (job46) precisely because ft had optimized LPIPS at native sampling;
    # this trains the ft at the supersampled operating point instead, so a
    # matching test-time --supersample no longer fights the optimum. Same
    # K-and-frame joint scaling as _render_ss => radial coeffs stay valid.
    ss_train = float(getattr(cfg, "ss_train", 1.0) or 1.0)
    if ss_train > 1.0:
        assert cfg.depth_weight == 0, \
            "ss_train does not rescale the depth-loss sample grid"
        W_ss, H_ss = int(round(W * ss_train)), int(round(H * ss_train))
        Ks_ss = Ks.clone()
        Ks_ss[:, 0, :] *= W_ss / float(W)
        Ks_ss[:, 1, :] *= H_ss / float(H)

    # Edge-aware L1: thin high-contrast structures (BTS lattice/cables/roof
    # edges) are a tiny fraction of pixels, so a uniform L1 lets the optimizer
    # under-invest there. Upweight the per-pixel L1 by the GT gradient
    # magnitude (renormalised to mean 1, so edge_weight=0 == plain F.l1_loss and
    # the overall loss scale is unchanged). Kernels built once.
    edge_w = getattr(cfg, "edge_weight", 0.0)
    if edge_w > 0:
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                               device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                               device=device).view(1, 1, 3, 3)
        print(f"[edge] edge_weight={edge_w} (per-pixel L1 upweight by GT grad)",
              flush=True)

    # --roi_ss: FOVEATED supersampling. The orbital rig keeps one subject (the
    # BTS tower + its thin cables) centred; global ss_train=2 already won, but
    # thin cables sit below the native sampling limit. Render a window around the
    # projected orbit-centre at a HIGHER factor and add it as an auxiliary
    # supervision term. This is a supervision-RESOLUTION lever (same family as
    # the winning ss_train), NOT a loss reweight (j61 edge-weight lost) nor a
    # Gaussian-count reallocation (j73 error-guided lost). Periphery/background is
    # left untouched (it may be genuine camera DoF).
    roi_ss = float(getattr(cfg, "roi_ss", 1.0) or 1.0)
    if roi_ss > 1.0:
        roi_frac = float(getattr(cfg, "roi_frac", 0.5))
        roi_w = float(getattr(cfg, "roi_w", 1.0))
        roi_lpips = int(getattr(cfg, "roi_lpips", 1))
        orbit_c = _orbit_center(viewmats).to(device)
        print(f"[roi_ss] foveated S={roi_ss} frac={roi_frac} w={roi_w} "
              f"lpips={roi_lpips} center={orbit_c.tolist()}", flush=True)

    # ---- distill-back (exp_diffix §3C): view giả ở pose nội suy đã qua fixer
    # làm giám sát phụ. Backward RIÊNG cho nhánh pseudo, không cộng vào loss
    # chính => đỉnh VRAM là MAX của hai lần render, không phải tổng.
    pseudo_imgs = None
    _pdir = str(getattr(cfg, "pseudo_dir", "") or "")
    if _pdir:
        import glob as _glob
        _pmeta = torch.load(os.path.join(_pdir, "meta.pt"), map_location="cpu")
        _pc2w = _pmeta["c2w"].numpy().astype(np.float64)  # (M,4,4)
        _pfiles = sorted(sum((_glob.glob(os.path.join(_pdir, "img_*" + e))
                              for e in (".png", ".jpg", ".JPG", ".jpeg")), []))
        assert len(_pfiles) == _pc2w.shape[0], \
            f"pseudo_dir: {len(_pfiles)} ảnh != {_pc2w.shape[0]} pose meta.pt"
        from PIL import Image as _PImage
        _PImage.MAX_IMAGE_PIXELS = None
        pseudo_imgs = [torch.from_numpy(
            np.asarray(_PImage.open(_f).convert("RGB")).copy())
            for _f in _pfiles]
        pseudo_vm = torch.from_numpy(np.stack(
            [np.linalg.inv(c) for c in _pc2w])).float().to(device)
        _pprob = float(getattr(cfg, "pseudo_prob", 0.3))
        _plam = float(getattr(cfg, "pseudo_lambda", 0.3))
        _pscale = float(getattr(cfg, "pseudo_scale", 1.0))
        _pW, _pH = int(round(W * _pscale)), int(round(H * _pscale))
        _pKs = Ks.clone()
        _pKs[:, 0, :] *= _pW / float(W)
        _pKs[:, 1, :] *= _pH / float(H)
        print(f"[pseudo] {len(_pfiles)} view giả từ {_pdir} prob={_pprob} "
              f"lambda={_plam} scale={_pscale} ({_pW}x{_pH})", flush=True)

    for step in range(cfg.max_steps):
        if ptr >= n_imgs:
            perm = np.random.permutation(n_imgs)
            ptr = 0
        idx = int(perm[ptr]); ptr += 1

        pixels = images_u8[idx].to(device, non_blocking=True).float() / 255.0
        pixels = pixels.unsqueeze(0)  # (1,H,W,3)

        sh_deg = (cfg.sh_degree if ckpt0 is not None
                  else min(step // cfg.sh_degree_interval, cfg.sh_degree))
        means_o = quats_o = None
        if pose_adj is not None:
            Mc = pose_adj.world_correction(viewmats[idx], idx)
            means_o = splats["means"] @ Mc[:3, :3].T + Mc[:3, 3]
            quats_o = _qmul(_rotmat_to_quat(Mc[:3, :3]), splats["quats"])
        # AbsGS cần means2d.absgrad từ chính lượt render sản xuất. Đường UT không
        # có means2d khả vi, và error-guided lấy tín hiệu từ sai số render.
        _absg = (strategy_kind == "default"
                 and bool(int(getattr(cfg, "adc_absgrad", 0))))
        if ss_train > 1.0:
            renders, alphas, info = rasterize(
                splats, viewmats[idx : idx + 1], Ks_ss, W_ss, H_ss, sh_deg, cfg,
                render_depth=False, radial_coeffs=radial,
                means_override=means_o, quats_override=quats_o, absgrad=_absg)
            colors = F.interpolate(renders[..., :3].permute(0, 3, 1, 2),
                                   size=(H, W), mode="area").permute(0, 2, 3, 1)
            depths = None
        else:
            renders, alphas, info = rasterize(
                splats, viewmats[idx : idx + 1], Ks, W, H, sh_deg, cfg,
                render_depth=(cfg.depth_weight > 0 or mono_disp is not None),
                radial_coeffs=radial,
                means_override=means_o, quats_override=quats_o, absgrad=_absg)
            if renders.shape[-1] == 4:
                colors, depths = renders[..., :3], renders[..., 3:]
            else:
                colors, depths = renders, None

        if ppisp_module is not None:
            gid = int(grid_ids[idx].item())
            colors = ppisp_module(colors[0], resolution=(W, H),
                                  camera_idx=0, frame_idx=gid).unsqueeze(0)
        elif bil_grids is not None:
            colors = apply_bilagrid(bil_grids, colors, grid_ids[idx : idx + 1])

        if strategy is not None:
            strategy.step_pre_backward(splats, optimizers, state, step, info)

        # transient masking: replace masked pixels of the render with GT so
        # they contribute zero gradient to every loss term
        colors_l = colors
        if masks_t[idx] is not None:
            m = masks_t[idx].to(device).unsqueeze(0).unsqueeze(-1)  # (1,H,W,1)
            colors_l = colors * (1 - m) + pixels * m

        if edge_w > 0:
            with torch.no_grad():
                lum = pixels.permute(0, 3, 1, 2).mean(1, keepdim=True)  # (1,1,H,W)
                gx = F.conv2d(lum, sobel_x, padding=1)
                gy = F.conv2d(lum, sobel_y, padding=1)
                gm = torch.sqrt(gx * gx + gy * gy)
                gm = gm / (gm.mean() + 1e-8)          # grad magnitude, mean 1
                ew = 1.0 + edge_w * gm                 # per-pixel weight
                ew = ew / ew.mean()                    # renormalise -> mean 1
            l1map = (colors_l - pixels).abs().mean(-1, keepdim=True) \
                .permute(0, 3, 1, 2)                   # (1,1,H,W)
            l1 = (l1map * ew).mean()
        else:
            l1 = F.l1_loss(colors_l, pixels)
        ssimval = ssim_torch(colors_l.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
        # "hf" discounts only the texture-driving terms so a blurry frame still
        # teaches colour/geometry; "all" discounts its whole photometric loss.
        w_all = w_hf = 1.0
        if sharp_w is not None:
            if cfg.sharp_mode == "all":
                w_all = float(sharp_w[idx])
            else:
                w_hf = float(sharp_w[idx])
        loss = w_all * ((1 - cfg.ssim_lambda) * l1
                        + cfg.ssim_lambda * w_hf * (1 - ssimval))

        # Perception/distortion Pareto controls.  Defaults are exactly zero,
        # preserving every historical recipe.  MSE directly matches the PSNR
        # component of the leaderboard, while the low-pass term anchors coarse
        # colour/structure and leaves LPIPS free to improve fine appearance.
        # Both operate after transient masking, so ignored pixels stay
        # gradient-free just like L1/SSIM/LPIPS.
        mse_weight = float(getattr(cfg, "mse_weight", 0.0))
        if mse_weight > 0:
            loss = loss + w_all * mse_weight * F.mse_loss(colors_l, pixels)
        lowpass_weight = float(getattr(cfg, "lowpass_weight", 0.0))
        if lowpass_weight > 0:
            kernel = int(getattr(cfg, "lowpass_kernel", 5))
            if kernel < 1 or kernel % 2 == 0:
                raise ValueError("--lowpass_kernel must be a positive odd integer")
            pred_lp = F.avg_pool2d(
                colors_l.permute(0, 3, 1, 2), kernel, stride=1,
                padding=kernel // 2)
            gt_lp = F.avg_pool2d(
                pixels.permute(0, 3, 1, 2), kernel, stride=1,
                padding=kernel // 2)
            loss = loss + w_all * lowpass_weight * F.l1_loss(pred_lp, gt_lp)

        # PrismGS (arXiv 2510.07830) thành phần (a): giám sát ĐA TỈ LỆ theo kim tự
        # tháp. `lowpass_weight` ở trên chỉ là MỘT mức và giữ nguyên độ phân giải;
        # ở đây thực sự hạ mẫu 2^l lần, nên mỗi mức là một bài toán khớp riêng và
        # ép Gaussian phải đúng cả ở tần số thấp — đúng cái ảnh drone cần khi một
        # Gaussian phủ nhiều pixel ở vùng xa. Mặc định 0 = tắt, mọi recipe cũ giữ nguyên.
        ms_weight = float(getattr(cfg, "ms_weight", 0.0))
        if ms_weight > 0:
            ms_levels = int(getattr(cfg, "ms_levels", 3))
            pc = colors_l.permute(0, 3, 1, 2)
            gc = pixels.permute(0, 3, 1, 2)
            for lv in range(1, ms_levels + 1):
                f = 2 ** lv
                if min(H, W) // f < 16:      # dưới 16 px thì SSIM window vô nghĩa
                    break
                p_lv = F.avg_pool2d(pc, f)   # avg_pool = tiền lọc, đúng tinh thần bài
                g_lv = F.avg_pool2d(gc, f)
                l1_lv = F.l1_loss(p_lv, g_lv)
                ssim_lv = ssim_torch(p_lv, g_lv)
                # chia 2^(l-1): mức thô đóng góp ít dần, tổng trọng số hội tụ về ~2
                loss = loss + w_all * ms_weight / (2 ** (lv - 1)) * (
                    (1 - cfg.ssim_lambda) * l1_lv
                    + cfg.ssim_lambda * w_hf * (1 - ssim_lv))

        # PrismGS thành phần (b): CHẶN DƯỚI kích thước Gaussian. `scale_reg` ở dòng
        # dưới chỉ phạt Gaussian QUÁ TO; cái gây răng cưa/kim châm ở cảnh lớn lại là
        # Gaussian quá NHỎ so với thứ một pixel phân giải nổi. `size_floor` tính từ
        # GSD thật (median depth / focal) nên là ràng buộc VẬT LÝ, không phải hằng số.
        if size_floor > 0:
            smin = torch.exp(splats["scales"]).min(dim=-1).values
            loss = loss + float(cfg.size_reg) * F.relu(size_floor - smin).mean()

        if (lpips_fn is not None or wd_fn is not None) and step >= cfg.lpips_start:
            lpips_scale = 1.0
            lpips_ramp = int(getattr(cfg, "lpips_ramp", 0))
            if lpips_ramp > 0:
                lpips_scale *= min(
                    1.0, (step - int(cfg.lpips_start) + 1) / lpips_ramp)
            polish = int(getattr(cfg, "distortion_polish_steps", 0))
            if polish > 0:
                remaining = max(0, int(cfg.max_steps) - 1 - step)
                lpips_scale *= min(1.0, remaining / polish)
            # crop-LPIPS (C2/j284-r2): ở 21MP, VGG full-image chiếm ~75-80GB
            # activation (tường §4b). Chuẩn văn liệu cho perceptual loss ở res
            # cao là k crop ngẫu nhiên 256-512² mỗi iter (receptive field VGG
            # ~192px nên crop >=256 xấp xỉ full-image về tín hiệu). Render vẫn
            # full-frame — chỉ cắt TRƯỚC khi vào VGG, nên rasterizer không đổi
            # một bit; k crop 512² = ~2MP vào VGG thay vì 21MP (~10x nhẹ hơn).
            # lpips_crop=0 giữ nguyên đường cũ bit-for-bit.
            _r_nchw = colors_l.permute(0, 3, 1, 2)
            _g_nchw = pixels.permute(0, 3, 1, 2)
            _crop = int(getattr(cfg, "lpips_crop", 0))
            if _crop > 0:
                _H, _W = _r_nchw.shape[-2], _r_nchw.shape[-1]
                _cs = min(_crop, _H, _W)
                _k = max(1, int(getattr(cfg, "lpips_crops", 8)))
                _rs, _gs = [], []
                for _ in range(_k):
                    _y0 = int(torch.randint(0, _H - _cs + 1, (1,)).item())
                    _x0 = int(torch.randint(0, _W - _cs + 1, (1,)).item())
                    _rs.append(_r_nchw[..., _y0:_y0 + _cs, _x0:_x0 + _cs])
                    _gs.append(_g_nchw[..., _y0:_y0 + _cs, _x0:_x0 + _cs])
                _r_nchw = torch.cat(_rs, 0)
                _g_nchw = torch.cat(_gs, 0)
            if lpips_fn is not None:
                loss = loss + w_all * w_hf * cfg.lpips_weight * lpips_scale * lpips_fn(
                    _r_nchw, _g_nchw,
                    normalize=bool(cfg.lpips_normalize)).mean()
            if wd_fn is not None:
                loss = loss + w_all * w_hf * float(cfg.wd_weight) * lpips_scale * wd_fn(
                    _r_nchw, _g_nchw)

        if roi_ss > 1.0:
            # project the orbit-centre into this view -> window placement
            ki = 0 if Ks.shape[0] == 1 else idx   # single shared camera -> Ks is (1,3,3)
            with torch.no_grad():
                pc = viewmats[idx, :3, :3] @ orbit_c + viewmats[idx, :3, 3]
                if pc[2] > 1e-3:
                    u = (Ks[ki, 0, 0] * pc[0] / pc[2] + Ks[ki, 0, 2]).item()
                    v = (Ks[ki, 1, 1] * pc[1] / pc[2] + Ks[ki, 1, 2]).item()
                else:
                    u, v = W / 2.0, H / 2.0
            wc = max(16, int(round(W * roi_frac)))
            hc = max(16, int(round(H * roi_frac)))
            x0 = int(min(max(0, round(u - wc / 2)), max(0, W - wc)))
            y0 = int(min(max(0, round(v - hc / 2)), max(0, H - hc)))
            Wc, Hc = int(round(wc * roi_ss)), int(round(hc * roi_ss))
            Kc = Ks[ki : ki + 1].clone()
            Kc[:, 0, 0] *= roi_ss
            Kc[:, 1, 1] *= roi_ss
            Kc[:, 0, 2] = (Ks[ki, 0, 2] - x0) * roi_ss
            Kc[:, 1, 2] = (Ks[ki, 1, 2] - y0) * roi_ss
            r_c, _, _ = rasterize(
                splats, viewmats[idx : idx + 1], Kc, Wc, Hc, sh_deg, cfg,
                render_depth=False, radial_coeffs=radial,
                means_override=means_o, quats_override=quats_o)
            crop = F.interpolate(r_c[..., :3].permute(0, 3, 1, 2),
                                 size=(hc, wc), mode="area").permute(0, 2, 3, 1)
            if bil_grids is not None:
                gy, gx = torch.meshgrid(
                    (torch.arange(hc, device=device) + 0.5) / hc,
                    (torch.arange(wc, device=device) + 0.5) / wc, indexing="ij")
                fx = (x0 + gx * wc) / W          # crop-local -> full-frame norm
                fy = (y0 + gy * hc) / H
                xy = torch.stack([fx, fy], -1).unsqueeze(0)
                crop = bg_slice(bil_grids, xy, crop, grid_ids[idx : idx + 1])["rgb"]
            gt_crop = pixels[:, y0 : y0 + hc, x0 : x0 + wc, :]
            if masks_t[idx] is not None:
                mc = masks_t[idx].to(device).unsqueeze(0).unsqueeze(-1)
                mc = mc[:, y0 : y0 + hc, x0 : x0 + wc, :]
                crop = crop * (1 - mc) + gt_crop * mc
            roi_l1 = F.l1_loss(crop, gt_crop)
            roi_ssim = ssim_torch(crop.permute(0, 3, 1, 2),
                                  gt_crop.permute(0, 3, 1, 2))
            roi_loss = ((1 - cfg.ssim_lambda) * roi_l1
                        + cfg.ssim_lambda * (1 - roi_ssim))
            if roi_lpips and lpips_fn is not None and step >= cfg.lpips_start:
                roi_loss = roi_loss + cfg.lpips_weight * lpips_fn(
                    crop.permute(0, 3, 1, 2), gt_crop.permute(0, 3, 1, 2),
                    normalize=bool(cfg.lpips_normalize)).mean()
            loss = loss + roi_w * roi_loss

        if cfg.depth_weight > 0 and uvds[idx].shape[0] > 0:
            uvd = uvds[idx]
            gx = uvd[:, 0] / (W - 1) * 2 - 1
            gy = uvd[:, 1] / (H - 1) * 2 - 1
            grid = torch.stack([gx, gy], -1)[None, :, None, :]  # (1,M,1,2)
            d = F.grid_sample(depths.permute(0, 3, 1, 2), grid, align_corners=True)
            d = d.squeeze()
            disp = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
            disp_gt = 1.0 / uvd[:, 2]
            loss = loss + F.l1_loss(disp, disp_gt) * scene.scene_scale * cfg.depth_weight

        if mono_disp is not None and mono_disp[idx] is not None and depths is not None:
            _dr = depths[0, ..., 0]
            _disp_r = torch.where(_dr > 1e-6, 1.0 / _dr.clamp_min(1e-6),
                                  torch.zeros_like(_dr))
            _mu = F.interpolate(mono_disp[idx].to(device)[None, None],
                                size=_disp_r.shape, mode="bilinear",
                                align_corners=False)[0, 0]
            _end = int(getattr(cfg, "mono_depth_end", 0)) or cfg.max_steps
            _w = cfg.mono_depth_weight * max(0.0, 1.0 - step / max(_end, 1))
            if step < int(getattr(cfg, "mono_depth_start", 0)):
                _w = 0.0
            _valid = _dr > 1e-6
            if _w > 0 and int(_valid.sum()) > 1024:
                with torch.no_grad():
                    _ix = _valid.nonzero(as_tuple=False)
                    if _ix.shape[0] > 8192:
                        _ix = _ix[torch.randint(_ix.shape[0], (8192,), device=device)]
                    _y = _disp_r[_ix[:, 0], _ix[:, 1]]
                    _best = None
                    for _inv in (False, True):
                        _x = _mu[_ix[:, 0], _ix[:, 1]]
                        if _inv:
                            _x = 1.0 / _x.clamp_min(1e-6)
                        _mx, _my = _x.mean(), _y.mean()
                        _vx = ((_x - _mx) ** 2).mean()
                        if _vx < 1e-12:
                            continue
                        _a = (((_x - _mx) * (_y - _my)).mean() / _vx)
                        if _a <= 0:
                            continue
                        _b = _my - _a * _mx
                        _sse = ((_a * _x + _b - _y) ** 2).mean()
                        if _best is None or _sse < _best[0]:
                            _best = (_sse, _a, _b, _inv)
                if _best is not None:
                    _, _a, _b, _inv = _best
                    _mm = 1.0 / _mu.clamp_min(1e-6) if _inv else _mu
                    _tgt = (_a * _mm + _b).detach()
                    loss = loss + _w * (torch.abs(_disp_r - _tgt)[_valid]).mean() \
                        * scene.scene_scale

        if bil_grids is not None:
            loss = loss + 10.0 * total_variation_loss(bil_grids.grids)
        if ppisp_module is not None:
            loss = loss + ppisp_module.get_regularization_loss()

        # --sh_reg (26/08, Mind-the-Gap 2607.01556 "view-dependent regularizer"): L2 lên SH bậc ≥1.
        # SH0 lúc render: train −3,4 / test −1,3 ⇒ ~40 % phần view-dependent là memorization.
        if float(getattr(cfg, "sh_reg", 0.0)) > 0:
            loss = loss + float(cfg.sh_reg) * (splats["shN"] ** 2).mean()
        if cfg.opacity_reg > 0:
            loss = loss + cfg.opacity_reg * torch.sigmoid(splats["opacities"]).abs().mean()
        if cfg.scale_reg > 0:
            loss = loss + cfg.scale_reg * torch.exp(splats["scales"]).abs().mean()
        if getattr(cfg, "aniso_reg", 0.0) > 0:
            # phạt DỊ HƯỚNG (kim): scale_reg ở trên chỉ phạt kích thước tổng, không
            # chạm tỉ lệ trục — chữ ký "nổ kim nadir trên tháp" smbu §19.47 C3.
            _s = torch.exp(splats["scales"])
            _ratio = _s.max(dim=-1).values / _s.min(dim=-1).values.clamp_min(1e-8)
            loss = loss + cfg.aniso_reg * F.relu(_ratio - cfg.aniso_max).mean()
        # 03/09 (x58) — ba luật vật lý (mặt đục / trọng lực / nhất quán đa view); log mỗi pbar_every
        if _rv_w > 0 and step >= int(getattr(cfg, "ray_var_start", 0)):
            _lrv, _sprd = _ray_var_loss(viewmats[idx : idx + 1], means_o, quats_o)
            if _lrv is not None:
                loss = loss + _rv_w * _lrv
                if step % pbar_every == 0:
                    print(f"[ray_var] step {step} spread={_sprd:.4f}", flush=True)
        if _mh_w > 0:
            _lmh, _pen_m = _manhattan_loss()
            if _lmh is not None:
                loss = loss + _mh_w * _lmh
                if step % pbar_every == 0:
                    print(f"[manhattan] step {step} pen={_pen_m:.4f}", flush=True)
        if _gc_w > 0 and depths is not None:
            _lgc, _cov_g = _geo_consist_loss(idx, depths, means_o, quats_o)
            if _lgc is not None:
                loss = loss + _gc_w * _lgc
                if step % pbar_every == 0:
                    print(f"[geo_consist] step {step} loss={_lgc.item():.4f} cov={_cov_g:.3f}", flush=True)
        if pose_adj is not None and cfg.pose_opt_reg > 0:
            loss = loss + cfg.pose_opt_reg * (pose_adj.res ** 2).sum()

        if _vpc_w > 0 and np.random.rand() < float(getattr(cfg, "vpc_prob", 1.0)):
            _lv, _cov = _vpc_loss(idx, sh_deg)
            if _lv is not None:
                loss = loss + _vpc_w * _lv
                if step % pbar_every == 0:
                    print(f"[vpc] step {step} loss={_lv.item():.4f} cov={_cov:.3f}", flush=True)

        loss.backward()

        # nhánh pseudo: render + backward riêng SAU khi graph chính đã giải
        # phóng; grad cộng dồn vào cùng optimizer, step một lần bên dưới.
        # Không bilagrid/pose_adj/LPIPS/strategy cho view giả (pose chính xác
        # theo thiết kế, target sinh từ render thô + fixer).
        if pseudo_imgs is not None and np.random.rand() < _pprob:
            _pi = int(np.random.randint(len(pseudo_imgs)))
            _pt = pseudo_imgs[_pi].to(device).float().div_(255.0)
            if _pscale != 1.0:
                _pt = F.interpolate(_pt.permute(2, 0, 1)[None],
                                    size=(_pH, _pW), mode="area")[0] \
                    .permute(1, 2, 0)
            _pt = _pt.unsqueeze(0)
            _prend, _, _ = rasterize(
                splats, pseudo_vm[_pi : _pi + 1], _pKs, _pW, _pH, sh_deg, cfg,
                render_depth=False, radial_coeffs=radial)
            _pc = _prend[..., :3]
            _pl1 = F.l1_loss(_pc, _pt)
            _pssim = ssim_torch(_pc.permute(0, 3, 1, 2),
                                _pt.permute(0, 3, 1, 2))
            _ploss = _plam * ((1 - cfg.ssim_lambda) * _pl1
                              + cfg.ssim_lambda * (1 - _pssim))
            _ploss.backward()

        if pose_adj is not None and step in (200, 2000, 5000, 10000) and getattr(pose_adj, "_mask", None) is not None:
            with torch.no_grad():
                _rm = pose_adj.res[pose_adj._mask > 0]; _fpx = float(scene.K[0, 0])
                print(f"[pose_opt] step {step}: residual masked cams — rot px max {_fpx * _rm[:, :3].norm(dim=1).max():.1f} med {_fpx * _rm[:, :3].norm(dim=1).median():.1f} | "
                      f"trans px max {_fpx * _rm[:, 3:].norm(dim=1).max() / 2.5:.1f} med {_fpx * _rm[:, 3:].norm(dim=1).median() / 2.5:.1f}", flush=True)
        if pose_adj is not None and step == 20:
            g = pose_adj.res.grad
            if getattr(pose_adj, "_mask", None) is not None and float(pose_adj._mask[idx]) == 0:
                g = None; print("[pose_opt] grad check @20: view ngoài mask, bỏ qua kiểm", flush=True)
            gn = 0.0 if g is None else g.abs().max().item()
            print(f"[pose_opt] grad check @20: max|grad|={gn:.3e} "
                  f"{'OK' if gn > 0 else 'NO GRADIENT — UT path may not support it'}")

        # --batch_views B (26/08, H1): gộp gradient của B view rồi mới bước Adam
        # (Grendel-GS / 2506.12727 — giảm phương sai gradient 1-view/step, đỉnh
        # capacity dịch phải). B=1 giữ nguyên bit-for-bit đường cũ. Strategy
        # (ADC/MCMC) vẫn gọi theo từng view-step; noise MCMC đã chia B ở parse.
        _bv = int(getattr(cfg, "batch_views", 1))
        _do_step = (_bv <= 1) or ((step + 1) % _bv == 0) or (step == cfg.max_steps - 1)
        if _do_step:
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

        # SÀN LOG-SCALE (§19.24). Chẩn đoán NaN đo được log-scale nhỏ nhất còn
        # hữu hạn là −20.28 ⇒ scale ≈ 1,6e−9 trên cảnh có scene_scale 576, tức
        # ~3e−12 tương đối: Gaussian đã sụp thành điểm, hiệp phương sai chiếu
        # xuống 2D suy biến, và nghịch đảo nó là đường sinh inf gần nhất.
        # Kẹp SAU opt.step() nên đây là phép chiếu về miền hợp lệ, không đụng
        # vào gradient. Mặc định 0 = tắt hẳn ⇒ mọi mốc cũ giữ nguyên bit-for-bit.
        if _scale_floor_log is not None:
            splats["scales"].data.clamp_(min=_scale_floor_log)

        if bil_opt is not None and _do_step:
            bil_opt.step()
            bil_opt.zero_grad(set_to_none=True)
        if pose_opt_optim is not None and _do_step:
            if step >= int(getattr(cfg, "pose_opt_start", 0)):
                pose_opt_optim.step()
            pose_opt_optim.zero_grad(set_to_none=True)
        if ppisp_opts is not None and _do_step:
            for opt in ppisp_opts:
                opt.step()
                opt.zero_grad(set_to_none=True)
        # scheduler bước theo view-step để lịch decay 0.01^(1/max_steps) không đổi
        for sch in schedulers:
            sch.step()
        if ppisp_scheds is not None:
            for sch in ppisp_scheds:
                sch.step()

        # J122-D topology-only signal.  This runs after every production
        # optimizer has stepped and zeroed its gradients.  autograd.grad() is
        # restricted to a disposable detached SH leaf, so the auxiliary 2D-UT
        # approximation cannot alter the production Eval3D update.
        if (strategy is not None
                and float(getattr(cfg, "las_quota", 0.0)) > 0
                and step % int(getattr(cfg, "las_signal_every", 10)) == 0):
            if ppisp_module is not None:
                raise RuntimeError(
                    "J122-D auxiliary AbsGrad is not validated with PPISP")
            aux_W, aux_H, aux_Ks = W, H, Ks
            if ss_train > 1.0:
                aux_W, aux_H, aux_Ks = W_ss, H_ss, Ks_ss
            aux_renders, _, aux_info, aux_leaf = rasterize_projected_ut_absgrad(
                splats, viewmats[idx : idx + 1], aux_Ks, aux_W, aux_H,
                sh_deg, radial_coeffs=radial,
                means_override=means_o, quats_override=quats_o)
            aux_colors = aux_renders[..., :3]
            if ss_train > 1.0:
                aux_colors = F.interpolate(
                    aux_colors.permute(0, 3, 1, 2), size=(H, W), mode="area"
                ).permute(0, 2, 3, 1)
            if bil_grids is not None:
                aux_colors = apply_bilagrid(
                    bil_grids, aux_colors, grid_ids[idx : idx + 1])
            aux_colors_l = aux_colors
            if masks_t[idx] is not None:
                aux_mask = masks_t[idx].to(device).unsqueeze(0).unsqueeze(-1)
                aux_colors_l = aux_colors * (1 - aux_mask) + pixels * aux_mask
            aux_l1 = F.l1_loss(aux_colors_l, pixels)
            aux_ssim = ssim_torch(
                aux_colors_l.permute(0, 3, 1, 2),
                pixels.permute(0, 3, 1, 2))
            aux_loss = ((1 - cfg.ssim_lambda) * aux_l1
                        + cfg.ssim_lambda * (1 - aux_ssim))
            if not torch.isfinite(aux_loss):
                raise RuntimeError(
                    f"J122D_ABSGRAD_PREFLIGHT_FAIL step={step} "
                    f"aux_loss={float(aux_loss.detach())}")
            # Requesting only this leaf prevents .grad accumulation on splats,
            # bilateral grids, and the production loss graph.
            torch.autograd.grad(aux_loss, aux_leaf, retain_graph=False,
                                create_graph=False)
            dirty = [name for name, value in splats.items()
                     if value.grad is not None]
            if dirty:
                raise RuntimeError(
                    "J122D_AUX_GRAD_ISOLATION_FAIL "
                    f"step={step} dirty_splats={dirty}")
            strategy.record_absgrad(splats, state, aux_info, step)
            del aux_renders, aux_colors, aux_colors_l, aux_loss, aux_leaf

        if strategy is not None:
            # Chốt chặn NaN ngay TRƯỚC khi MCMC lấy mẫu. Bài học 15/08: hai
            # nhánh (c2_err, c3_mip) chết vì torch.multinomial / np.random.choice
            # nhận xác suất NaN — mà assert CUDA thì không cứu được, mất trắng
            # cả giờ huấn luyện. Chỉ kiểm ở bước refine (isfinite ép đồng bộ
            # GPU), nên chi phí ~0 so với 30k bước.
            if step % int(getattr(cfg, "refine_every", 100)) == 0:
                # Lần NaN ĐẦU TIÊN in kèm bối cảnh, vì đó là lần duy nhất còn
                # đọc được nguyên nhân: các lần sau tập NaN đã bão hoà và mọi
                # thống kê đều bị chính NaN làm bẩn. j215 (15/08) mất cơ hội này.
                if not globals().get("_NANGUARD_SEEN", False) and any(
                        not torch.isfinite(v.data).all() for v in splats.values()):
                    globals()["_NANGUARD_SEEN"] = True
                    mipbuf = getattr(splats, "_mip_filter", None)
                    sc = splats["scales"].data
                    op = splats["opacities"].data
                    fs, fo = torch.isfinite(sc), torch.isfinite(op)
                    print(f"[nanguard] LẦN ĐẦU ở bước {step}: n={sc.shape[0]} "
                          f"| scales hữu hạn: min={sc[fs].min():.3e} "
                          f"max={sc[fs].max():.3e} "
                          f"| opacities hữu hạn: min={op[fo].min():.3e} "
                          f"max={op[fo].max():.3e} "
                          f"| bộ lọc mip: "
                          + ("KHÔNG BẬT" if mipbuf is None else
                             (f"hữu hạn, min={mipbuf.min():.3e} max={mipbuf.max():.3e}"
                              if bool(torch.isfinite(mipbuf).all())
                              else f"CÓ {int((~torch.isfinite(mipbuf)).sum())} PHẦN TỬ NaN/inf")),
                          flush=True)
                for k, v in splats.items():
                    bad = ~torch.isfinite(v.data)
                    n_bad = int(bad.sum())
                    # Gradient cũng phải dọn: nó là thứ Adam sắp nuốt vào moment.
                    g_bad = (None if v.grad is None
                             else ~torch.isfinite(v.grad))
                    n_gbad = 0 if g_bad is None else int(g_bad.sum())
                    if not n_bad and not n_gbad:
                        continue
                    if n_bad:
                        if k == "opacities":
                            v.data[bad] = -20.0   # sigmoid ~ 2e-9 => coi như chết
                        else:
                            v.data[bad] = 0.0
                    if n_gbad:
                        v.grad[g_bad] = 0.0
                    # ⚠ CHỖ NÀY TỪNG SAI VÀ ĐÃ LÀM HỎNG j215 (15/08): chỉ dọn
                    # v.data là KHÔNG ĐỦ. Moment của Adam (exp_avg, exp_avg_sq)
                    # vẫn giữ NaN, nên ngay bước sau Adam lại bơm NaN ngược vào
                    # đúng những phần tử vừa dọn — log j215 cho thấy số phần tử
                    # hỏng còn NHÍCH LÊN sau mỗi lần dọn (193345 -> 193840 ->
                    # 193967) thay vì về 0. Phải dọn cả trạng thái optimizer.
                    st = optimizers[k].state.get(v, None) if k in optimizers else None
                    n_sbad = 0
                    if st:
                        for sk in ("exp_avg", "exp_avg_sq"):
                            m = st.get(sk, None)
                            if m is None:
                                continue
                            sbad = ~torch.isfinite(m)
                            n_sbad += int(sbad.sum())
                            m[sbad] = 0.0
                    print(f"[nanguard] bước {step}: {k} — tham số {n_bad}, "
                          f"gradient {n_gbad}, moment Adam {n_sbad} phần tử không "
                          f"hữu hạn, đã dọn (opacity -> tắt)", flush=True)
            if float(getattr(cfg, "err_guided", 0.0)) > 0:
                # stash per-pixel render-vs-GT L1 for error-guided sampling
                info["_pix_err"] = (colors_l - pixels).detach().abs().mean(-1)
            strategy.step_post_backward(splats, optimizers, state, step, info,
                                        lr=schedulers[0].get_last_lr()[0])

        # refresh the 3D filter as means move / MCMC relocates (after any
        # post-backward topology change so N/means match). Cheap: single
        # intrinsic, one N-vector min over views.
        if mip_on and step % 100 == 0:
            refresh_mip()

        if step % pbar_every == 0 or step == cfg.max_steps - 1:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[{step:6d}/{cfg.max_steps}] loss={loss.item():.4f} l1={l1.item():.4f} "
                  f"ssim={ssimval.item():.4f} n={splats['means'].shape[0]} "
                  f"mem={mem:.1f}G t={time.time()-t0:.0f}s", flush=True)

        # bảo hiểm OOM cho run dài (tường LPIPS @21MP: mem nhảy ~+75GB đúng tại
        # lpips_start — forward của bước đó nổ TRƯỚC khi kịp save cuối vòng, nên
        # chu kỳ save phải đủ dày để ckpt sống gần nhất nằm sát lpips_start).
        # os.replace: không bao giờ để lại ckpt_mid.pt viết dở nếu chết giữa save.
        ce = int(getattr(cfg, "ckpt_every", 0))
        if ce > 0 and step > 0 and step % ce == 0 and int(getattr(cfg, "save_ckpt", 1)):
            ckpt = {"splats": splats.state_dict(), "cfg": vars(cfg), "steps": step}
            if mip_on:
                ckpt["mip_filter"] = splats._mip_filter.detach().cpu()
            if bil_grids is not None:
                ckpt["bil_grids"] = bil_grids.state_dict()
            if ppisp_module is not None:
                ckpt["ppisp"] = ppisp_module.state_dict()
            if pose_adj is not None:
                ckpt["pose_adj_res"] = pose_adj.res.detach().cpu()
            _tmp = os.path.join(cfg.result_dir, "ckpt_mid.pt.tmp")
            torch.save(ckpt, _tmp)
            os.replace(_tmp, os.path.join(cfg.result_dir, "ckpt_mid.pt"))
            print(f"[ckpt] lưu giữa chừng @bước {step} -> ckpt_mid.pt "
                  f"(n={splats['means'].shape[0]})", flush=True)

    if pose_adj is not None:
        r = pose_adj.res.detach()
        try:  # 03/09: biên độ residual quy ra px (f·|rot| ; f·|t|/z≈2.5) cho từng camera được học
            _f = float(scene.K[0, 0]); _rows = [(scene.train_metas[i].name[-10:-4], float(_f * r[i, :3].norm()), float(_f * r[i, 3:].norm() / 2.5))
                                                 for i in range(r.shape[0]) if (getattr(pose_adj, "_mask", None) is None or float(pose_adj._mask[i]) > 0)]
            print("[pose_opt] px-equiv (rot,trans) per cam: " + " ".join(f"{n}:{a:.0f}/{b:.0f}" for n, a, b in _rows), flush=True)
        except Exception as _e:
            print(f"[pose_opt] px summary lỗi: {_e}", flush=True)
        print(f"[pose_opt] residuals: rot max={r[:, :3].abs().max():.2e} "
              f"mean={r[:, :3].abs().mean():.2e} | "
              f"trans max={r[:, 3:].abs().max():.2e} mean={r[:, 3:].abs().mean():.2e}")

    # final filter refresh so the saved buffer + the render/eval below match
    # the final means exactly (last MCMC relocation may fall between %100 ticks)
    if mip_on:
        refresh_mip()

    # save checkpoint (skip for 0-step runs: pure eval of an init_ckpt).
    # --save_ckpt 0 also skips it: at cap 96M a ckpt is ~60 GB and the node's
    # workdir shares the root disk, so writing one risks DiskPressure evictions
    # for the whole cluster (../docs/h200_use.md §1).
    if cfg.max_steps > 0 and int(getattr(cfg, "save_ckpt", 1)):
        ckpt = {"splats": splats.state_dict(), "cfg": vars(cfg),
                "steps": cfg.max_steps}
        if mip_on:
            ckpt["mip_filter"] = splats._mip_filter.detach().cpu()
        if bil_grids is not None:
            ckpt["bil_grids"] = bil_grids.state_dict()
        if ppisp_module is not None:
            ckpt["ppisp"] = ppisp_module.state_dict()
        if pose_adj is not None:
            # res (N,6) se3 per-train-view — cần cho phân tích pose-drift:
            # nội suy hiệu chỉnh của train view sang test pose (§19.51, hợp lệ
            # không cần GT test). Trước 17/08 bị vứt — mọi ckpt ft cũ mù drift.
            ckpt["pose_adj_res"] = pose_adj.res.detach().cpu()
        torch.save(ckpt, os.path.join(cfg.result_dir, "ckpt.pt"))
    print(f"[done] training in {(time.time()-t0)/60:.1f} min")

    # pose corrections for --test_pose_interp: live module first, else the
    # ckpt we started from (pure-eval on an ft ckpt)
    _pres = (pose_adj.res.detach() if pose_adj is not None else
             (ckpt0.get("pose_adj_res") if ckpt0 is not None else None))
    if not getattr(cfg, "skip_test_render", 0):
        render_test(cfg, scene, splats, bil_grids, device, ppisp=ppisp_module,
                    pose_res=_pres)
    if getattr(cfg, "holdout_every", 0) > 0:
        eval_holdout(cfg, scene, splats, bil_grids, device, ppisp=ppisp_module,
                     pose_res=_pres)


# --------------------------------------------------------------------------
def _test_K(cfg, scene, tp):
    """Intrinsics dùng để render một pose test.

    `test_poses.csv` (quy ước BTC) ÉP `cx,cy = W/2,H/2`, vứt tâm quang thật. Ở data
    BTC vòng 1 chuyện đó VÔ HẠI vì COLMAP đã ghim tâm đúng giữa ảnh — đo được lệch
    0.00 px trên cả HCM0421 lẫn HCM0674. Ở data ContextCapture (GauU-Scene) tâm thật
    lệch **8,4 px**, mà ta lại TRAIN bằng tâm thật ⇒ mọi ảnh test lệch 8,4 px so GT.
    Đo trần điểm bằng cách dịch chính ảnh GT đúng lượng đó: PSNR 16.30 / SSIM 0.2026
    — tức tái dựng HOÀN HẢO cũng không vượt được. j202/a1_fix ra 16.74/0.259 nghĩa là
    nó đã chạm trần, và khoảng cách tới paper (25.49) KHÔNG phải chất lượng mô hình.

    `cameras.bin` mà BTC phát CÓ tâm quang thật — chỉ mỗi CSV ném nó đi. Bật cờ này
    để lấy tâm từ `cameras.bin` thay vì từ CSV. Mặc định 0 = giữ nguyên quy ước BTC.
    """
    K = tp.K.copy()
    if int(getattr(cfg, "test_use_train_K", 0)):
        K[0, 2] = float(scene.K[0, 2]) * (tp.width / float(scene.width))
        K[1, 2] = float(scene.K[1, 2]) * (tp.height / float(scene.height))
    return K


@torch.no_grad()
def render_test(cfg, scene: SceneData, splats, bil_grids, device, suffix="", ppisp=None, pose_res=None):
    """Render all test poses; save raw pinhole and (optionally) redistorted.

    In distorted (3DGUT) mode, renders the distorted frame directly into
    test_renders_direct{suffix}."""
    if scene.distorted:
        render_test_direct(cfg, scene, splats, bil_grids, device, suffix,
                           ppisp=ppisp, pose_res=pose_res)
        return
    # j208: the pinhole path used to call rasterize() directly, so --supersample
    # (the round-1 test-time win, §"ss2") was silently a NO-OP on undistorted
    # data — exactly the round-2 GauU-Scene case.  Route it through _render_ss
    # like the 3DGUT path already did, and honour --ss_sweep so several factors
    # are scored off ONE trained model (no ckpt to keep: a 96M-Gaussian ckpt is
    # ~60 GB, more than the node's free disk).
    factors = [float(x) for x in str(getattr(cfg, "ss_sweep", "") or
                                     getattr(cfg, "supersample", 1.0) or 1.0
                                     ).split(",") if x.strip()]
    # None unless --force_ut: keeps the classic EWA path bit-identical.
    radial_ut = make_radial(scene, cfg, device)
    for S in factors:
        sfx = suffix + ("" if S <= 1.0 else f"_ss{S:g}")
        out_pin = os.path.join(cfg.result_dir, f"test_renders_pinhole{sfx}")
        os.makedirs(out_pin, exist_ok=True)
        out_dist = None
        if scene.need_undistort and cfg.redistort:
            out_dist = os.path.join(cfg.result_dir, f"test_renders_redistort{sfx}")
            os.makedirs(out_dist, exist_ok=True)

        for tp in scene.test_poses:
            Wt, Ht = tp.width, tp.height
            Kt = _test_K(cfg, scene, tp)

            # padded canvas so redistortion never samples outside the render
            pad_x = pad_y = 0
            mapx = mapy = None
            if out_dist is not None:
                mapx, mapy = scene.redistort_map(tp.K, Wt, Ht)
                pad_x = int(np.ceil(max(0, -mapx.min(), mapx.max() - (Wt - 1)))) + 2
                pad_y = int(np.ceil(max(0, -mapy.min(), mapy.max() - (Ht - 1)))) + 2
                Kt[0, 2] += pad_x
                Kt[1, 2] += pad_y

            viewmat = torch.from_numpy(tp.w2c).float().to(device)[None]
            Kt_t = torch.from_numpy(Kt).float().to(device)[None]
            mo, qo = _test_pose_overrides(
                scene, splats, pose_res, tp,
                getattr(cfg, "test_pose_interp", ""), viewmat, device)
            colors = _render_ss(splats, viewmat, Kt_t,
                                Wt + 2 * pad_x, Ht + 2 * pad_y,
                                cfg, radial_ut, S, means_o=mo, quats_o=qo)

            if bil_grids is not None and cfg.test_bilagrid != "none":
                colors = _test_bilagrid_correct(cfg, scene, bil_grids, colors, tp,
                                                pad_x=pad_x, pad_y=pad_y,
                                                device=device)

            img = (colors[0].cpu().numpy() * 255).round().astype(np.uint8)

            # crop padding -> pinhole render at exactly (Wt, Ht)
            img_pin = img[pad_y : pad_y + Ht, pad_x : pad_x + Wt]
            save_image(os.path.join(out_pin, tp.image_name), img_pin)

            if out_dist is not None:
                img_d = cv2.remap(img, mapx + pad_x, mapy + pad_y,
                                  interpolation=_REDISTORT_INTERP,
                                  borderMode=cv2.BORDER_REPLICATE)
                save_image(os.path.join(out_dist, tp.image_name), img_d)

        print(f"[render] {len(scene.test_poses)} test poses (ss={S:g}) -> {out_pin}"
              + (f" and {out_dist}" if out_dist else ""))


_TB_RE = re.compile(r"^(t?)blend(\d*)(?:p([0-9.]+))?$")


def _parse_test_bilagrid(tb):
    """"blend4" -> (4, spatial, p=1); "tblend8" -> (8, temporal, p=1);
    "blend8p2" -> (8, spatial, p=2); "blend8p0" -> plain mean of 8.

    Legacy names kept verbatim so every number already in EXPERIMENTS.md still
    means what it says: blend=3, nn=1, temporal=1 temporal, tblend=2 temporal.
    """
    if tb in ("nn", "temporal"):
        return 1, ("temporal" if tb == "temporal" else "spatial"), 1.0
    m = _TB_RE.match(tb)
    if not m:
        raise ValueError(f"unknown test_bilagrid mode {tb!r}")
    t, kk, pp = m.groups()
    mode = "temporal" if t else "spatial"
    k = int(kk) if kk else (2 if t else 3)
    return k, mode, (float(pp) if pp is not None else 1.0)


def _test_bilagrid_correct(cfg, scene, bil_grids, colors, tp, pad_x, pad_y, device):
    """Apply exposure correction to a test render by blending the bilateral
    grids of nearby train views. xy coords are normalized w.r.t. the UNPADDED
    frame the grids were trained on; padded pixels fall slightly outside [0,1]
    (border-clamped by grid_sample)."""
    Hp, Wp = colors.shape[1:3]
    gy, gx = torch.meshgrid(
        (torch.arange(Hp, device=device) - pad_y + 0.5) / tp.height,
        (torch.arange(Wp, device=device) - pad_x + 0.5) / tp.width,
        indexing="ij")
    xy = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    k, nn_mode, p = _parse_test_bilagrid(cfg.test_bilagrid)
    nn_idx, nn_w = scene.nearest_train_views(tp, k=k, mode=nn_mode, p=p)
    blended = torch.zeros_like(colors)
    for i, w in zip(nn_idx, nn_w):
        gid = int(scene.train_orig_idx[int(i)])  # grids indexed by full list
        out = bg_slice(bil_grids, xy, colors,
                       torch.tensor([gid], device=device))["rgb"]
        blended += float(w) * out
    return blended.clamp(0, 1)


def _test_ppisp_correct(ppisp, colors, Wt, Ht):
    """Novel-view correction via the PPISP controller (frame_idx=-1): no GT
    frame index available at test time, so it predicts exposure/color from
    the rendered radiance itself; per-camera vignetting/CRF still apply."""
    out = ppisp(colors[0], resolution=(Wt, Ht), camera_idx=0, frame_idx=-1)
    return out.unsqueeze(0).clamp(0, 1)


@torch.no_grad()
def _gaussian_blur(t, sigma):
    """Separable Gaussian blur on [1,3,H,W]."""
    rad = max(1, int(round(sigma * 2)))
    xs = torch.arange(-rad, rad + 1, device=t.device, dtype=t.dtype)
    k = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    k = (k / k.sum())
    kx = k.view(1, 1, 1, -1).expand(3, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(3, 1, -1, 1)
    t = F.conv2d(t, kx, padding=(0, rad), groups=3)
    t = F.conv2d(t, ky, padding=(rad, 0), groups=3)
    return t


def _unsharp(t, amount, sigma):
    """Unsharp mask: add back amount*(detail) where detail=t-blur(t)."""
    if amount <= 0:
        return t
    return (t + amount * (t - _gaussian_blur(t, sigma))).clamp(0, 1)


def _render_ss(splats, viewmat, Kt, Wt, Ht, cfg, radial, S,
               scale_mult=1.0, scale_comp="none",
               means_o=None, quats_o=None):
    """Render at S x resolution, then box-average down to (Wt, Ht).

    3DGUT shoots one ray per pixel, so thin high-contrast structure (the BTS
    lattice masts and antennas, roof edges) aliases badly — and gsplat's
    "antialiased" mode cannot be used under UT, so nothing currently
    compensates. Supersampling is the test-time fallback: K and the frame are
    scaled together, which leaves normalized (hence distorted) coordinates
    untouched, so the radial coeffs stay valid as-is.
    """
    if S <= 1.0:
        renders, _, _ = rasterize(splats, viewmat, Kt, Wt, Ht, cfg.sh_degree,
                                  cfg, render_depth=False, radial_coeffs=radial,
                                  scale_mult=scale_mult, scale_comp=scale_comp,
                                  means_override=means_o, quats_override=quats_o)
        return renders[..., :3].clamp(0, 1)
    Wl, Hl = int(round(Wt * S)), int(round(Ht * S))
    Ks = Kt.clone()
    Ks[:, 0, :] *= Wl / float(Wt)
    Ks[:, 1, :] *= Hl / float(Ht)
    renders, _, _ = rasterize(splats, viewmat, Ks, Wl, Hl, cfg.sh_degree,
                              cfg, render_depth=False, radial_coeffs=radial,
                              scale_mult=scale_mult, scale_comp=scale_comp,
                              means_override=means_o, quats_override=quats_o)
    hi = renders[..., :3].clamp(0, 1).permute(0, 3, 1, 2)
    lo = F.interpolate(hi, size=(Ht, Wt), mode="area")
    return lo.permute(0, 2, 3, 1).contiguous()


def _parse_tp_mode(mode):
    """'interp2' -> k=2 temporal neighbours; 'interp3' -> k=3; ''/'none' -> off."""
    m = str(mode or "").strip()
    if m in ("", "none", "0"):
        return 0
    if m.startswith("interp"):
        return max(1, int(m[len("interp"):] or 2))
    raise ValueError(f"unknown test_pose_interp mode {mode!r}")


def _test_pose_overrides(scene, splats, pose_res, tp, mode, viewmat, device):
    """Pose correction for a test/holdout view: the per-train-view SE(3)
    residuals learned by --pose_opt are interpolated (1/d temporal weights)
    from the k nearest TRAIN frames and applied as the same world-space
    transform training used. The view itself is always excluded, so a holdout
    probe measures exactly what test deployment would get. GT-free: uses only
    train-side corrections + frame numbering.

    pose_res rows follow the TRAINING run's train-view order: a full-data ckpt
    (rows == n_total_views) is indexed through train_orig_idx, a holdout-run
    ckpt (rows == len(train_metas)) directly; any other shape -> skip loudly."""
    k = _parse_tp_mode(mode)
    if k <= 0 or pose_res is None:
        return None, None
    from dataset import frame_index
    ti = frame_index(tp.image_name)
    if ti is None:
        return None, None
    full = pose_res.shape[0] == getattr(scene, "n_total_views", -1)
    if not full and pose_res.shape[0] != len(scene.train_metas):
        print(f"[tpi] pose_res rows {pose_res.shape[0]} match neither full "
              f"({getattr(scene, 'n_total_views', '?')}) nor train "
              f"({len(scene.train_metas)}) -> no correction", flush=True)
        return None, None
    cands = []
    for j, m in enumerate(scene.train_metas):
        fj = frame_index(m.name)
        if fj is None or fj == ti:
            continue
        row = int(scene.train_orig_idx[j]) if full else j
        cands.append((abs(fj - ti), row))
    if not cands:
        return None, None
    cands.sort()
    top = cands[:k]
    w = np.array([1.0 / (d + 1e-8) for d, _ in top])
    w = w / w.sum()
    xi = torch.zeros(6, dtype=torch.float32)
    for wi, (_, row) in zip(w, top):
        xi += float(wi) * pose_res[row].detach().cpu().float()
    xi = xi.to(device)
    R = _so3_exp(xi[:3])
    bot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    dT = torch.cat([torch.cat([R, xi[3:].unsqueeze(-1)], dim=1), bot], dim=0)
    vm = viewmat[0]
    Rv, tv = vm[:3, :3], vm[:3, 3]
    inv = torch.cat([torch.cat([Rv.T, (-Rv.T @ tv).unsqueeze(-1)], dim=1),
                     bot], dim=0)
    Mc = inv @ dT @ vm
    means_o = splats["means"] @ Mc[:3, :3].T + Mc[:3, 3]
    quats_o = _qmul(_rotmat_to_quat(Mc[:3, :3]), splats["quats"])
    return means_o, quats_o


def render_test_direct(cfg, scene: SceneData, splats, bil_grids, device, suffix="", ppisp=None, pose_res=None):
    """3DGUT: render test poses with radial distortion modeled in-camera."""
    out = os.path.join(cfg.result_dir, f"test_renders_direct{suffix}")
    os.makedirs(out, exist_ok=True)
    radial = make_radial(scene, cfg, device)

    for tp in scene.test_poses:
        Wt, Ht = tp.width, tp.height
        viewmat = torch.from_numpy(tp.w2c).float().to(device)[None]
        Kt = torch.from_numpy(_test_K(cfg, scene, tp)).float().to(device)[None]
        mo, qo = _test_pose_overrides(
            scene, splats, pose_res, tp,
            getattr(cfg, "test_pose_interp", ""), viewmat, device)
        colors = _render_ss(splats, viewmat, Kt, Wt, Ht, cfg, radial,
                            getattr(cfg, "supersample", 1.0),
                            getattr(cfg, "render_scale", 1.0),
                            getattr(cfg, "render_scale_comp", "none"),
                            means_o=mo, quats_o=qo)

        if ppisp is not None:
            colors = _test_ppisp_correct(ppisp, colors, Wt, Ht)
        elif bil_grids is not None and cfg.test_bilagrid != "none":
            colors = _test_bilagrid_correct(cfg, scene, bil_grids, colors, tp,
                                            pad_x=0, pad_y=0, device=device)

        img = (colors[0].cpu().numpy() * 255).round().astype(np.uint8)
        name = tp.image_name
        if getattr(cfg, "render_png", 0):
            # ensemble members are an intermediate, not the deliverable — the
            # 350MiB cap only binds on the final zip, so keep them lossless and
            # let the averaging work on uncompressed pixels
            name = os.path.splitext(name)[0] + ".png"
        save_image(os.path.join(out, name), img)
    print(f"[render] {len(scene.test_poses)} test poses (direct/3DGUT) -> {out}")


@torch.no_grad()
def eval_holdout(cfg, scene: SceneData, splats, bil_grids, device, ppisp=None, pose_res=None):
    """Render held-out train views (exposure via blended train grids, exactly
    like test rendering) and score them against their original photos.
    Gives a private-scene proxy of the leaderboard metric."""
    assert scene.distorted or not scene.need_undistort, \
        "holdout eval needs distorted mode (or a distortion-free camera, " \
        "where the undistorted and original frames coincide)"
    import lpips as lpips_mod

    lp_alex = lpips_mod.LPIPS(net="alex").to(device)
    # VGG is the TRUE LB metric-net (confirmed 23/07): decisions gate on
    # SCORE_v below, alex kept only for continuity with older tables.
    lp_vgg = lpips_mod.LPIPS(net="vgg").to(device)
    Kt = torch.from_numpy(scene.K).float().to(device)[None]

    # --ss_sweep scores several supersampling factors off the SAME trained
    # model in one pass, so the comparison is free of run-to-run variance.
    sweep = getattr(cfg, "ss_sweep", "") or str(getattr(cfg, "supersample", 1.0))
    factors = [float(x) for x in str(sweep).split(",") if x.strip()]
    # every test-time knob can be scored off the same trained model, so sweep
    # the exposure-blend mode here too instead of paying for another training
    # run — K=32 was picked back on the round-1 pipeline (30k/2M, no ft, no
    # 3DGUT) and has never been re-checked against the current one.
    tb_modes = [m for m in (getattr(cfg, "tb_sweep", "") or
                            cfg.test_bilagrid).split(",") if m.strip()]
    scale_factors = [float(x) for x in (
        getattr(cfg, "render_scale_sweep", "") or
        str(getattr(cfg, "render_scale", 1.0))).split(",") if x.strip()]
    scale_comps = [x for x in (
        getattr(cfg, "render_scale_comp_sweep", "") or
        getattr(cfg, "render_scale_comp", "none")).split(",") if x.strip()]
    k1_scales = [float(x) for x in (
        getattr(cfg, "k1_scale_sweep", "") or
        str(getattr(cfg, "k1_scale", 1.0))).split(",") if x.strip()]
    k2_values = [float(x) for x in (
        getattr(cfg, "k2_sweep", "") or
        str(getattr(cfg, "k2", 0.0))).split(",") if x.strip()]
    tp_modes = [m.strip() for m in (
        getattr(cfg, "tp_sweep", "") or
        str(getattr(cfg, "test_pose_interp", "") or "none")).split(",")
        if m.strip()]

    best = None
    cases = ((S, tb, rscale, rcomp, k1_scale, k2, tpm)
             for S in factors for tb in tb_modes
             for rscale in scale_factors for rcomp in scale_comps
             for k1_scale in k1_scales for k2 in k2_values
             for tpm in tp_modes)
    for S, tb, rscale, rcomp, k1_scale, k2, tpm in cases:
        tag = ("" if S <= 1.0 else f"_ss{S:g}") + (
            "" if tb == cfg.test_bilagrid else f"_{tb}") + (
            "" if abs(rscale - 1.0) < 1e-8 else f"_rs{rscale:g}") + (
            "" if rcomp == "none" else f"_rsc-{rcomp}") + (
            "" if abs(k1_scale - 1.0) < 1e-8 else f"_k1x{k1_scale:g}") + (
            "" if abs(k2) < 1e-12 else f"_k2{k2:+g}") + (
            "" if tpm in ("", "none") else f"_tpi-{tpm}")
        out = os.path.join(cfg.result_dir, f"holdout_renders{tag}")
        os.makedirs(out, exist_ok=True)
        cfg_tb = copy.copy(cfg)
        cfg_tb.test_bilagrid = tb
        cfg_tb.k1_scale = k1_scale
        cfg_tb.k2 = k2
        radial = make_radial(scene, cfg_tb, device)
        rows = []
        # --post_sweep: score test-time unsharp variants off the SAME render
        # (no retraining) to probe whether adding high-freq recovers LB score
        # (LPIPS-VGG). id = identity baseline. (amount, sigma) unsharp masks.
        _post_variants = []
        if getattr(cfg, "post_sweep", 0):
            _post_variants = [("us_a0.3_s1", 0.3, 1.0), ("us_a0.6_s1", 0.6, 1.0),
                              ("us_a1.0_s1", 1.0, 1.0), ("us_a0.6_s2", 0.6, 2.0),
                              ("us_a1.0_s2", 1.0, 2.0)]
        _post_rows = {v[0]: [] for v in _post_variants}
        for meta, gt_u8 in zip(scene.holdout_metas, scene.holdout_images):
            tp = TestPose(meta.name, meta.w2c(), scene.K,
                          scene.width, scene.height)
            viewmat = torch.from_numpy(tp.w2c).float().to(device)[None]
            mo, qo = _test_pose_overrides(scene, splats, pose_res, tp,
                                          tpm, viewmat, device)
            colors = _render_ss(splats, viewmat, Kt, scene.width,
                                scene.height, cfg, radial, S,
                                rscale, rcomp, means_o=mo, quats_o=qo)
            if ppisp is not None:
                colors = _test_ppisp_correct(ppisp, colors,
                                             scene.width, scene.height)
            elif bil_grids is not None and tb != "none":
                colors = _test_bilagrid_correct(cfg_tb, scene, bil_grids,
                                                colors, tp, pad_x=0, pad_y=0,
                                                device=device)
            gt = torch.from_numpy(gt_u8).to(device).float() / 255.0
            t_pr = colors[0].permute(2, 0, 1)[None]
            t_gt = gt.permute(2, 0, 1)[None]
            mse = torch.mean((t_gt - t_pr) ** 2).item()
            psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
            ssim = ssim_torch(t_pr, t_gt).item()
            la = lp_alex(t_pr, t_gt, normalize=True).item()
            # The best reconstruction of BTC's reported LPIPS uses VGG
            # normalize=False. Keep both conventions until organiser code
            # or a fresh submission gives an independent confirmation.
            # Keep the old normalize=True value as an explicit audit field.
            lv_t = lp_vgg(t_pr, t_gt, normalize=True).item()
            lv = lp_vgg(t_pr, t_gt, normalize=False).item()
            rows.append(dict(name=meta.name, psnr=psnr, ssim=ssim,
                             lpips_alex=la, lpips_vgg=lv,
                             lpips_vgg_normtrue=lv_t))
            for vname, amt, sig in _post_variants:
                tv = _unsharp(t_pr, amt, sig)
                mse_v = torch.mean((t_gt - tv) ** 2).item()
                _post_rows[vname].append(dict(
                    psnr=10 * math.log10(1.0 / max(mse_v, 1e-12)),
                    ssim=ssim_torch(tv, t_gt).item(),
                    lpips_vgg=lp_vgg(tv, t_gt, normalize=False).item()))
            # same reason as the test path: when these renders are going to
            # be averaged across ensemble members, the averaging must see
            # uncompressed pixels or the curve measures JPEG noise too
            hname = meta.name
            if getattr(cfg, "render_png", 0):
                hname = os.path.splitext(hname)[0] + ".png"
            save_image(os.path.join(out, hname),
                       (colors[0].cpu().numpy() * 255).round().astype(np.uint8))

        agg = {k: float(np.mean([r[k] for r in rows]))
               for k in ("psnr", "ssim", "lpips_alex", "lpips_vgg",
                         "lpips_vgg_normtrue")}
        agg["score"] = (0.4 * (1 - agg["lpips_alex"]) + 0.3 * agg["ssim"]
                        + 0.3 * min(agg["psnr"] / 50.0, 1.0))
        # Reconstructed LB formula (matches the known score to ~4 digits):
        # LPIPS/SSIM in percent, PSNR in dB.
        agg["score_vgg"] = (40.0 - 0.4 * 100 * agg["lpips_vgg"]
                            + 0.3 * 100 * agg["ssim"] + 0.6 * agg["psnr"])
        agg["n_images"] = len(rows)
        agg["supersample"] = S
        agg["test_bilagrid"] = tb
        agg["render_scale"] = rscale
        agg["render_scale_comp"] = rcomp
        agg["k1_scale"] = k1_scale
        agg["k2"] = k2
        agg["test_pose_interp"] = tpm
        with open(os.path.join(cfg.result_dir,
                               f"metrics_holdout{tag}.json"), "w") as f:
            json.dump({"aggregate": agg, "per_image": rows}, f, indent=2)
        print(f"HOLDOUT ss={S:g} tb={tb} rs={rscale:g} rsc={rcomp} "
              f"k1x={k1_scale:g} k2={k2:+g} "
              f"n={agg['n_images']} "
              f"psnr={agg['psnr']:.3f} ssim={agg['ssim']:.4f} "
              f"lpips_a={agg['lpips_alex']:.4f} "
              f"lpips_v={agg['lpips_vgg']:.4f} "
              f"lpips_vT={agg['lpips_vgg_normtrue']:.4f} "
              f"SCORE_a={agg['score'] * 100:.3f} "
              f"SCORE_v={agg['score_vgg']:.3f}", flush=True)
        # post-process (unsharp) sweep vs the identity baseline above
        for vname, prs in _post_rows.items():
            if not prs:
                continue
            pv = {k: float(np.mean([r[k] for r in prs]))
                  for k in ("psnr", "ssim", "lpips_vgg")}
            sv = (40.0 - 0.4 * 100 * pv["lpips_vgg"]
                  + 0.3 * 100 * pv["ssim"] + 0.6 * pv["psnr"])
            d = sv - agg["score_vgg"]
            print(f"HOLDOUT_POST ss={S:g} tb={tb} rs={rscale:g} "
                  f"rsc={rcomp} variant={vname} "
                  f"psnr={pv['psnr']:.3f} ssim={pv['ssim']:.4f} "
                  f"lpips_v={pv['lpips_vgg']:.4f} SCORE_v={sv:.3f} "
                  f"d_vs_id={d:+.3f}", flush=True)
        if best is None or agg["score_vgg"] > best[6]:
            best = (S, tb, rscale, rcomp, k1_scale, k2, agg["score_vgg"])
    if (len(factors) * len(tb_modes) * len(scale_factors) * len(scale_comps)
            * len(k1_scales) * len(k2_values)) > 1:
        print(f"HOLDOUT_BEST ss={best[0]:g} tb={best[1]} "
              f"rs={best[2]:g} rsc={best[3]} "
              f"k1x={best[4]:g} k2={best[5]:+g} "
              f"SCORE_v={best[6]:.3f}", flush=True)


def save_image(path, img_rgb_u8):
    """Save honoring the extension in image_name (.png lossless / .jpg q98)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(path, cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 98])
    else:
        imageio.imwrite(path, img_rgb_u8)


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", required=True)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--cap_max", type=int, default=1_500_000)
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--sh_degree_interval", type=int, default=1000)
    p.add_argument("--ssim_lambda", type=float, default=0.2)
    p.add_argument("--init_opacity", type=float, default=0.5)
    p.add_argument("--init_scale", type=float, default=1.0)
    p.add_argument("--means_lr", type=float, default=1.6e-4)
    p.add_argument("--opacity_reg", type=float, default=0.01)
    p.add_argument("--scale_reg", type=float, default=0.01)
    p.add_argument("--aniso_reg", type=float, default=0.0,
                   help="0 = off. Phạt relu(max_scale/min_scale - aniso_max) trung "
                        "bình trên mọi Gaussian — trị 'nổ kim' khi nhìn dọc trục "
                        "facade (smbu §19.47). Thử 0.001-0.01.")
    p.add_argument("--aniso_max", type=float, default=10.0,
                   help="tỉ lệ trục tối đa không bị phạt bởi aniso_reg")
    # 03/09 (x58) — ba "luật vật lý" cho mặt đứng nhà cao (x57: ray_spread 1,8×, Gaussian 2,6× thưa, depth hỗn loạn):
    p.add_argument("--ray_var_weight", type=float, default=0.0,
                   help="0 = off. Luật MẶT ĐỤC: một tia chỉ có MỘT độ sâu — phạt Var[z]/E[z]² dọc tia (distortion "
                        "loss kiểu Mip-NeRF360/2DGS), render z và z² làm màu ở 1/ray_var_down res, pixel alpha>0,5.")
    p.add_argument("--ray_var_down", type=int, default=4, help="hệ số hạ res cho render z/z² của ray_var")
    p.add_argument("--ray_var_start", type=int, default=0, help="bước bắt đầu phạt ray_var (0 = ngay)")
    p.add_argument("--manhattan_weight", type=float, default=0.0,
                   help="0 = off. Luật TRỌNG LỰC: pháp tuyến đĩa mỏng (smid/smin>3) phải ∥ hoặc ⟂ g (mái/đất hay "
                        "tường); g = trục phương sai nhỏ nhất của đám mây SfM. Phạt op·min(|n·g|, 1−|n·g|).")
    p.add_argument("--geo_consist_weight", type=float, default=0.0,
                   help="0 = off. Luật NHẤT QUÁN ĐA VIEW (hình học, KHÔNG photometric — khác vpc bị sụp cổng): "
                        "depth render view i chiếu sang camera train gần nhất j phải khớp depth render tại j; "
                        "|Δz|/z, cổng che khuất 0,1, ở 1/ray_var_down res.")
    p.add_argument("--train_list", default="",
                   help="file 1 tên ảnh train/dòng: giới hạn tập train (chuyên gia "
                        "cục bộ per-cluster). Chỉ dùng khi ft từ ckpt splats thuần.")
    p.add_argument("--scene_scale", type=float, default=0.0,
                   help="ép scene_scale (0 = tự tính từ cụm camera train); cần khi --train_list ít view")
    # MCMC densification knobs (dead-budget / allocation probes). Defaults
    # reproduce the hardcoded production behavior exactly (25/30 stop ratio,
    # 0.005 prune, 5e5 noise) so unset == prior runs, bit-for-bit.
    p.add_argument("--refine_stop_ratio", type=float, default=25.0 / 30.0)
    p.add_argument("--refine_every", type=int, default=100,
                   help="MCMC relocation/growth cadence. Larger values slow the "
                        "fixed 5%% growth staircase and let optimization settle "
                        "between topology changes.")
    p.add_argument("--min_opacity", type=float, default=0.005)
    p.add_argument("--relocate_invisible", type=int, default=0,
                   help="MCMC: coi Gaussian có alpha đỉnh × bù antialias < 1/255 (không "
                        "render được ở mọi hướng) là chết để relocate. Xem _install_mcmc_opacity_probe.")
    # PrismGS (2510.07830) — hai regularizer, mặc định 0 = tắt hoàn toàn.
    p.add_argument("--ms_weight", type=float, default=0.0,
                   help="giám sát đa tỉ lệ kiểu kim tự tháp (PrismGS a). 0.5 là điểm khởi đầu.")
    p.add_argument("--ms_levels", type=int, default=3,
                   help="số mức hạ mẫu 2x cho --ms_weight")
    p.add_argument("--size_reg", type=float, default=0.0,
                   help="phạt Gaussian NHỎ hơn size_floor (PrismGS b) — ngược chiều --scale_reg")
    p.add_argument("--size_floor_px", type=float, default=1.0,
                   help="sàn kích thước tính theo pixel, quy ra mét bằng GSD của cảnh")
    p.add_argument("--test_use_train_K", type=int, default=0,
                   help="render test bằng tâm quang THẬT trong cameras.bin thay vì "
                        "cx,cy=W/2,H/2 mà test_poses.csv ép. Xem _test_K.")
    p.add_argument("--opacity_probe_every", type=int, default=5,
                   help="in phân bố opacity mỗi N lần refine (0=mỗi lần, -1=tắt hẳn "
                        "kể cả guard alive=0). Xem _install_mcmc_opacity_probe.")
    p.add_argument("--noise_lr", type=float, default=5e5)
    p.add_argument("--scale_floor_rel", type=float, default=0.0,
                   help="sàn cho scale, tính theo TỈ LỆ của scene_scale (0 = tắt, "
                        "giữ nguyên mọi mốc cũ). 1e-7 trên cuhk = 5.8e-5 đơn vị thế "
                        "giới ≈ 1/2800 pixel — thấp hơn mọi thứ nhìn thấy được, "
                        "nhưng chặn được scale suy biến ~1e-9 gây NaN (§19.24).")
    # --- chọn CÁCH CHỌN/NHÂN GAUSSIAN (selection method) -------------------
    # "mcmc"    = 3DGS-MCMC (relocate theo khối lượng opacity) — mọi kết quả
    #             vòng 2 tới 15/08 đều nằm ở đây.
    # "default" = adaptive density control của 3DGS gốc (clone/split theo
    #             gradient màn hình + reset opacity). Kèm --adc_absgrad thành
    #             AbsGS, kèm --adc_revised_opacity thành Revising Densification.
    # cap_max vẫn có tác dụng ở chế độ "default" nhưng chỉ như TRẦN VRAM.
    p.add_argument("--strategy", type=str, default="mcmc",
                   choices=["mcmc", "default"])
    p.add_argument("--adc_grow_grad2d", type=float, default=0.0002,
                   help="ngưỡng gradient màn hình để nhân đôi. 3DGS gốc 2e-4. "
                        "AbsGS dùng |grad| nên tổng lớn hơn -> ngưỡng phải cao hơn "
                        "(bài báo dùng ~4e-4..8e-4), nếu không sẽ mọc bùng nổ.")
    p.add_argument("--adc_absgrad", type=int, default=0,
                   help="1 = AbsGS (2404.10484): cộng |grad| thay vì cộng grad có "
                        "dấu, để gradient của các vùng thiếu tái tạo không tự triệt tiêu")
    p.add_argument("--adc_revised_opacity", type=int, default=0,
                   help="1 = Revising Densification (2404.06109): công thức opacity "
                        "cho con sau clone/split bảo toàn tích luỹ alpha")
    p.add_argument("--adc_prune_opa", type=float, default=0.005)
    p.add_argument("--adc_grow_scale3d", type=float, default=0.01)
    p.add_argument("--adc_prune_scale3d", type=float, default=0.1)
    p.add_argument("--adc_reset_every", type=int, default=3000,
                   help="chu kỳ reset opacity của 3DGS gốc. 0 = tắt.")
    p.add_argument("--adc_refine_every", type=int, default=100)
    p.add_argument("--adc_pixel_gs", type=int, default=0,
                   help="1 = Pixel-GS (2403.15530): trung bình gradient có trọng "
                        "số theo diện tích chiếu, thay vì mỗi view một phiếu")
    p.add_argument("--adc_budget", type=int, default=0,
                   help="Taming-3DGS (2406.15643): ngân sách Gaussian CUỐI cố "
                        "định. >0 thì ngưỡng gradient được suy ra mỗi lần refine "
                        "để đi đúng đường tới đích. 0 = tắt (ngưỡng cố định, số "
                        "Gaussian rơi vào đâu thì rơi).")
    p.add_argument("--adc_stop_ratio", type=float, default=0.5,
                   help="dừng densify ở bao nhiêu phần của max_steps. 3DGS gốc "
                        "dừng ở 15000/30000 = 0.5; MCMC của ta dùng 25/30.")
    # error-guided densification (AbsGS-weighted MCMC relocation/growth).
    # 0 == plain opacity-mass MCMC (unchanged). >0 biases budget to high-|grad|
    # (under-reconstructed) Gaussians. Only affects base (ft freezes topology).
    p.add_argument("--err_guided", type=float, default=0.0)
    # J122-D: topology-only projected-UT AbsGrad + ImprovedGS long-axis split.
    # Defaults are all off, preserving J73 and every production recipe.
    p.add_argument("--las_quota", type=float, default=0.0,
                   help="0 = off/plain MCMC. In (0,1] reserves this fraction "
                        "of each capped MCMC growth event for AbsGrad-selected "
                        "long-axis splits (J122-D: 0.25 or 0.50).")
    p.add_argument("--las_signal_every", type=int, default=10,
                   help="run the isolated projected-UT AbsGrad signal pass every "
                        "N base iterations; does not change production gradients")
    p.add_argument("--las_preflight_calls", type=int, default=1,
                   help="number of initial auxiliary calls that must expose a "
                        "finite, nonzero means2d.absgrad signal")
    p.add_argument("--las_distance", type=float, default=0.45,
                   help="ImprovedGS long-axis child offset ratio")
    p.add_argument("--las_opacity_reduction", type=float, default=0.6,
                   help="post-sigmoid opacity multiplier for LAS children")
    p.add_argument("--las_min_radius_px", type=float, default=2.0,
                   help="minimum observed projected radius for a LAS parent")
    p.add_argument("--las_min_anisotropy", type=float, default=1.0,
                   help="minimum max/min 3D scale ratio for a LAS parent")
    p.add_argument("--roi_ss", type=float, default=1.0,
                   help="foveated supersample factor on the orbit-centre ROI "
                        "window (1=off). Adds a higher-res aux supervision term "
                        "on the centred subject (tower/cables).")
    p.add_argument("--roi_frac", type=float, default=0.5,
                   help="ROI window size as a fraction of the frame, centred on "
                        "the projected orbit-centre.")
    p.add_argument("--roi_w", type=float, default=1.0,
                   help="weight of the ROI supersampled aux loss.")
    p.add_argument("--roi_lpips", type=int, default=1,
                   help="include LPIPS in the ROI aux loss (0/1).")
    p.add_argument("--depth_weight", type=float, default=1e-2)
    p.add_argument("--mono_depth_dir", default="",
                   help="thư mục .npy disparity DA-v2 per train image (half-res, "
                        "từ tools/depth_precompute.py). Bật cùng weight.")
    p.add_argument("--mono_depth_weight", type=float, default=0.0,
                   help="0 = off. L1(disparity render vs DA-v2 fit scale/shift "
                        "theo SfM per-image) × scene_scale — H3DGS-style, thuốc "
                        "chính cho facade-nát/vùng-trống (§19.47 E3). Thử 0.02-0.2.")
    p.add_argument("--mono_depth_end", type=int, default=0,
                   help="weight giảm tuyến tính về 0 tại step này (0 = max_steps)")
    p.add_argument("--mono_depth_start", type=int, default=0,
                   help="bật mono loss từ step này — chống vòng phản hồi dương "
                        "'fog neo fog' ở view cận cảnh khi render đầu base còn là "
                        "sương (lfls 0089/0090 sập 42→21, §19.51)")
    p.add_argument("--lpips_weight", type=float, default=0.1)
    p.add_argument("--lpips_start", type=int, default=15000)
    p.add_argument("--lpips_ramp", type=int, default=0,
                   help="linearly ramp LPIPS from zero to lpips_weight over this "
                        "many steps after lpips_start (0 = historical hard start)")
    p.add_argument("--lpips_crop", type=int, default=0,
                   help="0 = LPIPS full-image (đường cũ). >0 = cắt k crop "
                        "ngẫu nhiên cạnh này (vd 512) trước khi vào VGG — né "
                        "tường ~80GB @21MP, render vẫn full-frame.")
    p.add_argument("--lpips_crops", type=int, default=8,
                   help="số crop mỗi iter khi --lpips_crop > 0")
    p.add_argument("--distortion_polish_steps", type=int, default=0,
                   help="linearly decay LPIPS to zero over the final N steps, "
                        "allowing a distortion-only Pareto polish (0 = off)")
    p.add_argument("--lpips_net", default="vgg", choices=["vgg", "alex"])
    p.add_argument("--lpips_normalize", type=int, default=1, choices=[0, 1],
                   help="LPIPS input convention used for training: 0 matches "
                        "the reconstructed leaderboard VGG convention on "
                        "[0,1] inputs; 1 keeps "
                        "the historical [0,1] to [-1,1] remapping.")
    p.add_argument("--bilagrid", type=int, default=1)
    p.add_argument("--use_ppisp", type=int, default=0,
                   help="1 = learned exposure/vignetting/color/CRF correction "
                        "+ test-time controller (nv-tlabs/ppisp), replaces bilagrid")
    p.add_argument("--ppisp_controller_ratio", type=float, default=0.8,
                   help="fraction of max_steps at which the PPISP controller "
                        "activates; set >1 to keep it inactive this phase "
                        "(e.g. long base run) and let a later ft phase warm it up")
    p.add_argument("--supersample", type=float, default=1.0,
                   help="render test/holdout views at this factor then box-average "
                        "down; counters UT aliasing on thin structure (gsplat's "
                        "antialiased mode is unavailable under UT). 1.0 = off")
    p.add_argument("--ss_train", type=float, default=1.0,
                   help="render TRAINING views at this factor and box-average to "
                        "native before the losses, so the model optimizes at the "
                        "supersampled operating point; pair with a matching "
                        "--supersample at render time. Integer factors only "
                        "(non-integer resampling measured harmful). 1.0 = off")
    p.add_argument("--ss_sweep", default="",
                   help="comma list of supersample factors to score in eval_holdout "
                        "off one trained model, e.g. '1,1.5,2'")
    p.add_argument("--render_scale", type=float, default=1.0,
                   help="test/holdout-only multiplier on every Gaussian scale; "
                        "<1 narrows projected footprints. Training is unchanged.")
    p.add_argument("--render_scale_sweep", default="",
                   help="comma list of test-time Gaussian scale multipliers to "
                        "score from one checkpoint, e.g. '0.9,0.95,1,1.05'")
    p.add_argument("--render_scale_comp", default="none",
                   choices=("none", "area", "volume"),
                   help="opacity compensation for render_scale: none, projected "
                        "area (1/s^2), or volume (1/s^3)")
    p.add_argument("--render_scale_comp_sweep", default="",
                   help="comma list of compensation modes for holdout sweep")
    p.add_argument("--tb_sweep", default="",
                   help="comma list of test_bilagrid modes to score alongside "
                        "--ss_sweep off the same model, e.g. 'blend16,blend32,blend64'")
    p.add_argument("--png", dest="render_png", type=int, default=0,
                   help="write renders as lossless .png instead of the .jpg the "
                        "csv names ask for. Use whenever the renders are an "
                        "intermediate that will be averaged across ensemble "
                        "members — JPEG before averaging costs real score.")
    p.add_argument("--post_sweep", type=int, default=0,
                   help="in eval_holdout, also score test-time unsharp variants "
                        "(HF post-process) vs identity to probe recoverable "
                        "LPIPS-VGG score without retraining.")
    p.add_argument("--sharp_gamma", type=float, default=0.0,
                   help="0 = off (uniform weighting). >0 downweights motion-blurred "
                        "train frames by (laplacian_var/median)^gamma; 1 = linear, "
                        "2 = aggressive. Only meaningful on handheld-video scenes.")
    p.add_argument("--sharp_floor", type=float, default=0.1,
                   help="minimum weight a blurry frame keeps, so it still "
                        "contributes coverage instead of being dropped outright")
    p.add_argument("--sharp_mode", default="hf",
                   help="hf = downweight only ssim+lpips (keep l1 full, so blurry "
                        "frames still teach colour/geometry); all = whole photometric loss")
    p.add_argument("--edge_weight", type=float, default=0.0,
                   help="0 = off (uniform L1). >0 upweights the per-pixel L1 by GT "
                        "gradient magnitude (renormalised to mean 1) so the "
                        "optimiser spends capacity on thin high-contrast structures "
                        "(BTS lattice/cables/roof edges) instead of averaging them out.")
    p.add_argument("--mse_weight", type=float, default=0.0,
                   help="additional masked RGB MSE weight; directly anchors the "
                        "PSNR objective while perceptual fine-tuning is active")
    p.add_argument("--lowpass_weight", type=float, default=0.0,
                   help="additional masked low-pass RGB L1 weight; preserves coarse "
                        "colour/structure while LPIPS optimizes fine appearance")
    p.add_argument("--lowpass_kernel", type=int, default=5,
                   help="positive odd box-filter width for lowpass_weight")
    p.add_argument("--mip_filter", type=float, default=0.0,
                   help="0 = off. >0 enables the Mip-Splatting 3D smoothing "
                        "filter with this factor (paper: 0.2): each Gaussian's "
                        "scale is dilated in quadrature by factor*(nearest-view "
                        "depth/focal) with opacity compensation, so no primitive "
                        "is sub-pixel -> removes aliasing floaters on thin BTS "
                        "lattice/cables. Render-time reparam, UT-compatible.")
    p.add_argument("--test_bilagrid", default="blend",
                   help="none|nn|temporal|tblend|blend|blendK (K = #views to average)")
    p.add_argument("--test_pose_interp", default="",
                   help="''/none = off (default); interpK = correct each "
                        "test/holdout pose with the SE(3) residual interpolated "
                        "from the K temporally-nearest train views' --pose_opt "
                        "corrections (self excluded). Needs pose_adj_res in the "
                        "ckpt (or a --pose_opt run).")
    p.add_argument("--tp_sweep", default="",
                   help="comma list of test_pose_interp modes scored off the "
                        "same model in eval_holdout, e.g. 'none,interp2,interp3'")
    p.add_argument("--antialiased", type=int, default=1)
    p.add_argument("--redistort", type=int, default=1)
    p.add_argument("--distorted", type=int, default=0,
                   help="1 = train on original distorted images via 3DGUT")
    p.add_argument("--mask_dir", default="",
                   help="root of transient masks: <mask_dir>/<scene>/<image_name>.png")
    p.add_argument("--holdout_every", type=int, default=0,
                   help="hold out every Nth train view as pseudo-test and score it")
    p.add_argument("--holdout_offset", type=int, default=None,
                   help="residue class for --holdout_every; unset preserves the "
                        "historical N//2 split. Use a locked second offset to "
                        "audit selection overfitting.")
    p.add_argument("--multi_camera", default="auto",
                   choices=["auto", "dominant", "error"],
                   help="sparse/0 liệt kê >1 camera: auto = gộp các entry trùng nhau "
                        "(ContextCapture/GauU-Scene ghi 1 entry/ảnh) và chỉ bỏ nhóm lạc "
                        "<2%%; dominant = luôn giữ nhóm lớn nhất; error = như cũ.")
    p.add_argument("--camera_tol_px", type=float, default=None,
                   help="hai camera lệch dưới ngần này pixel thì coi là MỘT (mặc định 3.0). "
                        "GauU-Scene: ContextCapture hiệu chuẩn lại mỗi chuyến bay nên một "
                        "scene có 4-13 nhóm lệch nhau <8 px ở 5472x3648.")
    p.add_argument("--skip_test_render", type=int, default=0)
    p.add_argument("--force_ut", type=int, default=0,
                   help="1 = render through gsplat's unscented transform "
                        "(with_ut/with_eval3d) even when the camera has no "
                        "distortion, i.e. drop the local-affine projection "
                        "approximation. Requires --depth_weight 0 (UT + depth "
                        "render core-dumps in gsplat 1.5.3) and disables the "
                        "'antialiased' rasterize mode, so pair with "
                        "--ss_sweep to compensate.")
    p.add_argument("--ckpt_every", type=int, default=0,
                   help="lưu ckpt_mid.pt mỗi N bước (0=tắt) — bảo hiểm OOM cho "
                        "run dài; hồi phục bằng --init_ckpt (topology đóng băng, "
                        "an toàn khi ckpt ≥ ADC refine_stop)")
    p.add_argument("--save_ckpt", type=int, default=1,
                   help="0 = do not write ckpt.pt. A cap-96M ckpt is ~60 GB on "
                        "a node disk shared with the whole cluster.")
    p.add_argument("--init_ckpt", default="",
                   help="ckpt.pt to fine-tune from (freezes topology: no MCMC)")
    p.add_argument("--pseudo_dir", default="",
                   help="distill-back: thư mục view giả (img_*.png|jpg + meta.pt "
                        "chứa c2w) đã qua fixer; dùng làm giám sát phụ")
    p.add_argument("--pseudo_prob", type=float, default=0.3,
                   help="xác suất mỗi bước train thêm 1 view giả")
    p.add_argument("--pseudo_lambda", type=float, default=0.3,
                   help="trọng số loss nhánh pseudo (L1+SSIM, không LPIPS)")
    p.add_argument("--pseudo_scale", type=float, default=1.0,
                   help="render pseudo ở tỉ lệ này so với native (1.0 = full)")
    p.add_argument("--lr_scale", type=float, default=1.0,
                   help="multiply ALL learning rates (use ~0.1 for fine-tune)")
    p.add_argument("--vpc_weight", type=float, default=0.0,
                   help="virtual-pose photometric consistency: render ở pose giữa 2 frame kề, khớp với warp của 2 ảnh kề (0 = tắt)")
    p.add_argument("--vpc_scale", type=float, default=0.5, help="tỉ lệ res render pose ảo")
    p.add_argument("--vpc_tau", type=float, default=0.08, help="ngưỡng đồng thuận 2 warp (L1 màu)")
    p.add_argument("--vpc_prob", type=float, default=1.0, help="xác suất áp VPC mỗi bước")
    p.add_argument("--sh_reg", type=float, default=0.0,
                   help="L2 lên hệ số SH bậc ≥1 (view-dependent regularizer, Mind-the-Gap). 0 = tắt")
    p.add_argument("--depth_flat_thr", type=float, default=0.0,
                   help="bỏ điểm track COLMAP ở vùng phẳng (Sobel blur σ4 @~/4 < thr, ~20 = nước). 0 = tắt")
    p.add_argument("--wd_weight", type=float, default=0.0,
                   help="WD-R (2603.23297): trọng số Wasserstein Distortion trên feature "
                        "VGG16, bật từ --lpips_start, dùng crop của --lpips_crop. 0 = tắt")
    p.add_argument("--wd_sigma", type=float, default=4.0,
                   help="σ (px ở tầng pixel) của pool Gaussian tính thống kê cục bộ WD-R")
    p.add_argument("--batch_views", type=int, default=1,
                   help="gộp gradient B view rồi mới bước Adam (H1, Grendel-GS). "
                        "max_steps vẫn đếm theo VIEW; kèm --lr_scale sqrt(B); "
                        "noise_lr MCMC tự chia B")
    p.add_argument("--k1_scale", type=float, default=1.0,
                   help="multiply COLMAP k1 (distortion refinement sweep)")
    p.add_argument("--k1_scale_sweep", default="",
                   help="comma-separated render-only k1 multipliers evaluated "
                        "from the same checkpoint/holdout")
    p.add_argument("--k2", type=float, default=0.0,
                   help="add second-order radial coeff (COLMAP SIMPLE_RADIAL has none)")
    p.add_argument("--k2_sweep", default="",
                   help="comma-separated render-only k2 values evaluated from "
                        "the same checkpoint/holdout")
    p.add_argument("--pose_opt", type=int, default=0,
                   help="1 = learn SE(3) residuals on train poses (test poses fixed)")
    p.add_argument("--pose_opt_lr", type=float, default=1e-5)
    p.add_argument("--pose_opt_start", type=int, default=0, help="bước bắt đầu cập nhật pose (trước đó chỉ tích luỹ/bỏ gradient) — 03/09 x77")
    p.add_argument("--pose_opt_list", default="", help="03/09: file tên ảnh — chỉ học residual pose cho các camera này (ảnh pose yếu)")
    p.add_argument("--pose_opt_reg", type=float, default=1e-2,
                   help="L2 prior keeping pose residuals near zero")
    p.add_argument("--seed", type=int, default=42)
    cfg = p.parse_args()
    if int(getattr(cfg, "batch_views", 1)) > 1:
        # noise SGLD của MCMC được bơm mỗi view-step ⇒ B lần mỗi bước Adam; chia B
        # để tổng noise mỗi bước Adam không đổi (2506.12727 §3.3).
        cfg.noise_lr = float(cfg.noise_lr) / int(cfg.batch_views)
        print(f"[batch_views] B={cfg.batch_views} lr_scale={cfg.lr_scale} "
              f"noise_lr->{cfg.noise_lr:g}", flush=True)
    return cfg


if __name__ == "__main__":
    train(parse_args())
