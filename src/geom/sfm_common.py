"""sfm_common.py — bộ đo dùng chung cho các biến thể SfM (đánh giá một reconstruction mới so pose BTC):
  holdout_eval  — Sampson KHỬ MÉO trên cặp giữ lại (<ba_dir>/holdout_sampson.json), theo bin track chung BTC (xem check_geometry.py)
  radial_profile — Sampson theo bán kính ảnh, cặp ≥ min_shared track chung (như y34b)
  pose_diff     — xoay/dịch tâm so BTC, tỉ số pivot, xoay tương đối n↔n+3 đổi bao nhiêu (như y48)
  align_to_btc  — Sim3 về khung BTC theo tâm camera
"""
import os, sys, json, glob, math, numpy as np, pycolmap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa
from sfm_audit import read_images  # noqa
from dataset import frame_index  # noqa

def skew(t): return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
def w2c_of(im):
    M = np.eye(4); M[:3, :4] = (im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world).matrix(); return M
def rec_w2c(rec): return {im.name: w2c_of(im) for im in rec.images.values() if im.has_pose}
def btc_pts(scene_dir):
    ims = read_images(os.path.join(scene_dir, "train", "sparse", "0", "images.bin")); return {im["name"]: set(int(i) for i in im["pid"][im["pid"] >= 0]) for im in ims.values()}
def load_matches(match_dir, keys=None):
    ks = set(keys) if keys is not None else None; M = {}
    for p in sorted(glob.glob(os.path.join(match_dir, "matches*.npz"))):
        z = np.load(p); M.update({k: z[k] for k in z.files if ks is None or k in ks})
    return M
class KP:
    def __init__(self, match_dir): self.d = match_dir; self.c = {}
    def __call__(self, n):
        if n not in self.c: self.c[n] = np.load(os.path.join(self.d, "kp", n + ".npy")).astype(np.float64)
        return self.c[n]
def sampson_px(xa, xb, Wa, Wb, f):
    """xa, xb: toạ độ CHUẨN HOÁ (đã khử méo); trả Sampson quy ra px (×f)."""
    Rab = Wb[:3, :3] @ Wa[:3, :3].T; tab = Wb[:3, 3] - Rab @ Wa[:3, 3]; E = skew(tab / (np.linalg.norm(tab) + 1e-12)) @ Rab
    ha = np.c_[xa, np.ones(len(xa))]; hb = np.c_[xb, np.ones(len(xb))]; l_b = ha @ E.T; l_a = hb @ E
    return f * np.abs((hb * l_b).sum(1)) / np.sqrt(l_b[:, 0] ** 2 + l_b[:, 1] ** 2 + l_a[:, 0] ** 2 + l_a[:, 1] ** 2 + 1e-12)
class Norm:
    """keypoint px → chuẩn hoá qua camera (khử méo), cache theo tên ảnh."""
    def __init__(self, cam, kp): self.cam = cam; self.kp = kp; self.c = {}
    def __call__(self, n):
        if n not in self.c: self.c[n] = np.asarray(self.cam.cam_from_img(self.kp(n)), dtype=np.float64)
        return self.c[n]
BINS = ((0, 1), (1, 20), (20, 100), (100, 10 ** 9))
def holdout_eval(rec, hold, M, kp, pts, tag, quiet=False):
    cam = next(iter(rec.cameras.values())); f = float(cam.params[0]); w2c = rec_w2c(rec); norm = Norm(cam, kp); vals = []; fr3 = []; fr1 = []; sh = []
    for k in hold:
        na, nb = k.split("|")
        if na not in w2c or nb not in w2c or k not in M: continue
        m = M[k]; s = sampson_px(norm(na)[m[:, 0]], norm(nb)[m[:, 1]], w2c[na], w2c[nb], f)
        vals.append(float(np.median(s))); fr3.append(float((s < 3).mean())); fr1.append(float((s < 1).mean())); sh.append(len(pts.get(na, set()) & pts.get(nb, set())))
    vals, fr3, fr1, sh = map(np.array, (vals, fr3, fr1, sh)); out = dict(n=int(len(vals)), med=float(np.median(vals)), p75=float(np.percentile(vals, 75)), f3=float(fr3.mean()), f1=float(fr1.mean()), bins={})
    if not quiet: print(f"[hold] {tag}: camera {cam.model.name} | n={len(vals)} | med-của-med {out['med']:.2f} p75 {out['p75']:.2f} | %match<3px {100 * out['f3']:.0f} | %match<1px {100 * out['f1']:.0f}", flush=True)
    for lo, hi in BINS:
        mm = (sh >= lo) & (sh < hi)
        if mm.sum():
            out["bins"][f"{lo}-{hi}"] = dict(n=int(mm.sum()), med=float(np.median(vals[mm])), p75=float(np.percentile(vals[mm], 75)), f3=float(fr3[mm].mean()), f1=float(fr1[mm].mean()))
            if not quiet: print(f"      track chung [{lo},{hi}): n={int(mm.sum()):4d} | Sampson med {np.median(vals[mm]):.2f} p75 {np.percentile(vals[mm], 75):.2f} | %<3px {100 * fr3[mm].mean():.0f} | %<1px {100 * fr1[mm].mean():.0f}", flush=True)
    return out
def radial_profile(rec, M, kp, pts, audit, tag, min_shared=100, max_dfr=0, quiet=False):
    cam = next(iter(rec.cameras.values())); f = float(cam.params[0]); W, H = cam.width, cam.height; cx, cy = W / 2, H / 2; rmax = np.hypot(cx, cy); w2c = rec_w2c(rec); norm = Norm(cam, kp)
    acc = {"cùng hướng": [], "xoay 180°": []}
    for k, m in M.items():
        na, nb = k.split("|")
        if na not in w2c or nb not in w2c or len(pts.get(na, ())) == 0 or len(pts[na] & pts.get(nb, set())) < min_shared: continue
        if max_dfr and abs(frame_index(na) - frame_index(nb)) > max_dfr: continue
        dh = abs((audit[na]["heading"] - audit[nb]["heading"] + 180) % 360 - 180); grp = "xoay 180°" if dh > 90 else "cùng hướng"
        pa, pb = kp(na), kp(nb); s = sampson_px(norm(na)[m[:, 0]], norm(nb)[m[:, 1]], w2c[na], w2c[nb], f)
        r = 0.5 * (np.hypot(pa[m[:, 0], 0] - cx, pa[m[:, 0], 1] - cy) + np.hypot(pb[m[:, 1], 0] - cx, pb[m[:, 1], 1] - cy)) / rmax; acc[grp].append(np.stack([r, s], 1))
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]; out = {}
    for grp, lst in acc.items():
        if not lst: continue
        A = np.concatenate(lst, 0); row = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = (A[:, 0] >= lo) & (A[:, 0] < hi); row.append(float(np.median(A[mm, 1])) if mm.sum() else float("nan"))
        out[grp] = dict(n_pairs=len(lst), prof=row)
        if not quiet: print(f"[radial] {tag} {grp:10s} ({len(lst)} cặp): " + " | ".join(f"{v:.2f}" for v in row) + f"   (tâm→góc ×{row[-1] / max(row[0], 1e-6):.1f})", flush=True)
    return out
def pose_diff(rec, rec0, tag, quiet=False):
    cam = next(iter(rec.cameras.values())); f = float(cam.params[0]); new = rec_w2c(rec); old = {im.name: w2c_of(im) for im in rec0.images.values() if im.name in new}
    rot = []; cen = []; piv = []; rel = []; byfr = {frame_index(n): n for n in new}
    for n in new:
        Wo, Wn = old[n], new[n]; Co, Cn = -Wo[:3, :3].T @ Wo[:3, 3], -Wn[:3, :3].T @ Wn[:3, 3]; dR = math.degrees(math.acos(np.clip((np.trace(Wo[:3, :3].T @ Wn[:3, :3]) - 1) / 2, -1, 1))); rot.append(dR); cen.append(float(np.linalg.norm(Cn - Co))); piv.append(cen[-1] / max(1.8 * math.tan(math.radians(dR)), 1e-9))
        m = byfr.get(frame_index(n) + 3)
        if m: Ro = old[m][:3, :3] @ Wo[:3, :3].T; Rn = new[m][:3, :3] @ Wn[:3, :3].T; rel.append(math.degrees(math.acos(np.clip((np.trace(Ro.T @ Rn) - 1) / 2, -1, 1))))
    out = dict(rot_med=float(np.median(rot)), rot_p90=float(np.percentile(rot, 90)), cen_med=float(np.median(cen)), piv_med=float(np.median(piv)), rel3_med=float(np.median(rel)) if rel else float("nan"))
    if not quiet: print(f"[pose] {tag}: so BTC xoay med {out['rot_med']:.3f}° (≈{out['rot_med'] / 180 * math.pi * f:.0f} px) p90 {out['rot_p90']:.3f}° | dịch tâm med {out['cen_med']:.4f} | pivot med {out['piv_med']:.2f} | xoay tương đối n↔n+3 đổi med {out['rel3_med']:.3f}° (≈{out['rel3_med'] / 180 * math.pi * f:.0f} px)", flush=True)
    return out
def align_to_btc(rec, rec0):
    sim = pycolmap.align_reconstructions_via_proj_centers(rec, rec0, 100.0); rec.transform(sim); return float(sim.scale)
def rec_stats(rec, tag):
    tl = [p.track.length() for p in rec.points3D.values()]; er = [p.error for p in rec.points3D.values()]; cam = next(iter(rec.cameras.values()))
    print(f"[rec] {tag}: {rec.num_reg_images()} ảnh | {rec.num_points3D()} điểm | track med {np.median(tl):.0f} mean {np.mean(tl):.1f} | reproj mean {np.mean(er):.2f} px | camera {cam.model.name} {[round(float(p), 6) for p in cam.params]}", flush=True)
