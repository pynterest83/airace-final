"""check_geometry.py — CỔNG hình học: Sampson KHỬ MÉO trên cặp GIỮ LẠI (<ba_dir>/holdout_sampson.json),
theo bin track chung BTC, so pose BTC (pinhole) vs SfM mới. Cặp giữ lại KHÔNG hề vào tối ưu.
  python check_geometry.py --scene_dir $VT_SCENE --match_dir $VT_RUNS/geom/match --ba_dir $VT_RUNS/geom/db"""
import argparse, os, sys, json, glob, numpy as np, pycolmap
_H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _H)
from sfm_audit import read_images  # noqa
def skew(t): return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
ap = argparse.ArgumentParser(); ap.add_argument("--scene_dir", required=True); ap.add_argument("--match_dir", required=True); ap.add_argument("--ba_dir", required=True); ap.add_argument("--sparse", default="", help="sparse/0 khác để kiểm (mặc định ba_dir/sparse/0)"); a = ap.parse_args()
hold = json.load(open(os.path.join(a.ba_dir, "holdout_sampson.json")))["hold_pairs"]; hs = set(hold); M = {}
for p in sorted(glob.glob(os.path.join(a.match_dir, "matches*.npz"))): z = np.load(p); M.update({k: z[k] for k in z.files if k in hs})
ims = read_images(os.path.join(a.scene_dir, "train", "sparse", "0", "images.bin")); pts = {im["name"]: set(int(i) for i in im["pid"][im["pid"] >= 0]) for im in ims.values()}
kp = {}
def K_(n):
    if n not in kp: kp[n] = np.load(os.path.join(a.match_dir, "kp", n + ".npy")).astype(np.float64)
    return kp[n]
def run(rec, tag):
    cam = next(iter(rec.cameras.values())); f = float(cam.params[0]); w2c = {}
    for im in rec.images.values():
        if im.has_pose: Mx = np.eye(4); Mx[:3, :4] = (im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world).matrix(); w2c[im.name] = Mx
    vals = []; fr3 = []; sh = []; cache = {}
    def norm(n):
        if n not in cache: cache[n] = np.asarray(cam.cam_from_img(K_(n)), dtype=np.float64)
        return cache[n]
    for k in hold:
        na, nb = k.split("|")
        if na not in w2c or nb not in w2c or k not in M: continue
        m = M[k]; xa, xb = norm(na)[m[:, 0]], norm(nb)[m[:, 1]]; Wa, Wb = w2c[na], w2c[nb]; Rab = Wb[:3, :3] @ Wa[:3, :3].T; tab = Wb[:3, 3] - Rab @ Wa[:3, 3]; E = skew(tab / (np.linalg.norm(tab) + 1e-12)) @ Rab
        ha = np.c_[xa, np.ones(len(xa))]; hb = np.c_[xb, np.ones(len(xb))]; l_b = ha @ E.T; l_a = hb @ E; s = f * np.abs((hb * l_b).sum(1)) / np.sqrt(l_b[:, 0] ** 2 + l_b[:, 1] ** 2 + l_a[:, 0] ** 2 + l_a[:, 1] ** 2 + 1e-12)
        vals.append(float(np.median(s))); fr3.append(float((s < 3).mean())); sh.append(len(pts.get(na, set()) & pts.get(nb, set())))
    vals, fr3, sh = map(np.array, (vals, fr3, sh)); print(f"[check_geom] {tag}: camera {cam.model.name} | n={len(vals)} | med-của-med {np.median(vals):.2f} p75 {np.percentile(vals, 75):.2f} | %match<3px {100 * fr3.mean():.0f} | %cặp ≥50%<3px {100 * (fr3 >= 0.5).mean():.0f}")
    for lo, hi in ((0, 1), (1, 20), (20, 100), (100, 10 ** 9)):
        mm = (sh >= lo) & (sh < hi)
        if mm.sum(): print(f"      track chung [{lo},{hi}): n={int(mm.sum()):4d} | Sampson med {np.median(vals[mm]):.2f} p75 {np.percentile(vals[mm], 75):.2f} | %match<3px {100 * fr3[mm].mean():.0f}")
run(pycolmap.Reconstruction(os.path.join(a.scene_dir, "train", "sparse", "0")), "pose BTC pinhole")
run(pycolmap.Reconstruction(a.sparse or os.path.join(a.ba_dir, "sparse", "0")), f"BA {a.sparse or os.path.basename(a.ba_dir)}")
print("Y38_DONE")
