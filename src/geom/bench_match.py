"""bench_match.py — 07/09: đo thời gian từng khâu của match dày (match_dense) trên N cặp: extract/ảnh, LightGlue/cặp, RANSAC/cặp — và biến thể
(--kp, --half fp16 matcher, --ransac fm|usac|magsac, --compile) để ước lượng chi phí trên 1 card ngày thi.
  python bench_match.py --scene_dir data/phase2/phase2_f1 --audit runs/y01/sfm_audit.json --n 300 [--kp 8192 --res 3072 --half 0 --ransac fm]"""
import argparse, os, sys, json, time, numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa
from dataset import SceneData, frame_index  # noqa
from refiner_data import fit_ground_plane  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--scene_dir", required=True); ap.add_argument("--audit", required=True); ap.add_argument("--n", type=int, default=300)
ap.add_argument("--res", type=int, default=3072); ap.add_argument("--kp", type=int, default=8192); ap.add_argument("--min_ov", type=float, default=0.15); ap.add_argument("--half", type=int, default=0)
ap.add_argument("--ransac", default="fm", choices=["fm", "usac", "magsac", "none"]); ap.add_argument("--compile", type=int, default=0); ap.add_argument("--flash", type=int, default=int(os.environ.get("LG_FLASH", "0")),
    help="mặc định theo LG_FLASH (=0). flash-SDPA của torch 2.13 tốn 0,4-0,8 s MỖI cỡ tensor mới → bench với flash=1 cho số bi quan ~30 lần"); ap.add_argument("--stride", type=int, default=50); a = ap.parse_args(); dev = "cuda"
from lightglue import LightGlue, SuperPoint
ext = SuperPoint(max_num_keypoints=a.kp).eval().to(dev); mt = LightGlue(features="superpoint", flash=bool(a.flash)).eval().to(dev)
if a.compile: mt.compile(mode="reduce-overhead")
sc = SceneData(a.scene_dir, load_images=False, distorted=False, holdout_every=0); K = sc.K.astype(np.float64); f = K[0, 0]; W, H = sc.width, sc.height; s = a.res / max(W, H); Ws, Hs = int(W * s), int(H * s)
au = json.load(open(a.audit))["per_image"]; n_pl, d_pl, _ = fit_ground_plane(np.asarray(sc.points.xyz, dtype=np.float64)); n_pl = np.asarray(n_pl, dtype=np.float64); img_dir = os.path.join(a.scene_dir, "train", "images"); info = {}
for m in sc.train_metas:
    w2c = m.w2c().astype(np.float64); R, t = w2c[:3, :3], w2c[:3, 3]; C = -R.T @ t; v = R.T @ np.array([0, 0, 1.0]); den = float(n_pl @ v)
    sd = (d_pl - float(n_pl @ C)) / den if abs(den) > 1e-6 else 1e9; alt = abs(float(n_pl @ C) - d_pl); p = au[m.name]
    info[m.name] = dict(X=C + v * sd, foot=alt * W / f * (1.3 if p["pitch"] > 15 else 1.0), heading=float(p["heading"]), fr=frame_index(m.name))
names = sorted(info, key=lambda k: info[k]["fr"]); pairs = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        A, B = info[names[i]], info[names[j]]; dist = float(np.linalg.norm(A["X"] - B["X"])); foot = 0.5 * (A["foot"] + B["foot"])
        if 1 - dist / foot >= a.min_ov: pairs.append((names[i], names[j]))
pairs.sort(key=lambda p: (info[p[0]]["fr"], info[p[1]]["fr"])); pairs = pairs[:: a.stride][: a.n]; print(f"[p5] {len(names)} ảnh, tổng {len(pairs) * a.stride if a.stride > 1 else len(pairs)} cặp; đo {len(pairs)} cặp (stride {a.stride}) kp {a.kp} res {a.res} half {a.half} ransac {a.ransac}", flush=True)
feats = {}; T = dict(io=0.0, ext=0.0, match=0.0, ransac=0.0); n_ext = 0
def load_feat(nm, rot):
    global n_ext
    key = (nm, rot)
    if key in feats: return feats[key]
    t0 = time.time(); g = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(img_dir, nm)), (Ws, Hs), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    if rot: g = cv2.rotate(g, cv2.ROTATE_180)
    t1 = time.time(); T["io"] += t1 - t0
    with torch.no_grad(): fe = ext.extract(torch.from_numpy(g).float().div(255)[None, None].to(dev))
    torch.cuda.synchronize(); T["ext"] += time.time() - t1; n_ext += 1; feats[key] = fe
    if len(feats) > 60: feats.pop(next(iter(feats)))
    return fe
inl_all = []; t_all = time.time()
for idx, (na, nb) in enumerate(pairs):
    dh = abs((info[na]["heading"] - info[nb]["heading"] + 180) % 360 - 180); rot = dh > 90
    fa, fb = load_feat(na, False), load_feat(nb, rot); t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=bool(a.half)): mm = mt({"image0": fa, "image1": fb})["matches"][0].cpu().numpy()
    torch.cuda.synchronize(); t1 = time.time(); T["match"] += t1 - t0
    if len(mm) < 30 or a.ransac == "none": inl_all.append(len(mm)); continue
    pa = fa["keypoints"][0].cpu().numpy()[mm[:, 0]]; pb = fb["keypoints"][0].cpu().numpy()[mm[:, 1]]
    meth = {"fm": cv2.FM_RANSAC, "usac": cv2.USAC_DEFAULT, "magsac": cv2.USAC_MAGSAC}[a.ransac]
    Fm, inl = cv2.findFundamentalMat(pa, pb, meth, 3.0 * s if False else 3.0, 0.9999); T["ransac"] += time.time() - t1
    inl_all.append(int(inl.sum()) if inl is not None else 0)
    if idx % 50 == 0: print(f"  [{idx}] match {len(mm)} inl {inl_all[-1]} | {time.time() - t_all:.0f}s", flush=True)
n = len(pairs); tot = time.time() - t_all
print(f"P5 kp={a.kp} res={a.res} half={a.half} ransac={a.ransac} compile={a.compile} flash={a.flash}: {tot / n:.3f} s/cặp (match {T['match'] / n:.3f} ransac {T['ransac'] / n:.3f} | extract {T['ext'] / max(n_ext, 1):.3f} s/ảnh io {T['io'] / max(n_ext, 1):.3f} s/ảnh, {n_ext} bản) inlier med {np.median(inl_all):.0f} | ước 25 310 cặp + 808 bản extract = {(25310 * (T['match'] + T['ransac']) / n + 808 * (T['ext'] + T['io']) / max(n_ext, 1)) / 60:.0f} phút/1 card", flush=True)
