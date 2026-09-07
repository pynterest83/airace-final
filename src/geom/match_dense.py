"""match_dense.py — KHỚP DÀY mọi cặp ảnh TRAIN chồng phủ (SuperPoint @res + LightGlue), xoay 180° khi heading ngược, kiểm F-RANSAC.
Chỉ ảnh train (không đụng ảnh test). Ghi <out>/kp/<name>.npy (keypoint full-res) + <out>/matches.npz (cặp → chỉ số match inlier).
  PYTHONPATH=~/lg_pkgs python match_dense.py --scene_dir $VT_SCENE --audit $VT_RUNS/sfm_audit.json --out $VT_RUNS/geom/match [--res 3072] [--kp 8192] [--min_ov 0.15]
"""
import argparse, os, sys, json, math, time, numpy as np, torch, cv2
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa
from dataset import SceneData, frame_index  # noqa
from refiner_data import fit_ground_plane  # noqa
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene_dir", required=True); ap.add_argument("--audit", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=3072); ap.add_argument("--kp", type=int, default=8192); ap.add_argument("--min_ov", type=float, default=0.15); ap.add_argument("--min_inl", type=int, default=30); ap.add_argument("--shard", default="0/1"); ap.add_argument("--cache", type=int, default=0, help="07/09: số bản đặc trưng giữ trên GPU (0 = tất cả)"); ap.add_argument("--ransac", default="fm", choices=["fm", "usac"]); ap.add_argument("--save_every", type=int, default=2000)
    a = ap.parse_args(); dev = "cuda"; os.makedirs(os.path.join(a.out, "kp"), exist_ok=True)
    from lightglue import LightGlue, SuperPoint
    ext = SuperPoint(max_num_keypoints=a.kp).eval().to(dev); mt = LightGlue(features="superpoint", flash=bool(int(os.environ.get("LG_FLASH", "0")))).eval().to(dev)   # 07/09: flash-SDPA tốn 0,4–0,8 s MỖI cỡ tensor mới (torch 2.13) → mặc định tắt: 0,09 s/cặp thay 0,6
    sc = SceneData(a.scene_dir, load_images=False, distorted=False, holdout_every=0); K = sc.K.astype(np.float64); f = K[0, 0]; W, H = sc.width, sc.height; s = a.res / max(W, H)
    Ws, Hs = int(W * s), int(H * s); au = json.load(open(a.audit))["per_image"]; n_pl, d_pl, _ = fit_ground_plane(np.asarray(sc.points.xyz, dtype=np.float64)); n_pl = np.asarray(n_pl, dtype=np.float64)
    img_dir = os.path.join(a.scene_dir, "train", "images"); info = {}
    for m in sc.train_metas:
        w2c = m.w2c().astype(np.float64); R, t = w2c[:3, :3], w2c[:3, 3]; C = -R.T @ t; v = R.T @ np.array([0, 0, 1.0]); den = float(n_pl @ v)
        sd = (d_pl - float(n_pl @ C)) / den if abs(den) > 1e-6 else 1e9; alt = abs(float(n_pl @ C) - d_pl); p = au[m.name]
        info[m.name] = dict(X=C + v * sd, foot=alt * W / f * (1.3 if p["pitch"] > 15 else 1.0), heading=float(p["heading"]), pitch=float(p["pitch"]), fr=frame_index(m.name))
    names = sorted(info, key=lambda k: info[k]["fr"]); pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = info[names[i]], info[names[j]]; dist = float(np.linalg.norm(A["X"] - B["X"])); foot = 0.5 * (A["foot"] + B["foot"])
            if 1 - dist / foot >= a.min_ov: pairs.append((names[i], names[j]))
    print(f"[match] {len(names)} ảnh, {len(pairs)} cặp chồng phủ ≥ {a.min_ov}", flush=True)
    # 1) keypoint mọi ảnh (2 bản: gốc + xoay 180° — descriptor SuperPoint không bất biến xoay)
    feats = {}; TM = {"feat": 0.0, "match": 0.0, "ransac": 0.0}
    def load_feat(nm, rot):
        key = (nm, rot)
        if key in feats: return feats[key]
        g = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(img_dir, nm)), (Ws, Hs), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if rot: g = cv2.rotate(g, cv2.ROTATE_180)
        _t = time.time()
        with torch.no_grad(): fe = ext.extract(torch.from_numpy(g).float().div(255)[None, None].to(dev))
        fe = {k: (v.half() if k == "descriptors" else v) for k, v in fe.items()}   # 07/09: giữ TẤT CẢ đặc trưng trên GPU (808 bản × ~4 MB fp16) — cache 60 bản cũ làm nạp lại ảnh liên tục (0,25 s/lần)
        feats[key] = fe; TM["feat"] += time.time() - _t
        if a.cache and len(feats) > a.cache: feats.pop(next(iter(feats)))
        return fe
    kp_full = {}  # tên → (kp gốc full-res [N,2], kp xoay full-res [M,2]) — mỗi ảnh có 2 tập keypoint độc lập; ghép thành 1 danh sách: [gốc; xoay]
    def kps(nm):
        if nm in kp_full: return kp_full[nm]
        k0 = load_feat(nm, False)["keypoints"][0].cpu().numpy() / s; k1 = load_feat(nm, True)["keypoints"][0].cpu().numpy()
        k1 = np.stack([(Ws - 1) - k1[:, 0], (Hs - 1) - k1[:, 1]], 1) / s; kp_full[nm] = (k0, k1); np.save(os.path.join(a.out, "kp", nm + ".npy"), np.concatenate([k0, k1], 0).astype(np.float32)); return kp_full[nm]
    t0 = time.time(); out = {}; n_ok = 0; stats = []; RS = {"fm": cv2.FM_RANSAC, "usac": cv2.USAC_DEFAULT}[a.ransac]
    # sắp cặp theo ảnh A để tận dụng cache
    pairs.sort(key=lambda p: (info[p[0]]["fr"], info[p[1]]["fr"]))
    si, sn = [int(x) for x in a.shard.split("/")]; pairs = pairs[si::sn]; tag = f"_s{si}of{sn}" if sn > 1 else ""
    print(f"[match] mảnh {si}/{sn}: {len(pairs)} cặp", flush=True)
    def dump():
        np.savez_compressed(os.path.join(a.out, f"matches{tag}.npz"), **out)
        json.dump(dict(res=a.res, kp=a.kp, n_pairs=len(pairs), n_ok=n_ok, inl_med=float(np.median(stats)) if stats else 0, names=names, shard=a.shard), open(os.path.join(a.out, f"match_meta{tag}.json"), "w"))
    for idx, (na, nb) in enumerate(pairs):
        if idx and idx % a.save_every == 0: dump()
        dh = abs((info[na]["heading"] - info[nb]["heading"] + 180) % 360 - 180); rot = dh > 90
        fa, fb = load_feat(na, False), load_feat(nb, rot)
        _t = time.time()
        with torch.no_grad(): mm = mt({"image0": {k: (v.float() if k == "descriptors" else v) for k, v in fa.items()}, "image1": {k: (v.float() if k == "descriptors" else v) for k, v in fb.items()}})["matches"][0].cpu().numpy()
        TM["match"] += time.time() - _t
        k0a, k1a = kps(na); k0b, k1b = kps(nb)
        if len(mm) < a.min_inl: continue
        pa = k0a[mm[:, 0]]; pb = (k1b if rot else k0b)[mm[:, 1]]; ib = mm[:, 1] + (len(k0b) if rot else 0)
        _t = time.time(); Fm, inl = cv2.findFundamentalMat(pa, pb, RS, 3.0, 0.9999); TM["ransac"] += time.time() - _t
        if Fm is None or inl is None or inl.sum() < a.min_inl: continue
        inl = inl.ravel().astype(bool); out[f"{na}|{nb}"] = np.stack([mm[inl, 0], ib[inl]], 1).astype(np.int32); n_ok += 1; stats.append(int(inl.sum()))
        if idx % 500 == 0: print(f"  [{idx}/{len(pairs)}] {na[-10:-6]}-{nb[-10:-6]}{'R' if rot else ''} match {len(mm)} inl {int(inl.sum())} | {time.time() - t0:.0f}s | cặp OK {n_ok}", flush=True)
    if si == 0:
        for nm in names: kps(nm)  # đảm bảo mọi ảnh có keypoint (mảnh 0)
    dump()
    print(f"[match] thời gian: đặc trưng {TM['feat']:.0f}s ({len(feats)} bản) | match {TM['match']:.0f}s | ransac {TM['ransac']:.0f}s | tổng {time.time() - t0:.0f}s", flush=True)
    print(f"Y30_MATCH_SHARD_DONE {a.shard} cặp OK {n_ok}/{len(pairs)} | inlier med {np.median(stats) if stats else 0:.0f} | {time.time() - t0:.0f}s", flush=True)
if __name__ == "__main__": main()
