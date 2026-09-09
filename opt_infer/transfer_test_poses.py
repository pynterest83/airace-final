"""y44b_transfer3.py — 05/09: CHUYỂN POSE TEST QUA CẢNH (hợp lệ) bản 2: mỗi điểm 3D BTC cũ có các quan sát 2D (ảnh train, pixel) → undistort qua camera MỚI
→ tam giác hoá tuyến tính (DLT) bằng pose MỚI của các ảnh đó → X_new (lọc tái chiếu ≤ thr). Với view test: điểm cũ trong frustum theo pose cũ (hình chiếu
theo pose cũ, pinhole cũ = pixel méo → undistort) ↔ X_new → PnP-RANSAC + LM → pose test mới. Kiểm leave-one-out trên train (pose cũ → pose mới thật) và so oracle.
  python y44b_transfer3.py --orig_scene data/phase2/phase2_f1 --new_scene data/phase2/phase2_f1_reba4 --out_scene data/phase2/phase2_f1_reba4_tr3 [--oracle_csv S_oracle/test/test_poses.csv]"""
import argparse, os, sys, csv, math, numpy as np, cv2, pycolmap
_H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _H); sys.path.insert(0, os.path.join(_H, "..", "..", "..", "pipeline"))
from dataset import qvec2rotmat  # noqa
from y30_transfer import rotmat2qvec  # noqa
def w2c_of(im): M = np.eye(4); M[:3, :4] = (im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world).matrix(); return M
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--orig_scene", required=True); ap.add_argument("--new_scene", required=True); ap.add_argument("--out_scene", required=True)
    ap.add_argument("--min_inl", type=int, default=30, help="09/09: ngưỡng inlier PnP (kp); dưới → thất bại"); ap.add_argument("--fallback", default="none", choices=["none", "frustum", "sim3"], help="09/09: khi kp thất bại: frustum = PnP trên mọi điểm cũ trong frustum theo pose cũ (bản tr4); sim3 = Sim3 cục bộ từ 12 camera train gần nhất (old→new); ghi danh sách view fallback ra test/transfer_fallback.json")
    ap.add_argument("--oracle_csv", default=""); ap.add_argument("--reproj_thr", type=float, default=3.0); ap.add_argument("--max_pts", type=int, default=6000); ap.add_argument("--pnp_thr", type=float, default=4.0); ap.add_argument("--pts", default="frustum", help="frustum | nb (điểm do ảnh train kề Δframe quan sát) | own (chẩn đoán: track của chính ảnh)"); ap.add_argument("--nb_offsets", default="2,3"); ap.add_argument("--pre_save", default="", help="A: lưu X1/kp_obs/K/dist/pose ra .npy rồi tiếp tục"); ap.add_argument("--pre_load", default="", help="A: nạp từ .npy, BỎ QUA tam giác hoá (chỉ còn PnP)"); a = ap.parse_args()
    from dataset import frame_index
    if a.pre_load:
        _pre = np.load(a.pre_load, allow_pickle=True).item()
        K, dist, W, H, X1 = _pre["K"], _pre["dist"], int(_pre["W"]), int(_pre["H"]), _pre["X1"]
        kp_obs = {n: [(xy, int(r)) for xy, r in zip(v[0], v[1])] for n, v in _pre["kp_obs"].items()}
        old_w2c, new_w2c = _pre["old_w2c"], _pre["new_w2c"]; f = K[0, 0]; X0 = None
        def und(p): return cv2.undistortPoints(np.asarray(p, dtype=np.float64).reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
        def cand_rows(name): return None
        print(f"[A] nạp precompute: {len(X1)} điểm 3D, {len(kp_obs)} ảnh có keypoint", flush=True)
    else:
        rec0 = pycolmap.Reconstruction(os.path.join(a.orig_scene, "train", "sparse", "0")); rec1 = pycolmap.Reconstruction(os.path.join(a.new_scene, "train", "sparse", "0"))
        cam1 = next(iter(rec1.cameras.values())); K = cam1.calibration_matrix(); dist = np.asarray(cam1.params[4:8], dtype=np.float64) if cam1.model.name == "OPENCV" else np.zeros(4); W, H = cam1.width, cam1.height; Kinv = np.linalg.inv(K)
        tr_names = set(os.listdir(os.path.join(a.orig_scene, "train", "images"))); new_w2c = {im.name: w2c_of(im) for im in rec1.images.values() if im.has_pose}; old_w2c = {im.name: w2c_of(im) for im in rec0.images.values()}
        # quan sát 2D của từng điểm cũ trên ảnh train: pid -> list (name, xy)
        obs = {}
        for im in rec0.images.values():
            if im.name not in tr_names or im.name not in new_w2c: continue
            for p in im.points2D:
                if p.has_point3D(): obs.setdefault(p.point3D_id, []).append((im.name, np.asarray(p.xy, dtype=np.float64)))
        def und(p): return cv2.undistortPoints(np.asarray(p, dtype=np.float64).reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
        # DLT tam giác hoá với pose mới
        Xn = {}; nrej = 0
        for pid, ol in obs.items():
            if len(ol) < 2: continue
            A = []; Ps = []; uvs = und(np.array([xy for _, xy in ol]))
            for (nm, _), uv in zip(ol, uvs):
                P = K @ new_w2c[nm][:3, :4]; Ps.append(P); A.append(uv[0] * P[2] - P[0]); A.append(uv[1] * P[2] - P[1])
            _, _, Vt = np.linalg.svd(np.array(A)); X = Vt[-1]; X = X[:3] / X[3]
            errs = []
            for P, uv in zip(Ps, uvs):
                x = P @ np.r_[X, 1]
                if x[2] <= 0: errs.append(1e9); continue
                errs.append(np.linalg.norm(x[:2] / x[2] - uv))
            if np.median(errs) <= a.reproj_thr: Xn[pid] = X
            else: nrej += 1
        print(f"[y44b] tam giác hoá lại {len(Xn)} điểm cũ với pose mới (loại {nrej} tái chiếu > {a.reproj_thr} px; điểm cũ có ≥2 quan sát train: {sum(1 for v in obs.values() if len(v) >= 2)})", flush=True)
        pids = np.array(list(Xn)); X0 = np.array([rec0.points3D[p].xyz for p in pids]); X1 = np.array([Xn[p] for p in pids]); f = K[0, 0]; pid_row = {int(p): i for i, p in enumerate(pids)}
        seen_by = {}  # tên ảnh train -> set chỉ số hàng của điểm nó quan sát
        for im in rec0.images.values():
            if True: seen_by[im.name] = set(pid_row[p.point3D_id] for p in im.points2D if p.has_point3D() and p.point3D_id in pid_row)
        byfr = {frame_index(n): n for n in tr_names}; offs = [int(x) for x in a.nb_offsets.split(",")]
        def cand_rows(name):
            if a.pts == "own": return np.array(sorted(seen_by.get(name, set())), dtype=int)
            if a.pts == "nb":
                fr = frame_index(name); S = set()
                for d in offs:
                    for sgn in (-d, d):
                        m = byfr.get(fr + sgn)
                        if m and m != name: S |= seen_by.get(m, set())
                return np.array(sorted(S), dtype=int)
            return None
        kp_obs = {}
        for im in rec0.images.values(): kp_obs[im.name] = [(np.asarray(p.xy, dtype=np.float64), pid_row[p.point3D_id]) for p in im.points2D if p.has_point3D() and p.point3D_id in pid_row]
        def sim3_local(W_old, k=12):   # 09/09: Sim3 Umeyama old→new trên k camera train gần nhất (theo tâm cũ); xoay = R_new_nb·R_old_nbᵀ·R_old
            names = [n for n in old_w2c if n in new_w2c and n in tr_names]
            Co = np.array([-old_w2c[n][:3, :3].T @ old_w2c[n][:3, 3] for n in names]); Cn = np.array([-new_w2c[n][:3, :3].T @ new_w2c[n][:3, 3] for n in names])
            c_old = -W_old[:3, :3].T @ W_old[:3, 3]; nb = np.argsort(np.linalg.norm(Co - c_old, axis=1))[:k]
            A, B = Co[nb], Cn[nb]; ma, mb = A.mean(0), B.mean(0); U, D, Vt = np.linalg.svd((B - mb).T @ (A - ma) / k); S = np.eye(3)
            if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1
            R = U @ S @ Vt; sc = np.trace(np.diag(D) @ S) / ((A - ma) ** 2).sum(1).mean(); t = mb - sc * R @ ma
            c_new = sc * R @ c_old + t; n0 = names[nb[0]]; Rn = new_w2c[n0][:3, :3] @ old_w2c[n0][:3, :3].T @ W_old[:3, :3]
            Wn = np.eye(4); Wn[:3, :3] = Rn; Wn[:3, 3] = -Rn @ c_new; return Wn
        if a.pre_save:
            _ko = {n: (np.array([o[0] for o in v], dtype=np.float64).reshape(-1, 2), np.array([o[1] for o in v], dtype=np.int64)) for n, v in kp_obs.items()}
            np.save(a.pre_save, dict(K=K, dist=dist, W=W, H=H, X1=X1, kp_obs=_ko, old_w2c=old_w2c, new_w2c=new_w2c), allow_pickle=True)
            print(f"[A] đã lưu precompute → {a.pre_save}", flush=True)
    def transfer(W_old, name="", init=None, min_inl=None):
        min_inl = a.min_inl if min_inl is None else min_inl; W_init = W_old if init is None else init
        if a.pts == "kp" and name:
            ol = kp_obs.get(name, [])
            if len(ol) < max(6, min_inl): return None, 0
            p_und = und(np.array([o[0] for o in ol])); P_new = X1[[o[1] for o in ol]]
            okp, rvec, tvec, inl = cv2.solvePnPRansac(P_new, p_und, K, None, iterationsCount=3000, reprojectionError=a.pnp_thr, confidence=0.9999, flags=cv2.SOLVEPNP_ITERATIVE, rvec=cv2.Rodrigues(W_init[:3, :3])[0], tvec=W_init[:3, 3].reshape(3, 1), useExtrinsicGuess=True)
            if not okp or inl is None or len(inl) < min_inl: return None, 0
            rvec, tvec = cv2.solvePnPRefineLM(P_new[inl.ravel()], p_und[inl.ravel()], K, None, rvec, tvec); Wn = np.eye(4); Wn[:3, :3] = cv2.Rodrigues(rvec)[0]; Wn[:3, 3] = tvec.ravel(); return Wn, int(len(inl))
        rows_c = cand_rows(name) if name else None
        Xs = X0 if rows_c is None else X0[rows_c]
        Xc = (W_old[:3, :3] @ Xs.T).T + W_old[:3, 3]; z = Xc[:, 2]; uv = (K @ Xc.T).T; u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
        ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H); idx = np.where(ok)[0]
        if len(idx) < 30: return None, 0
        if rows_c is not None and len(rows_c) == 0: return None, 0
        if len(idx) > a.max_pts: idx = np.random.default_rng(0).choice(idx, a.max_pts, replace=False)   # 09/09: lấy mẫu TRƯỚC khi ánh xạ idx_glob (bug cũ: lệch số điểm 2D/3D)
        idx_glob = rows_c[idx] if rows_c is not None else idx
        p_und = und(np.stack([u[idx], v[idx]], 1)); P_new = X1[idx_glob]
        okp, rvec, tvec, inl = cv2.solvePnPRansac(P_new, p_und, K, None, iterationsCount=3000, reprojectionError=a.pnp_thr, confidence=0.9999, flags=cv2.SOLVEPNP_ITERATIVE, rvec=cv2.Rodrigues(W_old[:3, :3])[0], tvec=W_old[:3, 3].reshape(3, 1), useExtrinsicGuess=True)
        if not okp or inl is None or len(inl) < 30: return None, 0
        rvec, tvec = cv2.solvePnPRefineLM(P_new[inl.ravel()], p_und[inl.ravel()], K, None, rvec, tvec); Wn = np.eye(4); Wn[:3, :3] = cv2.Rodrigues(rvec)[0]; Wn[:3, 3] = tvec.ravel(); return Wn, int(len(inl))
    def px_err(Wa, Wb): return math.radians(math.degrees(math.acos(np.clip((np.trace(Wa[:3, :3].T @ Wb[:3, :3]) - 1) / 2, -1, 1)))) * f
    errs = []
    for n in list(new_w2c)[::6]:
        if n not in old_w2c: continue
        Wn, ni = transfer(old_w2c[n], n)
        if Wn is not None: errs.append(px_err(Wn, new_w2c[n]))
    print(f"[y44b pts={a.pts} nb={a.nb_offsets}] kiểm leave-one-out ảnh train ({len(errs)}): lệch xoay tương đương med {np.median(errs):.1f} px p75 {np.percentile(errs, 75):.1f} p90 {np.percentile(errs, 90):.1f}", flush=True)
    rows = list(csv.DictReader(open(os.path.join(a.orig_scene, "test", "test_poses.csv")))); out_rows = []; ninl = []; fb_list = []
    for r in rows:
        Wo = np.eye(4); Wo[:3, :3] = qvec2rotmat(np.array([float(r[k]) for k in ("qw", "qx", "qy", "qz")])); Wo[:3, 3] = [float(r[k]) for k in ("tx", "ty", "tz")]
        Wn, ni = transfer(Wo, r["image_name"]); r2 = dict(r); ninl.append(ni); how = "kp"
        if Wn is None and a.fallback != "none":   # 09/09: KHÔNG giữ pose cũ (hệ toạ độ cũ) — lỗi B của đồng đội: 0033/0317 giữ nguyên dòng CSV gốc → lệch ~289 px
            Wf = None
            if a.fallback == "frustum":
                _pts = a.pts; a.pts = "frustum"; Wf, nf = transfer(Wo, "", min_inl=30); a.pts = _pts; how = f"frustum({nf})"
            if Wf is None: Wf = sim3_local(Wo); how = "sim3"
            Wk, nk = transfer(Wo, r["image_name"], init=Wf, min_inl=max(8, a.min_inl // 2))   # thử lại kp với khởi tạo tốt, ngưỡng thấp
            if Wk is not None: Wn, ni = Wk, nk; how += f"+kp({nk})"
            else: Wn = Wf
            fb_list.append(dict(image=r["image_name"], how=how)); print(f"[y44b] FALLBACK {r['image_name'][-10:-4]}: {how}", flush=True)
        if Wn is not None:
            q = rotmat2qvec(Wn[:3, :3]); r2.update(qw=f"{q[0]:.17g}", qx=f"{q[1]:.17g}", qy=f"{q[2]:.17g}", qz=f"{q[3]:.17g}", tx=f"{Wn[0, 3]:.17g}", ty=f"{Wn[1, 3]:.17g}", tz=f"{Wn[2, 3]:.17g}")
        out_rows.append(r2)
    print(f"[y44b] test: inlier PnP med {int(np.median(ninl))} | view thất bại: {sum(1 for x in ninl if x == 0)}", flush=True)
    os.makedirs(os.path.join(a.out_scene, "train"), exist_ok=True); os.makedirs(os.path.join(a.out_scene, "test"), exist_ok=True)
    for sub in ("train/images", "train/sparse", "test/images"):
        dst = os.path.join(a.out_scene, sub)
        if not os.path.lexists(dst): os.symlink(os.path.abspath(os.path.join(a.new_scene, sub)), dst)
    with open(os.path.join(a.out_scene, "test", "test_poses.csv"), "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in out_rows]
    import json; _kept_old = [r["image_name"] for r, n in zip(rows, ninl) if n == 0 and a.fallback == "none"]
    json.dump(dict(fallback=fb_list, kept_old_frame=_kept_old), open(os.path.join(a.out_scene, "test", "transfer_fallback.json"), "w"), indent=1)
    if _kept_old: print(f"[y44b] !! CẢNH BÁO: {len(_kept_old)} view GIỮ POSE HỆ CŨ (PnP thất bại, --fallback none): {[n[-10:-4] for n in _kept_old]}", flush=True)
    if a.oracle_csv and os.path.exists(a.oracle_csv):
        O = {r["image_name"]: r for r in csv.DictReader(open(a.oracle_csv))}; oe = []
        for r2 in out_rows:
            ro = O.get(r2["image_name"])
            if not ro: continue
            Wa = np.eye(4); Wa[:3, :3] = qvec2rotmat(np.array([float(r2[k]) for k in ("qw", "qx", "qy", "qz")])); Wb = np.eye(4); Wb[:3, :3] = qvec2rotmat(np.array([float(ro[k]) for k in ("qw", "qx", "qy", "qz")])); oe.append(px_err(Wa, Wb))
        print(f"[y44b] so oracle (chẩn đoán, {len(oe)} view): lệch med {np.median(oe):.1f} px p75 {np.percentile(oe, 75):.1f} p90 {np.percentile(oe, 90):.1f} | %≤5px {100 * (np.array(oe) <= 5).mean():.0f}", flush=True)
    print(f"[y44b] scene -> {a.out_scene}"); print("Y44B_DONE", flush=True)
if __name__ == "__main__": main()
