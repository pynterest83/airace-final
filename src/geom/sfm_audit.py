"""sfm_audit.py — AUDIT SfM BTC từ CHÍNH file sparse (không cần model, không cần GT).
images.bin chứa cả 506 ảnh (404 train + 102 test) với keypoint 2D + point3D_id → với MỖI ảnh đo được:
  - residual tái chiếu dưới pose BTC (median / p90 / %>3px), số quan sát 3D, track length
  - ĐỘ RÀNG BUỘC: số điểm 3D chia sẻ với ≥1/≥3 ảnh TRAIN khác (với ảnh test = mức neo vào cấu trúc train),
    bậc đồng-thị (số ảnh khác chia ≥30 điểm)
  - góc gimbal (pitch so pháp tuyến mặt đất), độ cao, heading, dải bay, láng giềng train (test)
  - trường residual GỘP theo ô ảnh 12×9 (chỉ train): mẫu hệ thống ⇒ K sai (tâm quang / tiêu cự / méo) — fit affine+radial.
  python sfm_audit.py --scene_dir $VT_SCENE --out $VT_RUNS/sfm_audit.json
"""
import argparse, os, sys, json, struct, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa
from dataset import frame_index  # noqa
from colmap_io import qvec2rotmat  # noqa


def read_cameras(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; cams = {}
        for _ in range(n):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24)); npar = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 6: 12}[mid]
            cams[cid] = (mid, int(w), int(h), np.array(struct.unpack("<" + "d" * npar, f.read(8 * npar))))
    return cams


def read_images(p):
    ims = {}
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]; q = np.array(struct.unpack("<dddd", f.read(32))); t = np.array(struct.unpack("<ddd", f.read(24))); cam = struct.unpack("<i", f.read(4))[0]
            s = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                s += c
            m = struct.unpack("<Q", f.read(8))[0]; arr = np.frombuffer(f.read(24 * m), dtype=[("x", "<f8"), ("y", "<f8"), ("p", "<i8")])
            ims[iid] = dict(id=iid, q=q, t=t, cam=cam, name=s.decode(), xy=np.stack([arr["x"], arr["y"]], 1).astype(np.float64), pid=arr["p"].astype(np.int64))
    return ims


def read_points(p):
    fmt = "<QdddBBBdQ"; sz = struct.calcsize(fmt)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; ids = np.empty(n, np.int64); xyz = np.empty((n, 3)); err = np.empty(n); tl = np.empty(n, np.int64); tracks = []
        for i in range(n):
            pid, x, y, z, r, g, b, e, L = struct.unpack(fmt, f.read(sz))
            ids[i] = pid; xyz[i] = (x, y, z); err[i] = e; tl[i] = L
            tracks.append(np.frombuffer(f.read(8 * L), dtype="<i4").reshape(L, 2)[:, 0].copy())  # image ids
    return ids, xyz, err, tl, tracks


def fit_plane(P, iters=400, thr_frac=0.02, seed=0):
    rng = np.random.default_rng(seed); P = P[np.isfinite(P).all(1)]
    if P.shape[0] > 200000: P = P[rng.choice(P.shape[0], 200000, replace=False)]
    ext = np.linalg.norm(P.max(0) - P.min(0)); thr = thr_frac * ext; best = (0, None, None)
    for _ in range(iters):
        i = rng.choice(P.shape[0], 3, replace=False); n = np.cross(P[i[1]] - P[i[0]], P[i[2]] - P[i[0]]); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n = n / nn; d = n @ P[i[0]]; k = int((np.abs(P @ n - d) < thr).sum())
        if k > best[0]: best = (k, n, d)
    k, n, d = best; inl = P[np.abs(P @ n - d) < thr]; c = inl.mean(0); u, sv, vt = np.linalg.svd(inl - c, full_matrices=False); n = vt[2]; d = n @ c
    return n, d, k / P.shape[0], thr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene_dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    sp = os.path.join(a.scene_dir, "train", "sparse", "0")
    cams = read_cameras(os.path.join(sp, "cameras.bin")); ims = read_images(os.path.join(sp, "images.bin")); ids, xyz, perr, tl, tracks = read_points(os.path.join(sp, "points3D.bin"))
    (mid, W, H, par) = cams[list(cams)[0]]; f, cx, cy = par[0], par[1], par[2]
    print(f"[cam] model {mid} {W}x{H} f={f:.3f} cx={cx:.2f} (W/2={W/2}) cy={cy:.2f} (H/2={H/2}) | {len(ims)} ảnh | {len(ids)} điểm 3D | track med {np.median(tl):.0f} mean {tl.mean():.1f} | err mean {perr.mean():.3f} px", flush=True)
    id2row = {int(p): i for i, p in enumerate(ids)}
    train_names = set(os.listdir(os.path.join(a.scene_dir, "train", "images"))); test_names = set(os.listdir(os.path.join(a.scene_dir, "test", "images")))
    is_train = {iid: (im["name"] in train_names) for iid, im in ims.items()}; is_test = {iid: (im["name"] in test_names) for iid, im in ims.items()}
    ntr = np.array([int(sum(is_train[int(i)] for i in tr)) for tr in tracks]); nte = np.array([int(sum(is_test[int(i)] for i in tr)) for tr in tracks])
    print(f"[points] chỉ ảnh train thấy: {int((nte == 0).sum())} | có ảnh test: {int((nte > 0).sum())} | CHỈ test thấy (ntr=0): {int((ntr == 0).sum())} | track≥2 train: {int((ntr >= 2).sum())}", flush=True)
    n_pl, d_pl, frac_pl, thr_pl = fit_plane(xyz)
    Rs = {iid: qvec2rotmat(im["q"]) for iid, im in ims.items()}; Cs = {iid: -Rs[iid].T @ im["t"] for iid, im in ims.items()}
    if np.median([n_pl @ Cs[i] - d_pl for i in ims]) < 0: n_pl, d_pl = -n_pl, -d_pl
    alts = np.array([n_pl @ Cs[i] - d_pl for i in ims if is_train[i]]); alt_med = float(np.median(alts))
    print(f"[plane] n={np.round(n_pl, 3)} inlier {100*frac_pl:.0f}% (thr {thr_pl:.4f}) | độ cao train med {alt_med:.3f} p10 {np.percentile(alts,10):.3f} p90 {np.percentile(alts,90):.3f} (đơn vị cảnh) | 1 px ≈ alt/f = {alt_med/f:.2e} đơn vị", flush=True)
    # cơ sở mặt phẳng để lấy toạ độ 2D
    e1 = np.cross(n_pl, [1, 0, 0]); e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 0.1 else np.cross(n_pl, [0, 1, 0]) / np.linalg.norm(np.cross(n_pl, [0, 1, 0])); e2 = np.cross(n_pl, e1)
    # đồng-thị: với mỗi ảnh, số điểm chia với từng ảnh khác
    per = {}; pooled = []
    covis = {iid: {} for iid in ims}
    for row, tr in enumerate(tracks):
        tr = [int(i) for i in tr]
        for i in tr:
            for j in tr:
                if j != i: covis[i][j] = covis[i].get(j, 0) + 1
    for iid, im in ims.items():
        R, t, C = Rs[iid], im["t"], Cs[iid]; ok = im["pid"] >= 0
        rows = np.array([id2row.get(int(p), -1) for p in im["pid"][ok]]); xy = im["xy"][ok]; v = rows >= 0; rows, xy = rows[v], xy[v]
        X = xyz[rows]; Xc = X @ R.T + t; z = Xc[:, 2]; uv = Xc[:, :2] / np.clip(z[:, None], 1e-9, None) * f + np.array([cx, cy]); r = xy - uv; rn = np.linalg.norm(r, axis=1)
        axis = R.T @ np.array([0, 0, 1.0]); pitch = float(np.degrees(np.arccos(np.clip(axis @ (-n_pl), -1, 1))))
        up = R.T @ np.array([0, -1.0, 0]); hd = up - (up @ n_pl) * n_pl
        if np.linalg.norm(hd) < 0.2: hd = axis - (axis @ n_pl) * n_pl
        heading = float(np.degrees(np.arctan2(hd @ e2, hd @ e1)))
        it = 1 if is_train[iid] else 0
        shared1 = int(((ntr[rows] - it) >= 1).sum()); shared3 = int(((ntr[rows] - it) >= 3).sum())
        cv = covis[iid]; deg30 = int(sum(1 for j, c in cv.items() if c >= 30 and is_train[j])); top = sorted(((c, j) for j, c in cv.items() if is_train[j]), reverse=True)[:3]
        strong = (ntr[rows] - it) >= 2
        per[im["name"]] = dict(id=iid, name=im["name"], frame=frame_index(im["name"]), train=bool(is_train[iid]), test=bool(is_test[iid]), n_kp=int(len(im["pid"])), n_obs=int(len(rows)),
                               res_med=float(np.median(rn)) if len(rn) else -1, res_p90=float(np.percentile(rn, 90)) if len(rn) else -1, res_gt3=float((rn > 3).mean()) if len(rn) else -1,
                               res_strong_med=float(np.median(rn[strong])) if strong.sum() > 10 else -1, mean_vec=[float(r[:, 0].mean()), float(r[:, 1].mean())] if len(rn) else [0, 0],
                               track_med=float(np.median(tl[rows])) if len(rows) else 0, perr_mean=float(perr[rows].mean()) if len(rows) else -1,
                               shared1=shared1, shared3=shared3, deg30=deg30, top_covis=[[ims[j]["name"][-10:-6], c] for c, j in top],
                               pitch=pitch, heading=heading, alt=float(n_pl @ C - d_pl), pos=[float(C @ e1), float(C @ e2)], z_med=float(np.median(z)) if len(z) else -1)
        if is_train[iid] and len(rn):
            pooled.append(np.c_[xy, r, np.full((len(rn), 1), pitch)])
    # láng giềng train cho từng ảnh (khoảng cách / alt, cùng pitch)
    tr_ids = [i for i in ims if is_train[i]]; tC = np.array([Cs[i] for i in tr_ids]); tP = np.array([per[ims[i]["name"]]["pitch"] for i in tr_ids]); tF = np.array([per[ims[i]["name"]]["frame"] or -999 for i in tr_ids])
    for iid, im in ims.items():
        p = per[im["name"]]; d = np.linalg.norm(tC - Cs[iid], axis=1) / alt_med; dp = np.abs(tP - p["pitch"]); me = np.array([j == iid for j in tr_ids])
        d = np.where(me, np.inf, d); p["nn_train_dist"] = float(d.min()); p["n_train_near"] = int(((d < 0.3) & (dp < 15)).sum()); p["n_train_near_any"] = int((d < 0.3).sum())
        fr = p["frame"] or -999; gaps = np.abs(tF - fr); gaps = np.where(me, 999, gaps); p["frame_gap_train"] = int(gaps.min())
    # dải bay (train, theo frame): ngắt khi heading đổi > 45° hoặc gap frame > 3 hoặc nhảy vị trí > 0.6 alt
    trs = sorted([per[ims[i]["name"]] for i in tr_ids], key=lambda p: p["frame"]); sid = 0; prev = None
    for p in trs:
        if prev is not None:
            dh = abs((p["heading"] - prev["heading"] + 180) % 360 - 180); dpos = np.hypot(*(np.array(p["pos"]) - np.array(prev["pos"]))) / alt_med
            if dh > 45 or (p["frame"] - prev["frame"]) > 3 or dpos > 0.6 or abs(p["pitch"] - prev["pitch"]) > 20: sid += 1
        p["strip"] = sid; prev = p
    for p in per.values():
        if "strip" not in p:  # test: gán theo train frame gần nhất
            k = int(np.argmin(np.abs(tF - (p["frame"] or -999)))); p["strip"] = per[ims[tr_ids[k]]["name"]]["strip"]
    # ---- tóm tắt
    def S(v, q=(10, 50, 90)): v = np.asarray(v, float); return " / ".join(f"{np.percentile(v, x):.1f}" for x in q)
    for grp, sel in (("TRAIN", [p for p in per.values() if p["train"]]), ("TEST", [p for p in per.values() if p["test"]])):
        nad = [p for p in sel if p["pitch"] < 15]; obl = [p for p in sel if p["pitch"] >= 15]
        print(f"\n== {grp} n={len(sel)} (nadir<15° {len(nad)}, oblique {len(obl)}) | pitch p10/50/90 {S([p['pitch'] for p in sel])}")
        print(f"  n_obs p10/50/90 {S([p['n_obs'] for p in sel])} | shared3 {S([p['shared3'] for p in sel])} | deg30 {S([p['deg30'] for p in sel])} | res_med {S([p['res_med'] for p in sel])} px | res>3px {S([100*p['res_gt3'] for p in sel])} % | track_med {S([p['track_med'] for p in sel])}")
        for nm, g in (("nadir", nad), ("oblique", obl)):
            if g: print(f"  {nm:8s}: n_obs med {np.median([p['n_obs'] for p in g]):.0f} | shared3 med {np.median([p['shared3'] for p in g]):.0f} | deg30 med {np.median([p['deg30'] for p in g]):.0f} | res_med med {np.median([p['res_med'] for p in g]):.2f} | %shared3<300: {100*np.mean([p['shared3'] < 300 for p in g]):.0f}")
        if grp == "TEST": print(f"  nn_train_dist/alt p10/50/90 {S([p['nn_train_dist'] for p in sel])} | n_train_near(0.3alt,Δpitch<15) {S([p['n_train_near'] for p in sel])} | frame_gap {S([p['frame_gap_train'] for p in sel])}")
        wk = sorted(sel, key=lambda p: p["shared3"])[:15]
        print("  15 ảnh RÀNG BUỘC YẾU nhất (shared3): " + " ".join(f"{p['name'][-10:-6]}(s3={p['shared3']},obs={p['n_obs']},deg={p['deg30']},res={p['res_med']:.1f},pitch={p['pitch']:.0f})" for p in wk))
        # tương quan trong nhóm
        def sp(x, y):
            x, y = np.asarray(x, float), np.asarray(y, float); rx, ry = x.argsort().argsort(), y.argsort().argsort(); return float(np.corrcoef(rx, ry)[0, 1])
        print(f"  Spearman: shared3~n_obs {sp([p['shared3'] for p in sel],[p['n_obs'] for p in sel]):+.2f} | shared3~pitch {sp([p['shared3'] for p in sel],[p['pitch'] for p in sel]):+.2f} | res_med~n_obs {sp([p['res_med'] for p in sel],[p['n_obs'] for p in sel]):+.2f} | res_med~pitch {sp([p['res_med'] for p in sel],[p['pitch'] for p in sel]):+.2f}")
    strips = {}
    for p in trs: strips.setdefault(p["strip"], []).append(p)
    print(f"\n== DẢI BAY (train): {len(strips)} dải")
    for s_, g in strips.items():
        print(f"  dải {s_:2d}: frame {g[0]['frame']:4d}–{g[-1]['frame']:4d} n={len(g):3d} pitch {np.mean([p['pitch'] for p in g]):5.1f}° heading {np.mean([p['heading'] for p in g]):6.1f}° alt {np.mean([p['alt'] for p in g])/alt_med:5.2f} | shared3 med {np.median([p['shared3'] for p in g]):5.0f} | res_med {np.median([p['res_med'] for p in g]):.2f} | n_test {sum(1 for q in per.values() if q['test'] and q['strip']==s_)}")
    # ---- trường residual gộp (train)
    P = np.concatenate(pooled); u, v, ru, rv, pit = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4]
    def field_fit(mask, tag):
        x = (u[mask] - cx) / f; y = (v[mask] - cy) / f; r2 = x * x + y * y; one = np.ones_like(x); zero = np.zeros_like(x)
        A = np.r_[np.c_[one, zero, x, y, zero, zero, r2 * x], np.c_[zero, one, zero, zero, x, y, r2 * y]]; b = np.r_[ru[mask], rv[mask]]
        sol, *_ = np.linalg.lstsq(A, b, rcond=None); res = A @ sol - b; R2 = 1 - (res ** 2).sum() / ((b - b.mean()) ** 2).sum()
        print(f"  [{tag}] n={mask.sum()} | Δpp=({sol[0]:+.2f},{sol[1]:+.2f}) px | affine [[{sol[2]*f:+.2f},{sol[3]*f:+.2f}],[{sol[4]*f:+.2f},{sol[5]*f:+.2f}]] px/đơn-vị-chuẩn-hoá (Δf/f≈{(sol[2]+sol[5])/2:+.2e}) | radial k·f={sol[6]*f:+.2f} px tại r=1 | R² {R2:.4f} | residual rms trước {np.sqrt((b**2).mean()):.3f} → sau {np.sqrt((res**2).mean()):.3f} px")
    print("\n== TRƯỜNG RESIDUAL GỘP (train): fit r(u,v) = Δpp + A·[x,y] + k·r²·[x,y]")
    field_fit(np.ones(len(u), bool), "tất cả"); field_fit(pit < 15, "nadir"); field_fit(pit >= 15, "oblique")
    gx, gy = np.minimum((u / W * 12).astype(int), 11), np.minimum((v / H * 9).astype(int), 8); cell = {}
    for tag, mk in (("all", np.ones(len(u), bool)),):
        mu = np.zeros((9, 12)); mv = np.zeros((9, 12)); cnt = np.zeros((9, 12))
        np.add.at(mu, (gy, gx), ru); np.add.at(mv, (gy, gx), rv); np.add.at(cnt, (gy, gx), 1); mu /= np.maximum(cnt, 1); mv /= np.maximum(cnt, 1)
        print(f"  ô 12×9 — |mean residual| (×100 px), hàng = v từ trên xuống; max {100*np.hypot(mu, mv).max():.0f}, med {100*np.median(np.hypot(mu, mv)):.0f}")
        for yy in range(9): print("   " + " ".join(f"{100*np.hypot(mu[yy, xx], mv[yy, xx]):4.0f}" for xx in range(12)))
        # radial: thành phần hướng tâm theo bán kính
        rad = np.hypot(u - cx, v - cy) / np.hypot(cx, cy); dirr = np.stack([u - cx, v - cy], 1) / np.maximum(np.hypot(u - cx, v - cy), 1)[:, None]; rr = ru * dirr[:, 0] + rv * dirr[:, 1]
        print("  thành phần hướng tâm theo bán kính (r/rmax → mean ×100 px): " + " ".join(f"{lo:.1f}:{100*rr[(rad>=lo)&(rad<lo+0.1)].mean():+.0f}" for lo in np.arange(0, 1.0, 0.1)))
        cell = dict(mu=mu.tolist(), mv=mv.tolist(), cnt=cnt.tolist())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(cam=dict(f=f, cx=cx, cy=cy, W=W, H=H), plane=dict(n=n_pl.tolist(), d=float(d_pl), inlier=frac_pl, alt_med=alt_med, e1=e1.tolist(), e2=e2.tolist()), n_points=int(len(ids)), per_image=per, cell=cell), open(a.out, "w"), indent=1)
    print("Y01_DONE", flush=True)


if __name__ == "__main__":
    main()
