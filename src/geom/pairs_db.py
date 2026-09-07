"""pairs_db.py — BA LẠI pose TRAIN với match dày (match_dense.py) bằng pycolmap 4.2: DB mới (camera BTC cố định) → tam giác hoá với pose BTC
→ bundle adjustment (K cố định, pose + điểm tự do, loss Huber) → Sim3 căn về khung BTC (tâm camera) → ghi sparse mới + json pose.
Kiểm chứng: Sampson trên 10 % cặp GIỮ LẠI (không đưa vào BA) dưới pose cũ vs mới. KHÔNG dùng ảnh/keypoint test.
  python pairs_db.py --scene_dir $VT_SCENE --match_dir $VT_RUNS/geom/match --out $VT_RUNS/geom/db [--holdout_frac 0.1] [--huber 2.0] [--iters 100]
"""
import argparse, os, sys, json, math, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa
import pycolmap
def skew(t): return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
def sampson(pa, pb, Wa, Wb, K):
    Rab = Wb[:3, :3] @ Wa[:3, :3].T; tab = Wb[:3, 3] - Rab @ Wa[:3, 3]; E = skew(tab / (np.linalg.norm(tab) + 1e-12)) @ Rab; Fm = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    ha = np.c_[pa, np.ones(len(pa))]; hb = np.c_[pb, np.ones(len(pb))]; l_b = ha @ Fm.T; l_a = hb @ Fm
    return np.abs((hb * l_b).sum(1)) / np.sqrt(l_b[:, 0] ** 2 + l_b[:, 1] ** 2 + l_a[:, 0] ** 2 + l_a[:, 1] ** 2 + 1e-12)
def w2c_of(rec, iid):
    im = rec.images[iid]; M = np.eye(4); M[:3, :4] = im.cam_from_world().matrix() if callable(im.cam_from_world) else im.cam_from_world.matrix(); return M
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene_dir", required=True); ap.add_argument("--match_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--holdout_frac", type=float, default=0.1); ap.add_argument("--huber", type=float, default=2.0); ap.add_argument("--iters", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_track", type=int, default=3); ap.add_argument("--max_reproj", type=float, default=4.0); ap.add_argument("--snap", type=float, default=2.0); ap.add_argument("--kp_half", type=int, default=0)
    ap.add_argument("--db_only", type=int, default=0, help="07/09: 1 = chỉ ghi DB + holdout_sampson.json rồi thoát (cho GLOMAP)"); ap.add_argument("--tri_reproj", type=float, default=30.0, help="ngưỡng tái chiếu khi tam giác hoá dưới pose BTC (px) — phải lớn hơn độ lệch cần sửa"); ap.add_argument("--tri_angle", type=float, default=0.5); ap.add_argument("--loss1", type=float, default=8.0); ap.add_argument("--cam_model", default="", help="vd OPENCV: nâng camera lên model có méo, BA tinh chỉnh k1,k2,p1,p2 (giả thuyết méo còn sót)"); ap.add_argument("--refine_f", type=int, default=0); ap.add_argument("--fix_dist", default="", help="RADIAL k1,k2 cố định (vd -0.02,-0.02): camera RADIAL, không tinh chỉnh méo — chọn gauge méo nhỏ")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    img_dir = os.path.join(a.scene_dir, "train", "images"); rec0 = pycolmap.Reconstruction(os.path.join(a.scene_dir, "train", "sparse", "0"))
    cam0 = next(iter(rec0.cameras.values())); K = cam0.calibration_matrix(); print(f"[pairs_db] camera BTC {cam0.model.name} {list(cam0.params)}", flush=True)
    train_names = set(os.listdir(img_dir)); old = {im.name: im for im in rec0.images.values() if im.name in train_names}
    import glob as _g; metas = [json.load(open(p)) for p in sorted(_g.glob(os.path.join(a.match_dir, "match_meta*.json")))]; names = [n for n in metas[0]["names"] if n in old]; print(f"[pairs_db] {len(names)} ảnh train, {len(rec0.images) - len(old)} ảnh test bị loại", flush=True)
    # 1) DB
    db_path = os.path.join(a.out, "db.db"); [os.remove(p) for p in (db_path,) if os.path.exists(p)]
    opts = pycolmap.ImageReaderOptions()
    if a.fix_dist: k1, k2 = [float(x) for x in a.fix_dist.split(",")]; opts.camera_model = "RADIAL"; opts.camera_params = ",".join(f"{p:.10g}" for p in (cam0.params[0], cam0.params[1], cam0.params[2], k1, k2))
    elif a.cam_model == "OPENCV": opts.camera_model = "OPENCV"; opts.camera_params = ",".join(f"{p:.10g}" for p in (cam0.params[0], cam0.params[0], cam0.params[1], cam0.params[2], 0, 0, 0, 0))
    else: opts.camera_model = cam0.model.name; opts.camera_params = ",".join(f"{p:.10g}" for p in cam0.params)
    pycolmap.Database.open(db_path).close()  # import_images đòi file DB có sẵn
    pycolmap.import_images(db_path, img_dir, pycolmap.CameraMode.SINGLE, names, opts)
    db = pycolmap.Database.open(db_path); ids = {im.name: im.image_id for im in db.read_all_images()}; cams = db.read_all_cameras(); print(f"[pairs_db] DB: {len(ids)} ảnh, camera {[(c.camera_id, c.model.name, list(c.params)) for c in cams]}", flush=True)
    kp = {n: np.load(os.path.join(a.match_dir, "kp", n + ".npy")).astype(np.float32) for n in names}
    # gộp keypoint: file kp = [k0 (gốc); k1 (xoay 180°)]; k1 gần k0 ≤ snap px → dùng chỉ số k0 (track nối được qua hai bản)
    from scipy.spatial import cKDTree
    remap = {}; n_snap = 0; n_tot = 0
    for n in names:
        k = kp[n]; h = len(k) // 2 if a.kp_half <= 0 else a.kp_half  # số kp gốc = nửa đầu (2 tập cùng max_kp, có thể lệch vài) → đọc từ meta nếu có
        _pj = os.path.join(a.match_dir, "kp", n + ".json"); _meta = json.load(open(_pj)) if os.path.exists(_pj) else {}
        if "n0" in _meta: h = int(_meta["n0"])  # y69: SIFT/ALIKED trả số kp gốc ≠ xoay
        if "segments" in _meta:
            # y69_merge: nhiều đoạn [A gốc; A xoay; B gốc; B xoay] — snap mọi keypoint về keypoint CHUẨN gần nhất (đoạn trước) ≤ snap px, đoạn đầu là chuẩn
            idx = np.arange(len(k)); s0, e0 = _meta["segments"][0]; canon = k[s0:e0]; tree = cKDTree(canon); cnt = 0
            for (sa, ea) in _meta["segments"][1:]:
                d, j = tree.query(k[sa:ea], distance_upper_bound=a.snap); ok = np.isfinite(d); idx[sa:ea][ok] = j[ok] + s0; cnt += int(ok.sum())
            remap[n] = idx; n_snap += cnt; n_tot += len(k) - (e0 - s0); continue
        k0, k1 = k[:h], k[h:]; d, j = cKDTree(k0).query(k1, distance_upper_bound=a.snap); ok = np.isfinite(d)
        idx = np.arange(len(k)); idx[h:][ok] = j[ok]; remap[n] = idx; n_snap += int(ok.sum()); n_tot += len(k1)
    print(f"[pairs_db] gộp keypoint xoay→gốc ≤{a.snap}px: {n_snap}/{n_tot} ({100 * n_snap / max(n_tot, 1):.0f}%)", flush=True)
    for n in names: db.write_keypoints(ids[n], kp[n])
    M = {}
    for p in sorted(_g.glob(os.path.join(a.match_dir, "matches*.npz"))):
        z = np.load(p); M.update({k: z[k] for k in z.files})
    print(f"[pairs_db] nạp {len(M)} cặp match từ {len(_g.glob(os.path.join(a.match_dir, 'matches*.npz')))} file", flush=True); keys = sorted(M); rng = np.random.default_rng(a.seed); rng.shuffle(keys)
    n_hold = int(len(keys) * a.holdout_frac); hold, use = keys[:n_hold], keys[n_hold:]; n_w = 0
    for k in use:
        na, nb = k.split("|")
        if na not in ids or nb not in ids: continue
        m = M[k].astype(np.int64); m = np.stack([remap[na][m[:, 0]], remap[nb][m[:, 1]]], 1); m = np.unique(m, axis=0).astype(np.uint32); db.write_matches(ids[na], ids[nb], m); tvg = pycolmap.TwoViewGeometry(); tvg.inlier_matches = m; tvg.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED; db.write_two_view_geometry(ids[na], ids[nb], tvg); n_w += 1
    db.close(); print(f"[pairs_db] ghi {n_w} cặp vào DB, giữ lại {len(hold)} cặp kiểm chứng | {time.time() - t0:.0f}s", flush=True)
    if a.db_only:   # 07/09: chỉ cần DB + cặp giữ lại cho GLOMAP (sfm_glomap.py) — bỏ tam giác hoá + BA từ pose BTC (~55 phút CPU)
        _by = {im.name: im.image_id for im in rec0.images.values()}; json.dump({n: w2c_of(rec0, _by[n]).tolist() for n in names if n in _by}, open(os.path.join(a.out, "train_w2c_old.json"), "w"))   # pose BTC cho build_scene.py
        json.dump(dict(hold_old=[], hold_new=[], hold_pairs=hold), open(os.path.join(a.out, "holdout_sampson.json"), "w")); print("Y30_BA_DONE (db_only)", flush=True); return
    # 2) reconstruction train-only với pose BTC, id theo DB
    rec = pycolmap.Reconstruction(); cam = pycolmap.Camera(); cam.camera_id = cams[0].camera_id; cam.model = cams[0].model; cam.width = cams[0].width; cam.height = cams[0].height; cam.params = cams[0].params
    rec.add_camera_with_trivial_rig(cam)
    for n in names:
        im = pycolmap.Image(name=n, camera_id=cam.camera_id, image_id=ids[n]); rec.add_image_with_trivial_frame(im)
        fr = rec.frames[rec.images[ids[n]].frame_id]; fr.rig_from_world = old[n].cam_from_world() if callable(old[n].cam_from_world) else old[n].cam_from_world; rec.register_frame(fr.frame_id)
    print(f"[pairs_db] rec: {rec.num_images()} ảnh, {rec.num_reg_images() if hasattr(rec, 'num_reg_images') else '?'} đăng ký", flush=True)
    # 3) tam giác hoá với pose cố định
    tri_dir = os.path.join(a.out, "tri"); os.makedirs(tri_dir, exist_ok=True); popt = pycolmap.IncrementalPipelineOptions()
    ang = math.degrees(a.tri_reproj / K[0, 0])  # px → độ
    for k_, v_ in dict(create_max_angle_error=ang, continue_max_angle_error=ang, merge_max_reproj_error=a.tri_reproj, complete_max_reproj_error=a.tri_reproj, re_max_angle_error=ang, min_angle=a.tri_angle, ignore_two_view_tracks=False).items():
        try: setattr(popt.triangulation, k_, v_)
        except Exception as e: print("[pairs_db] tri opt", k_, e)
    for k_, v_ in dict(filter_max_reproj_error=a.tri_reproj, filter_min_tri_angle=a.tri_angle).items():
        try: setattr(popt.mapper, k_, v_)
        except Exception as e: print("[pairs_db] mapper opt", k_, e)
    print(f"[pairs_db] tam giác hoá nới: reproj ≤ {a.tri_reproj} px (= {ang:.2f}°), góc ≥ {a.tri_angle}°, giữ track 2-view", flush=True)
    rec = pycolmap.triangulate_points(rec, db_path, img_dir, tri_dir, clear_points=True, options=popt, refine_intrinsics=False)
    def stats(r, tag):
        errs = []; tl = []
        for p in r.points3D.values(): tl.append(p.track.length()); errs.append(p.error)
        print(f"[pairs_db] {tag}: {r.num_points3D()} điểm | track med {np.median(tl):.0f} mean {np.mean(tl):.1f} | reproj mean {np.mean(errs):.2f} px | {time.time() - t0:.0f}s", flush=True)
    stats(rec, "tam giác hoá (pose BTC)")
    old_w2c = {n: w2c_of(rec, ids[n]) for n in names}
    sys.path.insert(0, _H); from sfm_audit import read_images as _ri
    _ims = _ri(os.path.join(a.scene_dir, "train", "sparse", "0", "images.bin")); _pts = {im["name"]: set(int(i) for i in im["pid"][im["pid"] >= 0]) for im in _ims.values()}
    shared = {k: len(_pts.get(k.split("|")[0], set()) & _pts.get(k.split("|")[1], set())) for k in hold}
    def holdout_sampson(w2c, tag):
        vals = []; fr3 = []; sh = []
        if not hold: print(f"[pairs_db] không có cặp giữ lại ({tag})", flush=True); return np.zeros(0)
        for k in hold:
            na, nb = k.split("|")
            if na not in w2c or nb not in w2c: continue
            m = M[k]; s = sampson(kp[na][m[:, 0]], kp[nb][m[:, 1]], w2c[na], w2c[nb], K); vals.append(float(np.median(s))); fr3.append(float((s < 3).mean())); sh.append(shared[k])
        vals = np.array(vals); fr3 = np.array(fr3); sh = np.array(sh); print(f"[pairs_db] Sampson cặp GIỮ LẠI ({tag}, n={len(vals)}): med-của-med {np.median(vals):.2f} p25 {np.percentile(vals, 25):.2f} p75 {np.percentile(vals, 75):.2f} | %match<3px {100 * fr3.mean():.0f} | %cặp ≥50%<3px {100 * (fr3 >= 0.5).mean():.0f}", flush=True)
        for lo, hi in ((0, 1), (1, 20), (20, 100), (100, 10 ** 9)):
            mm = (sh >= lo) & (sh < hi)
            if mm.sum(): print(f"      track chung BTC [{lo},{hi}): n={int(mm.sum()):4d} | Sampson med {np.median(vals[mm]):.2f} p75 {np.percentile(vals[mm], 75):.2f} | %match<3px {100 * fr3[mm].mean():.0f}", flush=True)
        return vals
    s_old = holdout_sampson(old_w2c, "pose BTC")
    # 4) BA: K cố định, pose + điểm tự do, Huber
    bo = pycolmap.BundleAdjustmentOptions(); bo.refine_focal_length = bool(a.refine_f); bo.refine_principal_point = False; bo.refine_extra_params = bool(a.cam_model) and not a.fix_dist; bo.refine_rig_from_world = True; bo.refine_sensor_from_rig = False; bo.refine_points3D = True
    bo.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER; bo.ceres.loss_function_scale = a.huber; bo.ceres.solver_options.max_num_iterations = a.iters; bo.print_summary = True
    try:
        om = pycolmap.ObservationManager(rec); n_f = om.filter_all_points3D(1e9, max(a.tri_angle, 1.0)); print(f"[pairs_db] lọc trước BA: {n_f} điểm góc <{max(a.tri_angle, 1.0)}°", flush=True)
    except Exception as e: print("[pairs_db] không lọc được:", e)
    sched = [(pycolmap.LossFunctionType.HUBER, a.loss1, None), (pycolmap.LossFunctionType.HUBER, a.loss1 / 3, a.tri_reproj), (pycolmap.LossFunctionType.HUBER, a.huber, a.max_reproj * 2), (pycolmap.LossFunctionType.HUBER, a.huber, a.max_reproj)]
    for rnd, (lt, ls, fr) in enumerate(sched):
        bo.ceres.loss_function_type = lt; bo.ceres.loss_function_scale = ls
        pycolmap.bundle_adjustment(rec, bo); stats(rec, f"BA vòng {rnd + 1} ({lt.name} {ls}px)")
        if fr is None: continue
        try:
            om = pycolmap.ObservationManager(rec); n_f = om.filter_all_points3D(fr, a.tri_angle); print(f"[pairs_db] lọc {n_f} quan sát/điểm xấu (>{fr} px hoặc góc <{a.tri_angle}°)", flush=True)
        except Exception as e: print("[pairs_db] không lọc được:", e)
    # 5) căn Sim3 về khung BTC theo tâm camera
    try:
        sim = pycolmap.align_reconstructions_via_proj_centers(rec, rec0, 100.0); rec.transform(sim); print(f"[pairs_db] Sim3 về khung BTC: scale {sim.scale:.6f}", flush=True)
    except Exception as e:
        print("[pairs_db] align lỗi, tự căn:", e); import scipy  # fallback không dùng
    stats(rec, "sau căn"); print("[pairs_db] camera sau BA:", next(iter(rec.cameras.values())).model.name, [round(float(p), 6) for p in next(iter(rec.cameras.values())).params], flush=True)
    new_w2c = {n: w2c_of(rec, ids[n]) for n in names}; s_new = holdout_sampson(new_w2c, "pose MỚI")
    # 6) thống kê dịch chuyển pose
    gsd = []; rot = []; cen = []
    for n in names:
        Wo, Wn = old_w2c[n], new_w2c[n]; Co, Cn = -Wo[:3, :3].T @ Wo[:3, 3], -Wn[:3, :3].T @ Wn[:3, 3]; z = float(Wo[2, 3]) if False else None
        cen.append(float(np.linalg.norm(Cn - Co))); rot.append(float(np.degrees(np.arccos(np.clip((np.trace(Wo[:3, :3].T @ Wn[:3, :3]) - 1) / 2, -1, 1)))))
    alt = np.median([abs(v) for v in [old[n].projection_center()[2] for n in names]]) if False else None
    print(f"[pairs_db] dịch tâm camera: med {np.median(cen):.4f} p90 {np.percentile(cen, 90):.4f} (đơn vị cảnh) | xoay: med {np.median(rot):.4f}° p90 {np.percentile(rot, 90):.4f}° ≈ {np.median(rot) / 180 * math.pi * K[0, 0]:.1f} px / {np.percentile(rot, 90) / 180 * math.pi * K[0, 0]:.1f} px", flush=True)
    out_sp = os.path.join(a.out, "sparse", "0"); os.makedirs(out_sp, exist_ok=True); rec.write_binary(out_sp)
    json.dump({n: new_w2c[n].tolist() for n in names}, open(os.path.join(a.out, "train_w2c_new.json"), "w")); json.dump({n: old_w2c[n].tolist() for n in names}, open(os.path.join(a.out, "train_w2c_old.json"), "w"))
    json.dump(dict(hold_old=s_old.tolist(), hold_new=s_new.tolist(), hold_pairs=hold), open(os.path.join(a.out, "holdout_sampson.json"), "w"))
    print("Y30_BA_DONE", flush=True)
if __name__ == "__main__": main()
