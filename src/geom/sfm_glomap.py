"""sfm_glomap.py — SfM TOÀN CỤC (GLOMAP, có sẵn trong pycolmap 4.2 `global_mapping`) trên DB match dày (<ba_dir>/db.db), thay cho
incremental mapping (sfm_rebuild.py). Biến thể camera: OPENCV / OPENCV + tinh chỉnh tâm quang / FULL_OPENCV (k1..k6,p1,p2).
Đánh giá cùng thước với đường incremental: Sampson khử méo trên cặp giữ lại theo bin track chung (check_geometry), profile bán kính, pivot so BTC.
  python sfm_glomap.py --scene_dir $VT_SCENE --match_dir $VT_RUNS/geom/match --ba_dir $VT_RUNS/geom/db \
    --out $VT_RUNS/geom/sfm [--cam_model OPENCV|FULL_OPENCV] [--pp 1] [--focal 1] [--incremental 1]
"""
import argparse, os, sys, json, shutil, time, numpy as np, pycolmap
_H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _H)
from sfm_common import holdout_eval, radial_profile, pose_diff, align_to_btc, rec_stats, load_matches, KP, btc_pts, rec_w2c  # noqa
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene_dir", required=True); ap.add_argument("--match_dir", required=True); ap.add_argument("--ba_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--audit", default=""); ap.add_argument("--cam_model", default="OPENCV"); ap.add_argument("--pp", type=int, default=0); ap.add_argument("--focal", type=int, default=0); ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--incremental", type=int, default=0, help="1: incremental_mapping (như sfm_rebuild.py) thay vì GLOMAP — để tách biến camera model vs mapper"); ap.add_argument("--min_matches", type=int, default=30)
    ap.add_argument("--hold_dir", default="", help="thư mục có holdout_sampson.json để eval (mặc định = ba_dir)"); ap.add_argument("--ba_iters", type=int, default=3); ap.add_argument("--min_track", type=int, default=3); ap.add_argument("--final_ba", type=int, default=1, help="1: sau mapping chạy thêm 1 BA toàn cục Huber 1px (pose+điểm+camera) rồi lọc >4px"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time(); tag = os.path.basename(a.out.rstrip("/"))
    img_dir = os.path.join(a.scene_dir, "train", "images"); rec0 = pycolmap.Reconstruction(os.path.join(a.scene_dir, "train", "sparse", "0"))
    db_path = os.path.join(a.out, "db.db"); shutil.copy(os.path.join(a.ba_dir, "db.db"), db_path); db = pycolmap.Database.open(db_path); cam = db.read_all_cameras()[0]
    if cam.model.name != a.cam_model:
        f, cx, cy = float(cam.params[0]), float(cam.params[2]) if cam.model.name in ("OPENCV", "FULL_OPENCV", "PINHOLE") else float(cam.params[1]), float(cam.params[3]) if cam.model.name in ("OPENCV", "FULL_OPENCV", "PINHOLE") else float(cam.params[2])
        params = {"OPENCV": [f, f, cx, cy, 0, 0, 0, 0], "FULL_OPENCV": [f, f, cx, cy, 0, 0, 0, 0, 0, 0, 0, 0], "RADIAL": [f, cx, cy, 0, 0], "PINHOLE": [f, f, cx, cy], "SIMPLE_RADIAL": [f, cx, cy, 0]}[a.cam_model]
        cam2 = pycolmap.Camera(model=a.cam_model, width=cam.width, height=cam.height, params=params, camera_id=cam.camera_id); cam2.has_prior_focal_length = True; db.update_camera(cam2); print(f"[glomap] camera DB {cam.model.name} → {a.cam_model} {params}", flush=True)
    print(f"[glomap] DB: {db.num_images()} ảnh, {db.num_matched_image_pairs()} cặp match, {db.num_verified_image_pairs()} cặp verified | camera {db.read_all_cameras()[0].model.name}", flush=True); db.close()
    if a.incremental:
        opt = pycolmap.IncrementalPipelineOptions(); opt.num_threads = a.threads; opt.ba_refine_focal_length = bool(a.focal); opt.ba_refine_principal_point = bool(a.pp); opt.ba_refine_extra_params = a.cam_model != "PINHOLE"
        opt.min_num_matches = a.min_matches; opt.multiple_models = False; opt.max_num_models = 1
        recs = pycolmap.incremental_mapping(db_path, img_dir, a.out, options=opt)
    else:
        opt = pycolmap.GlobalPipelineOptions(); opt.num_threads = a.threads; opt.min_num_matches = a.min_matches; opt.multiple_models = False
        m = opt.mapper; m.num_threads = a.threads; m.ba_num_iterations = a.ba_iters; m.track_min_num_views_per_track = a.min_track
        m.bundle_adjustment.refine_focal_length = bool(a.focal); m.bundle_adjustment.refine_principal_point = bool(a.pp); m.bundle_adjustment.refine_extra_params = a.cam_model != "PINHOLE"
        print("[glomap] GLOMAP options:", {k: v for k, v in m.todict().items() if not isinstance(v, dict)}, flush=True)
        recs = pycolmap.global_mapping(db_path, img_dir, a.out, options=opt)
    if not recs: print("[glomap] không dựng được model"); return
    rec = max(recs.values(), key=lambda r: r.num_reg_images()); rec_stats(rec, f"{tag} sau mapping ({time.time() - t0:.0f}s)")
    if a.final_ba:
        bo = pycolmap.BundleAdjustmentOptions(); bo.refine_focal_length = bool(a.focal); bo.refine_principal_point = bool(a.pp); bo.refine_extra_params = a.cam_model != "PINHOLE"; bo.refine_rig_from_world = True; bo.refine_points3D = True
        bo.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER; bo.ceres.loss_function_scale = 1.0; bo.ceres.solver_options.max_num_iterations = 100
        pycolmap.bundle_adjustment(rec, bo); om = pycolmap.ObservationManager(rec); nf = om.filter_all_points3D(4.0, 1.0); pycolmap.bundle_adjustment(rec, bo); rec_stats(rec, f"{tag} sau BA cuối (lọc {nf})")
    sc = align_to_btc(rec, rec0); print(f"[glomap] Sim3 về khung BTC: scale {sc:.6f}", flush=True)
    sp = os.path.join(a.out, "sparse", "0"); os.makedirs(sp, exist_ok=True); rec.write_binary(sp)
    w2c = rec_w2c(rec); json.dump({n: v.tolist() for n, v in w2c.items()}, open(os.path.join(a.out, "train_w2c_new.json"), "w"))
    # đánh giá
    hold = json.load(open(os.path.join(a.hold_dir or a.ba_dir, "holdout_sampson.json")))["hold_pairs"]; kp = KP(a.match_dir); pts = btc_pts(a.scene_dir); Mh = load_matches(a.match_dir, hold)
    res = dict(tag=tag, cam=[float(p) for p in next(iter(rec.cameras.values())).params], n_img=rec.num_reg_images(), n_pts=rec.num_points3D(), secs=time.time() - t0)
    res["hold"] = holdout_eval(rec, hold, Mh, kp, pts, tag)
    if a.audit:
        au = {k: v for k, v in json.load(open(a.audit))["per_image"].items()}; Ma = load_matches(a.match_dir); res["radial"] = radial_profile(rec, Ma, kp, pts, au, tag, min_shared=300, max_dfr=3)
    res["pose"] = pose_diff(rec, rec0, tag); json.dump(res, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    print(f"Y61_DONE {tag} | hold med {res['hold']['med']:.2f} | bin0 {res['hold']['bins'].get('0-1', {}).get('med', float('nan')):.2f} | {time.time() - t0:.0f}s", flush=True)
if __name__ == "__main__": main()
