"""sfm_rebuild.py — ĐƯỜNG LÙI: dựng SfM từ đầu bằng match dày (pycolmap incremental_mapping, K BTC cố định) trên DB của match_dense (<ba_dir>/db.db, chỉ ảnh train),
lấy model lớn nhất, Sim3 về khung BTC theo tâm camera, kiểm Sampson cặp giữ lại (cùng cặp holdout của <ba_dir>), ghi sparse + json pose như pairs_db.
  python sfm_rebuild.py --scene_dir $VT_SCENE --match_dir $VT_RUNS/geom/match --ba_dir $VT_RUNS/geom/db --out $VT_RUNS/geom/sfm"""
import argparse, os, sys, json, math, time, glob, numpy as np, pycolmap
_H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _H)
from pairs_db import sampson, w2c_of  # noqa
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene_dir", required=True); ap.add_argument("--match_dir", required=True); ap.add_argument("--ba_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=32); ap.add_argument("--refine_extra", type=int, default=0, help="1: BA trong mapping tinh chỉnh tham số méo (DB phải có camera OPENCV)"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    img_dir = os.path.join(a.scene_dir, "train", "images"); rec0 = pycolmap.Reconstruction(os.path.join(a.scene_dir, "train", "sparse", "0")); cam0 = next(iter(rec0.cameras.values())); K = cam0.calibration_matrix()
    db_path = os.path.join(a.ba_dir, "db.db"); db = pycolmap.Database.open(db_path); ids = {im.name: im.image_id for im in db.read_all_images()}; db.close()
    opt = pycolmap.IncrementalPipelineOptions(); opt.num_threads = a.threads; opt.ba_refine_focal_length = False; opt.ba_refine_principal_point = False; opt.ba_refine_extra_params = bool(a.refine_extra)
    opt.min_num_matches = 30; opt.multiple_models = False; opt.max_num_models = 1
    print(f"[sfm_rebuild] incremental_mapping trên {len(ids)} ảnh (K cố định)…", flush=True)
    recs = pycolmap.incremental_mapping(db_path, img_dir, a.out, options=opt)
    if not recs: print("[sfm_rebuild] không dựng được model"); return
    rec = max(recs.values(), key=lambda r: r.num_reg_images()); print(f"[sfm_rebuild] model lớn nhất: {rec.num_reg_images()}/{len(ids)} ảnh, {rec.num_points3D()} điểm | reproj mean {np.mean([p.error for p in rec.points3D.values()]):.2f} px | track mean {np.mean([p.track.length() for p in rec.points3D.values()]):.1f} | {time.time() - t0:.0f}s", flush=True)
    sim = pycolmap.align_reconstructions_via_proj_centers(rec, rec0, 100.0); rec.transform(sim); print(f"[sfm_rebuild] Sim3 về khung BTC: scale {sim.scale:.6f}", flush=True)
    cm = next(iter(rec.cameras.values())); print("[sfm_rebuild] camera sau mapping:", cm.model.name, [round(float(p), 6) for p in cm.params], flush=True)
    names = [n for n in ids if any(im.name == n and im.has_pose for im in rec.images.values())]
    new = {}; old = {}
    for im in rec.images.values():
        if im.has_pose: new[im.name] = w2c_of(rec, im.image_id)
    for im in rec0.images.values():
        if im.name in new: M = np.eye(4); M[:3, :4] = (im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world).matrix(); old[im.name] = M
    kp = {n: np.load(os.path.join(a.match_dir, "kp", n + ".npy")).astype(np.float32) for n in new}
    hold = json.load(open(os.path.join(a.ba_dir, "holdout_sampson.json")))["hold_pairs"]; M = {}
    for p in sorted(glob.glob(os.path.join(a.match_dir, "matches*.npz"))): z = np.load(p); M.update({k: z[k] for k in z.files if k in set(hold)})
    for tag, w2c in (("pose BTC", old), ("SfM mới", new)):
        vals = []; fr3 = []
        for k in hold:
            na, nb = k.split("|")
            if na not in w2c or nb not in w2c or k not in M: continue
            m = M[k]; s = sampson(kp[na][m[:, 0]], kp[nb][m[:, 1]], w2c[na], w2c[nb], K); vals.append(float(np.median(s))); fr3.append(float((s < 3).mean()))
        vals = np.array(vals); fr3 = np.array(fr3); print(f"[sfm_rebuild] Sampson cặp GIỮ LẠI ({tag}, n={len(vals)}): med-của-med {np.median(vals):.2f} p25 {np.percentile(vals, 25):.2f} p75 {np.percentile(vals, 75):.2f} | %match<3px {100 * fr3.mean():.0f} | %cặp ≥50% match<3px {100 * (fr3 >= 0.5).mean():.0f}", flush=True)
    cen = [float(np.linalg.norm((-new[n][:3, :3].T @ new[n][:3, 3]) - (-old[n][:3, :3].T @ old[n][:3, 3]))) for n in new]; rot = [float(np.degrees(np.arccos(np.clip((np.trace(old[n][:3, :3].T @ new[n][:3, :3]) - 1) / 2, -1, 1)))) for n in new]
    print(f"[sfm_rebuild] so pose BTC: dịch tâm med {np.median(cen):.4f} p90 {np.percentile(cen, 90):.4f} | xoay med {np.median(rot):.3f}° p90 {np.percentile(rot, 90):.3f}° ≈ {np.median(rot) / 180 * math.pi * K[0, 0]:.1f} / {np.percentile(rot, 90) / 180 * math.pi * K[0, 0]:.1f} px", flush=True)
    sp = os.path.join(a.out, "sparse", "0"); os.makedirs(sp, exist_ok=True); rec.write_binary(sp)
    json.dump({n: v.tolist() for n, v in new.items()}, open(os.path.join(a.out, "train_w2c_new.json"), "w")); json.dump({n: v.tolist() for n, v in old.items()}, open(os.path.join(a.out, "train_w2c_old.json"), "w"))
    print("Y32_DONE", flush=True)
if __name__ == "__main__": main()
