"""build_scene.py — CHUYỂN pose TEST sang khung mới theo LÁNG GIỀNG train (không dùng pixel/keypoint ảnh test): với mỗi view test, các ảnh train
|Δframe| ≤ 2 cho biến đổi thế giới cũ→mới T_i = c2w_new_i ∘ w2c_old_i; lấy trung bình (quaternion + tịnh tiến, trọng số 1/|Δfr|) → c2w_new_test = T ∘ c2w_old_test.
Dựng scene biến thể: train/images → gốc, train/sparse/0 = BA mới, test/images → gốc (chấm), test/test_poses.csv = pose test mới.
  python build_scene.py --scene_dir $VT_SCENE --ba_dir $VT_RUNS/geom/sfm_ocv --out_scene $VT_SCENE_FIXED_RAW
"""
import argparse, os, sys, json, csv, shutil, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa
from dataset import frame_index, qvec2rotmat  # noqa
def rotmat2qvec(R):
    q = np.empty(4); t = np.trace(R)
    if t > 0: s = np.sqrt(t + 1) * 2; q[0] = 0.25 * s; q[1] = (R[2, 1] - R[1, 2]) / s; q[2] = (R[0, 2] - R[2, 0]) / s; q[3] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]: s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; q[0] = (R[2, 1] - R[1, 2]) / s; q[1] = 0.25 * s; q[2] = (R[0, 1] + R[1, 0]) / s; q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]: s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; q[0] = (R[0, 2] - R[2, 0]) / s; q[1] = (R[0, 1] + R[1, 0]) / s; q[2] = 0.25 * s; q[3] = (R[1, 2] + R[2, 1]) / s
    else: s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; q[0] = (R[1, 0] - R[0, 1]) / s; q[1] = (R[0, 2] + R[2, 0]) / s; q[2] = (R[1, 2] + R[2, 1]) / s; q[3] = 0.25 * s
    return q / np.linalg.norm(q)
def avg_rigid(Ts, ws):
    qs = np.array([rotmat2qvec(T[:3, :3]) for T in Ts]); qs[qs @ qs[0] < 0] *= -1; q = (qs * np.array(ws)[:, None]).sum(0); q /= np.linalg.norm(q)
    t = (np.array([T[:3, 3] for T in Ts]) * np.array(ws)[:, None]).sum(0) / sum(ws); T = np.eye(4); T[:3, :3] = qvec2rotmat(q); T[:3, 3] = t; return T
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene_dir", required=True); ap.add_argument("--ba_dir", required=True); ap.add_argument("--out_scene", required=True); ap.add_argument("--max_dfr", type=int, default=2); ap.add_argument("--offsets", default="3:1.0,2:0.6,6:0.25,4:0.15", help="Δframe:trọng số — theo y33: view test chia track với train ở ±3 (192/110), ±2 (121/56), ±6 (34/38), ±4 (20/7); ±1 ≈ 0"); a = ap.parse_args()
    offs = [(int(k), float(v)) for k, v in (x.split(":") for x in a.offsets.split(","))]
    old = {n: np.array(v) for n, v in json.load(open(os.path.join(a.ba_dir, "train_w2c_old.json"))).items()}; new = {n: np.array(v) for n, v in json.load(open(os.path.join(a.ba_dir, "train_w2c_new.json"))).items()}
    byfr = {frame_index(n): n for n in old}; f = float(next(csv.DictReader(open(os.path.join(a.scene_dir, "test", "test_poses.csv"))))["fx"])
    rows = list(csv.DictReader(open(os.path.join(a.scene_dir, "test", "test_poses.csv")))); out_rows = []; shifts = []; nnb = []
    for r in rows:
        q = np.array([float(r[k]) for k in ("qw", "qx", "qy", "qz")]); t = np.array([float(r[k]) for k in ("tx", "ty", "tz")]); Wo = np.eye(4); Wo[:3, :3] = qvec2rotmat(q); Wo[:3, 3] = t
        fr = frame_index(r["image_name"]); Ts, ws = [], []
        for d, w in offs:
            for s in (-d, d):
                n = byfr.get(fr + s)
                if n is None: continue
                c2w_new = np.linalg.inv(new[n]); Ts.append(c2w_new @ old[n]); ws.append(w)
        if not Ts: out_rows.append(r); shifts.append(0.0); nnb.append(0); continue
        T = avg_rigid(Ts, ws); Wn = np.linalg.inv(T @ np.linalg.inv(Wo)); qn = rotmat2qvec(Wn[:3, :3]); tn = Wn[:3, 3]
        dR = np.degrees(np.arccos(np.clip((np.trace(Wo[:3, :3].T @ Wn[:3, :3]) - 1) / 2, -1, 1))); shifts.append(float(np.radians(dR) * f)); nnb.append(len(Ts))
        r2 = dict(r); r2.update(qw=f"{qn[0]:.17g}", qx=f"{qn[1]:.17g}", qy=f"{qn[2]:.17g}", qz=f"{qn[3]:.17g}", tx=f"{tn[0]:.17g}", ty=f"{tn[1]:.17g}", tz=f"{tn[2]:.17g}"); out_rows.append(r2)
    os.makedirs(os.path.join(a.out_scene, "train"), exist_ok=True); os.makedirs(os.path.join(a.out_scene, "test"), exist_ok=True)
    for src, dst in ((os.path.join(a.scene_dir, "train", "images"), os.path.join(a.out_scene, "train", "images")), (os.path.join(a.scene_dir, "test", "images"), os.path.join(a.out_scene, "test", "images"))):
        if not os.path.lexists(dst): os.symlink(os.path.abspath(src), dst)
    sp = os.path.join(a.out_scene, "train", "sparse"); shutil.rmtree(sp, ignore_errors=True); shutil.copytree(os.path.join(a.ba_dir, "sparse"), sp)
    with open(os.path.join(a.out_scene, "test", "test_poses.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in out_rows]
    shifts = np.array(shifts); print(f"[build_scene] {len(rows)} view test: xoay tương đương med {np.median(shifts):.1f} px p90 {np.percentile(shifts, 90):.1f} px | láng giềng med {int(np.median(nnb))} | 0 láng giềng: {int((np.array(nnb) == 0).sum())}")
    print("Y30_TRANSFER_DONE", flush=True)
if __name__ == "__main__": main()
