"""probe_data.py — ba phép đo quyết định recipe, chạy trong 2 giờ đầu ngày thi.

CPU thuần, không đụng GPU, không đọc một pixel test nào. Chạy song song với
baseline đang train. Mỗi phép đo trả về một QUYẾT ĐỊNH, không phải một con số đẹp.

  python src/probe_data.py --scene $VT_SCENE --out $VT_RUNS/probe.json

Ba câu, và sai câu nào thì mất đúng đòn tương ứng:
  1. Camera còn méo không, tâm quang test có bị ép về W/2,H/2 không
     → quyết --test_use_train_K và có cần dựng lại SfM không
  2. points3D có track 2D thật không
     → quyết --depth_weight (track rỗng mà vẫn bật = null-test, đã dính một lần)
  3. Test xen kẽ trong chuyến bay hay ngoại suy theo vùng
     → quyết có dùng refiner láng giềng (warp) hay không. Ngoại suy ⇒ refiner teo.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa
from colmap_io import qvec2rotmat, read_sparse  # noqa


def cam_center(qvec, tvec):
    R = qvec2rotmat(np.asarray(qvec, dtype=np.float64))
    return -R.T @ np.asarray(tvec, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    sparse = os.path.join(a.scene, "train", "sparse", "0")
    cams, images, points = read_sparse(sparse)
    csv_path = os.path.join(a.scene, "test", "test_poses.csv")
    tests = list(csv.DictReader(open(csv_path)))
    rep = {}

    # BẪY: sparse của BTC có thể chứa CẢ pose test (vòng trước: 506 = 404 train + 102 test).
    # Nếu không tách ra, "khoảng cách test→train gần nhất" bằng 0 và mọi chẩn đoán
    # về kiểu chia test đều vô nghĩa. Train thật = ảnh có file trên đĩa.
    img_dir = os.path.join(a.scene, "train", "images")
    on_disk = set(os.listdir(img_dir)) if os.path.isdir(img_dir) else set()
    test_stems = {os.path.splitext(r["image_name"])[0] for r in tests}
    train_imgs = {i: im for i, im in images.items()
                  if (im.name in on_disk if on_disk
                      else os.path.splitext(os.path.basename(im.name))[0] not in test_stems)}
    n_extra = len(images) - len(train_imgs)

    # ── 1. CAMERA ────────────────────────────────────────────────────────────
    cam = next(iter(cams.values()))
    model, params = cam.model, np.asarray(cam.params, dtype=float)
    W, H = int(cam.width), int(cam.height)
    K = cam.K()
    cx, cy = K[0, 2], K[1, 2]
    dist = np.asarray(cam.dist_coeffs(), dtype=float).ravel()

    t0 = tests[0]
    tcx, tcy = float(t0["cx"]), float(t0["cy"])
    pp_forced = abs(tcx - W / 2) < 1e-6 and abs(tcy - H / 2) < 1e-6
    pp_shift = float(np.hypot(tcx - cx, tcy - cy))

    rep["camera"] = dict(
        model=str(model), width=W, height=H,
        n_poses_in_sparse=len(images), n_train_images=len(train_imgs),
        n_test_poses_also_in_sparse=n_extra, n_test=len(tests), dist_params=dist.round(6).tolist(),
        has_distortion=bool(dist.size and np.abs(dist).max() > 1e-6),
        train_pp=[round(float(cx), 2), round(float(cy), 2)],
        test_pp=[tcx, tcy], test_pp_forced_to_centre=pp_forced,
        test_vs_train_pp_shift_px=round(pp_shift, 3),
    )
    d1 = []
    if rep["camera"]["has_distortion"]:
        d1.append("ẢNH CÒN MÉO ⇒ trục prewarp/redistort SỐNG. Bắt buộc chạy giai đoạn 0.")
    else:
        d1.append("k≈0, ảnh đã khử méo ⇒ không cần prewarp ở khâu nạp.")
    if pp_shift > 2:
        d1.append(f"tâm quang test lệch train {pp_shift:.1f} px ⇒ BẮT BUỘC --test_use_train_K 1.")
    else:
        d1.append(f"tâm quang test lệch {pp_shift:.2f} px ⇒ --test_use_train_K vô hại.")
    if n_extra > 0:
        d1.append(f"sparse chứa THÊM {n_extra} pose không có ảnh — gần chắc là pose test. "
                  "Có track 2D của test ⇒ dùng được cho PnP khi chuyển pose (giai đoạn 0), "
                  "nhưng KHÔNG BAO GIỜ đưa vào tập train.")
    rep["camera"]["QUYET_DINH"] = d1

    # ── 2. TRACK 2D–3D ───────────────────────────────────────────────────────
    per_img = np.array([int((np.asarray(im.point3D_ids) >= 0).sum())
                        for im in train_imgs.values()])
    med = float(np.median(per_img))
    rep["tracks"] = dict(
        n_points3D=int(len(points.ids)), median_tracks_per_image=med,
        frac_images_under_400=round(float((per_img < 400).mean()), 3),
        n_images_zero_tracks=int((per_img == 0).sum()),
    )
    rep["tracks"]["QUYET_DINH"] = [
        "có track thật ⇒ giữ --depth_weight 0.05."
        if med >= 200 else
        f"track quá thưa (median {med:.0f}) ⇒ TẮT TAY --depth_weight 0, "
        "nếu không loss depth chạy trên track rỗng = null-test."
    ]

    # ── 3. KIỂU CHIA TEST ────────────────────────────────────────────────────
    tr_c = np.array([im.center() for im in train_imgs.values()])
    te_c = np.array([cam_center(
        [float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])],
        [float(r["tx"]), float(r["ty"]), float(r["tz"])]) for r in tests])
    # bước bay điển hình = khoảng cách trung vị giữa hai train gần nhau nhất
    dtr = np.linalg.norm(tr_c[:, None] - tr_c[None], axis=-1)
    np.fill_diagonal(dtr, np.inf)
    step = float(np.median(dtr.min(1)))
    d_te = np.linalg.norm(te_c[:, None] - tr_c[None], axis=-1)
    nn = np.sort(d_te, axis=1)
    ratio = nn[:, 0] / max(step, 1e-9)
    interleaved = float((ratio < 1.5).mean())

    rep["test_split"] = dict(
        train_flight_step=round(step, 4),
        test_to_nearest_train_median=round(float(np.median(nn[:, 0])), 4),
        test_to_nearest_train_p90=round(float(np.percentile(nn[:, 0], 90)), 4),
        ratio_to_flight_step_median=round(float(np.median(ratio)), 3),
        frac_test_inside_flight=round(interleaved, 3),
        median_n_train_within_1_step=float(np.median((d_te < step * 1.5).sum(1))),
    )
    if interleaved >= 0.7:
        d3 = (f"{interleaved:.0%} view test nằm XEN KẼ trong chuyến bay ⇒ luôn có láng giềng "
              "để warp. Refiner láng giềng ĂN ĐẬM — đây là +6..9 điểm, ưu tiên số 1.")
    elif interleaved >= 0.3:
        d3 = (f"chỉ {interleaved:.0%} xen kẽ ⇒ refiner ăn một phần. Chạy nhưng đo A/B "
              "trên holdout trước khi dồn giờ.")
    else:
        d3 = (f"chỉ {interleaved:.0%} xen kẽ ⇒ TEST LÀ NGOẠI SUY. Refiner láng giềng TEO. "
              "Dồn giờ vào dung lượng 3DGS + prior hình học, đừng đầu tư warp.")
    rep["test_split"]["QUYET_DINH"] = [d3]

    # ── in ra ────────────────────────────────────────────────────────────────
    print("=" * 78)
    for sec in ("camera", "tracks", "test_split"):
        print(f"\n── {sec.upper()} " + "─" * (72 - len(sec)))
        for k, v in rep[sec].items():
            if k != "QUYET_DINH":
                print(f"   {k:38} {v}")
        for line in rep[sec]["QUYET_DINH"]:
            print(f"   ➜ {line}")
    print("\n" + "=" * 78)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(rep, open(a.out, "w"), indent=2, ensure_ascii=False)
        print(f"[ghi] {a.out}")


if __name__ == "__main__":
    main()
