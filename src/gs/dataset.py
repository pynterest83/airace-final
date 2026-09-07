"""Scene loading: COLMAP parsing, undistortion, sparse depths, test poses."""
import csv
import os
import re
from collections import Counter
from dataclasses import dataclass

import cv2
# 05/09 y67/y68: undistort ảnh train — lanczos sắc hơn linear trên GT hoàn hảo (98,5 vs 93 điểm) nhưng train lại (y68 C) thô 53,38 vs 53,46, +refiner 60,13 vs 60,19
# = ±0 (nhiễu seed) → giữ linear cho khớp bit-for-bit đường cũ; VT_UNDISTORT_INTERP=lanczos để thử lại
_UNDISTORT_INTERP = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}[os.environ.get("VT_UNDISTORT_INTERP", "linear")]
import numpy as np


def frame_index(name):
    """Capture-order index, or None if the name carries no ordering.

    DJI drone: DJI_20241227155343_0023_V.JPG -> 23.
    Video frames: frame_001305.jpg -> 1305. Without this second pattern the
    chair/bonsai scenes returned None, and nearest_train_views SILENTLY fell
    back to spatial — so every "temporal" measurement on chair was really a
    spatial one wearing its name (caught 22/07: chair's tblend2/4/8 scores were
    bit-identical to job46's blend2/4/8).
    """
    m = re.search(r"_(\d{3,6})_V\.", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{3,8})\.[A-Za-z]+$", name)
    return int(m.group(1)) if m else None

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa
from colmap_io import qvec2rotmat, read_sparse


def rotmat2qvec(R):
    """ma trận quay → quaternion (w,x,y,z) COLMAP."""
    q = np.empty(4); t = np.trace(R)
    if t > 0:
        s_ = np.sqrt(t + 1.0) * 2; q[:] = [0.25 * s_, (R[2, 1] - R[1, 2]) / s_, (R[0, 2] - R[2, 0]) / s_, (R[1, 0] - R[0, 1]) / s_]
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s_ = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2; q[0] = (R[k, j] - R[j, k]) / s_
        v = np.zeros(3); v[i] = 0.25 * s_; v[j] = (R[j, i] + R[i, j]) / s_; v[k] = (R[k, i] + R[i, k]) / s_; q[1:] = v
    return q / np.linalg.norm(q)


@dataclass
class TestPose:
    image_name: str
    w2c: np.ndarray  # 4x4
    K: np.ndarray  # 3x3
    width: int
    height: int


def load_test_poses(csv_path):
    poses = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            q = np.array([float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])])
            t = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])])
            m = np.eye(4)
            m[:3, :3] = qvec2rotmat(q)
            m[:3, 3] = t
            K = np.array([
                [float(row["fx"]), 0, float(row["cx"])],
                [0, float(row["fy"]), float(row["cy"])],
                [0, 0, 1],
            ])
            poses.append(TestPose(row["image_name"], m, K, int(row["width"]), int(row["height"])))
    return poses


class SceneData:
    """Loads one scene: undistorted train images, poses, sparse points, test poses.

    All images share a single SIMPLE_RADIAL camera. Images are undistorted to a
    pinhole camera with the same K (f, cx, cy) and full canvas, matching the
    intrinsics given in test_poses.csv.
    """

    # -- cameras ---------------------------------------------------------
    #
    # Round-1 BTC data ships exactly one camera per scene, so this used to be a
    # bare `assert len(cameras) == 1`.  Large-scene exports break that in two
    # different ways, and only one of them is a real multi-camera scene:
    #
    #   * ContextCapture / GauU-Scene writes ONE camera ENTRY PER IMAGE, one per
    #     photogroup.  SMBU has 829 entries in 9 groups whose intrinsics differ
    #     by at most 0.85 px in f and 2.81 px in cy — the same physical camera
    #     re-calibrated per flight, not nine cameras.  Measured across all seven
    #     GauU-Scene scenes at 5472x3648 the worst spread is LFLS: df 3.64,
    #     dcx 2.64, dcy 7.39 px.  For scale: BTC's own test_poses.csv convention
    #     throws away a 20/27 px principal-point offset, so folding these
    #     together is 3-4x SMALLER than an error the competition format already
    #     bakes in.  Hence a tolerance, not exact equality.
    #   * A scene stitched from genuinely different cameras (different model,
    #     size, or intrinsics far outside the tolerance) is a different story.
    #     The whole pipeline (trainer, renderers, 13 other files) broadcasts a
    #     single `scene.K`, so that case cannot be handled here by quietly
    #     picking one — it has to be a loud decision.
    #
    # The honest cost of collapsing: a view whose true f is off by df from the
    # representative gets a systematic radial reprojection error of
    # df/f * r, i.e. ~0.7 px at the corner for LFLS at data_factor 4.  Sub-pixel,
    # partly absorbable by the SE(3) pose refinement, and printed every run so it
    # is never silently forgotten.  The real fix is per-image intrinsics; that is
    # a 14-file change and does not belong in a data loader.
    CAM_TOL_PX = 3.0

    @staticmethod
    def _cam_key(c):
        return (c.model, int(c.width), int(c.height), tuple(float(p) for p in c.params))

    @staticmethod
    def _cam_dev(c, ref):
        """max |delta| in pixels between two cameras' intrinsics, or None if they
        are not even comparable (different model/size/distortion)."""
        if c.model != ref.model or c.width != ref.width or c.height != ref.height:
            return None
        n_intr = 2 if c.model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE") else 1
        if any(abs(c.params[i] - ref.params[i]) > 1e-9
               for i in range(n_intr + 2, len(ref.params))):
            return None  # distortion differs -> not the same camera, no tolerance
        return max(abs(c.params[i] - ref.params[i]) for i in range(n_intr + 2))

    def _resolve_camera(self, metas, policy, tol_px=None):
        """-> (representative camera, set of camera ids whose images to drop).

        policy: "auto" collapses entries agreeing within tol_px, drops a stray
        group only if it is under 2% of views, and otherwise raises with the
        numbers in hand; "dominant" always keeps the largest group; "error" is
        the old exact-match assert.
        """
        tol = self.CAM_TOL_PX if tol_px is None else float(tol_px)
        used = [m.camera_id for m in metas] or list(self.cameras)
        groups = {}
        for cid in used:
            groups.setdefault(self._cam_key(self.cameras[cid]), []).append(cid)

        counts = sorted(((len(v), k, v) for k, v in groups.items()), reverse=True)

        if len(groups) == 1:
            ids = counts[0][2]
            if len(set(ids)) > 1:
                print(f"[camera] {len(set(ids))} camera entries, all identical "
                      f"-> collapsed to one ({self.cameras[ids[0]].model} "
                      f"{self.cameras[ids[0]].width}x{self.cameras[ids[0]].height})")
            return self.cameras[ids[0]], set()

        # more than one distinct entry: are they the same camera within tol_px?
        if policy != "error":
            reps = [self.cameras[v[0]] for _n, _k, v in counts]
            # Pick the group that MINIMISES the worst deviation to every other
            # group (a 1-centre), not the most-populous one: the most-populous
            # group can sit at one end of the spread and double the error we
            # then bake into every view.  <=13 groups, so O(n^2) is free.
            best, best_dev = None, None
            for r in reps:
                ds = [self._cam_dev(c, r) for c in reps]
                if any(d is None for d in ds):
                    continue
                m = max(ds)
                if best_dev is None or m < best_dev:
                    best, best_dev = r, m
            if best is not None and best_dev <= tol:
                px = best_dev / best.params[0] * max(best.width, best.height) / 2
                print(f"[camera] {len(set(used))} camera entries in {len(groups)} groups, "
                      f"agreeing within {best_dev:.3f} px (tol {tol:.1f}) -> collapsed to the "
                      f"most central one (cam {best.id}). Systematic reprojection error from "
                      f"this is <= {px:.2f} px at the image corner.")
                return best, set()

        desc = "; ".join(f"{n} views: {self.cameras[v[0]].model} "
                         f"{self.cameras[v[0]].width}x{self.cameras[v[0]].height} "
                         f"params={[round(float(p), 4) for p in self.cameras[v[0]].params]}"
                         for n, _k, v in counts)
        if policy == "error":
            raise AssertionError(f"expected single shared camera, got {len(groups)}: {desc}")
        n_top = counts[0][0]
        minority = len(used) - n_top
        if policy == "auto" and minority > 0.02 * len(used):
            raise AssertionError(
                f"{len(groups)} distinct cameras in {self.scene_dir}, differing by more "
                f"than {tol:.1f} px: {desc}. The pipeline broadcasts a single K, so this "
                f"needs a decision: raise --camera_tol_px if they are the same physical "
                f"camera re-calibrated per flight, pass multi_camera='dominant' to train "
                f"on the {n_top} views of the largest camera and drop the other "
                f"{minority}, or split the scene per camera.")
        drop = {cid for _n, _k, v in counts[1:] for cid in v}
        print(f"[camera] {len(groups)} distinct cameras -> keeping the largest "
              f"({n_top} views), dropping {minority}. Groups: {desc}")
        return self.cameras[counts[0][2][0]], drop

    def __init__(self, scene_dir, load_images=True, distorted=False,
                 mask_dir=None, holdout_every=0, holdout_offset=None,
                 multi_camera="auto", camera_tol_px=None, train_list=None):
        """distorted=True: keep original (distorted) images; no undistortion.
        Use with a rasterizer that models distortion directly (3DGUT).

        multi_camera: what to do when sparse/0 lists more than one camera —
        "auto" (default), "dominant" or "error".  See _resolve_camera.
        camera_tol_px: how far intrinsics may differ and still count as the same
        camera (default CAM_TOL_PX = 3.0 px).

        mask_dir: root containing <scene_name>/<image_name>.png transient
        masks (255 = exclude from loss), in the original distorted frame.
        holdout_every=N: hold out every Nth train view (by sorted name) as a
        pseudo-test set — excluded from training, poses/GT kept for eval.
        holdout_offset selects the residue class; None keeps the historical
        N//2 split.  A second locked offset is needed to detect split tuning.
        """
        self.scene_dir = scene_dir
        self.distorted = distorted
        sparse = os.path.join(scene_dir, "train", "sparse", "0")
        # read_sparse tự nhận .bin hay .txt — đừng đổi lại thành read_*_bin. COLMAP xuất
        # cả hai dạng và GauU-Scene/ContextCapture ship .txt; trước 15/08 chỗ này cứng hoá
        # .bin nên một scene .txt sẽ chết ở đây, SAU khi mọi công cụ soi scene đã báo OK.
        self.cameras, self.images_meta, self.points = read_sparse(sparse)
        # 03/09 (x73): ghi đè pose theo tên ảnh từ JSON {name: w2c 4x4} — tái định vị ảnh train pose yếu
        # (reloc_weak.py) trong CHÍNH khung BTC. Bật bằng env VT_POSE_OVERRIDE=<json>; test pose không đổi.
        _po = os.environ.get("VT_POSE_OVERRIDE", "")
        if _po:
            import json as _json
            _ov = _json.load(open(_po)); _n = 0
            for _m in self.images_meta.values():
                if _m.name in _ov:
                    _w = np.asarray(_ov[_m.name]["w2c_new"] if isinstance(_ov[_m.name], dict) else _ov[_m.name], dtype=np.float64)
                    _m.qvec = rotmat2qvec(_w[:3, :3]); _m.tvec = _w[:3, 3].copy(); _n += 1
            print(f"[pose_override] {_po}: ghi đè {_n}/{len(_ov)} pose", flush=True)

        # keep only images that exist on disk in train/images
        img_dir = os.path.join(scene_dir, "train", "images")
        on_disk = set(os.listdir(img_dir))
        metas = [m for m in self.images_meta.values() if m.name in on_disk]
        metas.sort(key=lambda m: m.name)

        self.camera, dropped_cam_ids = self._resolve_camera(metas, multi_camera, camera_tol_px)
        self.K = self.camera.K().astype(np.float64)
        self.dist = self.camera.dist_coeffs()
        self.width, self.height = self.camera.width, self.camera.height
        if dropped_cam_ids:
            metas = [m for m in metas if m.camera_id not in dropped_cam_ids]
        missing = on_disk - {m.name for m in metas}
        if missing:
            print(f"[warn] {len(missing)} train images not in images.bin (skipped): {sorted(missing)[:5]}")

        # train_list: file 1 tên ảnh/dòng — giới hạn train về đúng tập đó (chuyên gia
        # cục bộ per-cluster, §19.47 E4). Áp TRƯỚC holdout; mọi chỉ số per-view phía sau
        # (pose_opt, bilagrid, sparse_uvd) đều theo danh sách ĐÃ LỌC, nên chỉ dùng cùng
        # ckpt splats thuần (ft) — không nạp ckpt có state per-view của danh sách đầy đủ.
        if train_list:
            with open(train_list) as f:
                req = [ln.strip() for ln in f if ln.strip()]
            cnt = Counter(req)
            before = len(metas)
            metas = [m for m in metas if m.name in cnt]
            # Tên lặp k lần trong file → view vào dataset k lần (oversample
            # per-view — orphan rescue §19.52 E1). Bản sao đứng CUỐI danh sách,
            # mỗi bản có pose_opt/bilagrid index riêng ⇒ chỉ dùng với ckpt
            # splats thuần (ft), như ràng buộc sẵn có của train_list ở trên.
            dups = [m for m in metas for _ in range(cnt[m.name] - 1)]
            metas = metas + dups
            print(f"[train_list] giữ {len(metas) - len(dups)}/{before} train view"
                  f" + {len(dups)} bản sao oversample "
                  f"({os.path.basename(train_list)}; yêu cầu {len(cnt)} tên)")
            if not metas:
                raise SystemExit("train_list lọc hết sạch view — kiểm tra tên ảnh")

        self.holdout_metas = []
        self.n_total_views = len(metas)  # before holdout removal
        orig_idx = list(range(len(metas)))
        if holdout_every and holdout_every > 0:
            holdout_residue = (holdout_every // 2 if holdout_offset is None
                               else int(holdout_offset) % holdout_every)
            keep, keep_idx = [], []
            for i, m in enumerate(metas):
                if i % holdout_every == holdout_residue:
                    self.holdout_metas.append(m)
                else:
                    keep.append(m)
                    keep_idx.append(i)
            metas, orig_idx = keep, keep_idx
            print(f"[holdout] {len(self.holdout_metas)} views held out "
                  f"(every={holdout_every}, offset={holdout_residue}), "
                  f"{len(metas)} remain for training")
        self.train_metas = metas
        # position of each train view in the full sorted list — bilateral grids
        # are always sized/indexed by the full list so checkpoints stay
        # compatible regardless of the holdout setting
        self.train_orig_idx = np.array(orig_idx, dtype=np.int64)

        # undistortion maps (undistorted target -> distorted source)
        self.need_undistort = (not distorted) and np.abs(self.dist).max() > 1e-12
        if self.need_undistort:
            self.umap1, self.umap2 = cv2.initUndistortRectifyMap(
                self.K, np.asarray(self.dist, dtype=np.float64),
                None, self.K, (self.width, self.height), cv2.CV_32FC1)

        # w2c matrices, camera centers
        self.w2c = np.stack([m.w2c() for m in metas])  # (N,4,4)
        self.centers = np.stack([m.center() for m in metas])  # (N,3)

        # scene scale (gsplat convention)
        center = self.centers.mean(0)
        self.scene_scale = float(np.max(np.linalg.norm(self.centers - center, axis=1))) * 1.1

        # sparse depths: project each image's 3D track points with pinhole K
        self.sparse_uvd = []
        for m in metas:
            pids = m.point3D_ids[m.point3D_ids >= 0]
            rows = [self.points.id_to_row[int(p)] for p in pids if int(p) in self.points.id_to_row]
            if not rows:
                self.sparse_uvd.append(np.zeros((0, 3), dtype=np.float32))
                continue
            xyz = self.points.xyz[rows]
            cam = (m.R() @ xyz.T).T + m.tvec
            z = cam[:, 2]
            ok = z > 1e-6
            xy = cam[ok, :2] / z[ok, None]
            if distorted and np.abs(self.dist).max() > 1e-12:
                r2 = (xy**2).sum(axis=1, keepdims=True)
                xy = xy * (1.0 + self.dist[0] * r2)
            uv = xy * self.K[0, 0]
            uv[:, 0] += self.K[0, 2]
            uv[:, 1] += self.K[1, 2]
            inb = (uv[:, 0] >= 0) & (uv[:, 0] < self.width) & (uv[:, 1] >= 0) & (uv[:, 1] < self.height)
            self.sparse_uvd.append(
                np.concatenate([uv[inb], z[ok][inb, None]], axis=1).astype(np.float32))

        # test poses
        tp = os.path.join(scene_dir, "test", "test_poses.csv")
        self.test_poses = load_test_poses(tp) if os.path.exists(tp) else []

        self.images = None
        self.masks = [None] * len(metas)
        self.holdout_images = []
        if load_images:
            # Every train view is decoded and kept in RAM as uint8 RGB.  At the
            # 1320x989 of round 1 that is 3.8 MB/view and nobody ever noticed;
            # at the 5472x3648 of an aerial capture it is 60 MB/view, so 829
            # views want ~50 GB.  Say so before the OOM, not after.
            need_gb = (len(metas) + len(self.holdout_metas)) * self.width * self.height * 3 / 2**30
            if need_gb > 8:
                print(f"[mem] {len(metas)}+{len(self.holdout_metas)} views at "
                      f"{self.width}x{self.height} = ~{need_gb:.1f} GB of RAM for images alone."
                      + (" Prepare the scene with a downscale factor instead."
                         if need_gb > 24 else ""))
            self.images = []
            for m in metas:
                img = cv2.imread(os.path.join(img_dir, m.name), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.need_undistort:
                    img = cv2.remap(img, self.umap1, self.umap2,
                                    interpolation=_UNDISTORT_INTERP,
                                    borderMode=cv2.BORDER_REPLICATE)
                self.images.append(img)
            if mask_dir:
                mdir = os.path.join(
                    mask_dir, os.path.basename(os.path.normpath(scene_dir)))
                n_found = 0
                for i, m in enumerate(metas):
                    mp = os.path.join(mdir, m.name + ".png")
                    if not os.path.exists(mp):
                        continue
                    mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                    if self.need_undistort:
                        # masks are stored in the distorted frame
                        mk = cv2.remap(mk, self.umap1, self.umap2,
                                       interpolation=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT)
                    self.masks[i] = (mk > 127).astype(np.float32)
                    n_found += 1
                print(f"[masks] {n_found}/{len(metas)} train views have transient masks")
            for m in self.holdout_metas:
                # holdout GT stays in the ORIGINAL (distorted) frame
                img = cv2.imread(os.path.join(img_dir, m.name), cv2.IMREAD_COLOR)
                self.holdout_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # -- helpers ---------------------------------------------------------
    def init_points(self):
        """(xyz, rgb01) for gaussian init."""
        return self.points.xyz.astype(np.float32), (self.points.rgb.astype(np.float32) / 255.0)

    def nearest_train_views(self, test_pose: TestPose, k=3, mode="spatial",
                            p=1.0):
        """Indices + weights of k nearest train views.

        mode="spatial": by camera-center distance.
        mode="temporal": by DJI frame-number distance (test frames interleave
        train frames of the same flight, so exposure varies with time, not
        position). Falls back to spatial if frame numbers can't be parsed.

        p: weight falloff exponent, w ∝ 1/d^p. p=0 is a plain average of the k
        neighbours; large p approaches nearest-neighbour. k is a HARD cutoff and
        p is the soft one, so they trade off — job46 tuned k with p pinned at 1
        and never checked whether the cutoff was doing the work or the falloff.
        """
        if mode == "temporal":
            ti = frame_index(test_pose.image_name)
            tidx = [frame_index(m.name) for m in self.train_metas]
            if ti is not None and all(x is not None for x in tidx):
                d = np.abs(np.array(tidx, dtype=np.float64) - ti)
            else:
                # falling back quietly makes a temporal experiment report a
                # spatial number under a temporal name — say so, once
                if not getattr(self, "_warned_temporal", False):
                    self._warned_temporal = True
                    print("[warn] temporal mode requested but frame numbers are "
                          "unparseable -> FALLING BACK TO SPATIAL; any 'tblend' "
                          "score from this run is a spatial score", flush=True)
                mode = "spatial"
        if mode == "spatial":
            c = -test_pose.w2c[:3, :3].T @ test_pose.w2c[:3, 3]
            d = np.linalg.norm(self.centers - c, axis=1)
        idx = np.argsort(d)[:k]
        if p == 0.0:
            w = np.ones(len(idx), dtype=np.float64)
        else:
            w = 1.0 / (d[idx] + 1e-8) ** p
        return idx, (w / w.sum())

    def redistort_map(self, K, width, height):
        """Maps for warping an undistorted render back to the distorted original frame.

        For each pixel of the (distorted) output, gives sampling coords in the
        undistorted render.
        """
        u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        pts = np.stack([u.ravel(), v.ravel()], axis=1)[:, None, :]
        # truyền NGUYÊN vector: FULL_OPENCV có 8 hệ số, cắt còn 4 là vứt k3..k6.
        dist = np.asarray(self.dist, dtype=np.float64)
        und = cv2.undistortPoints(pts, K, dist, P=K).reshape(height, width, 2)
        return und[..., 0].astype(np.float32), und[..., 1].astype(np.float32)
