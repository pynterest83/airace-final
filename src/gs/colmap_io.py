"""Minimal COLMAP model reader — nhị phân (.bin) và văn bản (.txt).

`.txt` KHÔNG phải trường hợp giả định: GauU-Scene V2.1 (xuất từ ContextCapture) ship
đúng dạng đó, và COLMAP thì xuất cả hai tuỳ cờ. Trước 15/08 chỉ `colmap_bin.py` (công cụ
soi scene) đọc được `.txt`, còn `dataset.py` — tức ĐƯỜNG TRAIN — cứng hoá `.bin`. Nghĩa
là ta có thể soi một scene rất đẹp rồi mới chết lúc bắt đầu train. Dùng `read_sparse()`
để khỏi phải nhớ điều này.
"""
import os
import struct
from dataclasses import dataclass, field

import numpy as np

CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # model-specific

    def K(self) -> np.ndarray:
        """3x3 intrinsics (ignoring distortion)."""
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            f, cx, cy = self.params[0], self.params[1], self.params[2]
            fx = fy = f
        else:  # PINHOLE, OPENCV, ...
            fx, fy, cx, cy = self.params[0], self.params[1], self.params[2], self.params[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def dist_coeffs(self) -> np.ndarray:
        """Hệ số méo kiểu OpenCV: (k1,k2,p1,p2) — hoặc 8 phần tử cho FULL_OPENCV.

        cv2.undistort/initUndistortRectifyMap nhận vector 4, 5, 8, 12 hoặc 14 phần tử, nên
        trả về 8 cho FULL_OPENCV là ĐÚNG chứ không phải mẹo — cắt còn 4 sẽ vứt lặng lẽ
        k3..k6 và làm hỏng ảnh ở rìa khung đúng chỗ ta đang thiếu tần số cao (§16).
        """
        if self.model in ("SIMPLE_PINHOLE", "PINHOLE"):
            return np.zeros(4, dtype=np.float64)
        if self.model == "SIMPLE_RADIAL":
            return np.array([self.params[3], 0.0, 0.0, 0.0], dtype=np.float64)
        if self.model == "RADIAL":
            return np.array([self.params[3], self.params[4], 0.0, 0.0], dtype=np.float64)
        if self.model == "OPENCV":
            return np.asarray(self.params[4:8], dtype=np.float64)
        if self.model == "FULL_OPENCV":
            # fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6 -> (k1,k2,p1,p2,k3,k4,k5,k6)
            return np.asarray(self.params[4:12], dtype=np.float64)
        if self.model == "FOV":
            # params[4] = omega, mô hình FOV không quy về được vector OpenCV.
            raise NotImplementedError(
                "camera FOV: không quy được về hệ số OpenCV. Cách xử lý ngày thi: chạy "
                "`colmap image_undistorter` để BTC-data thành PINHOLE rồi train trên bản đó."
            )
        if "FISHEYE" in self.model:
            raise NotImplementedError(
                f"camera {self.model}: fisheye cần cv2.fisheye.* chứ không phải cv2.undistort, "
                "coi như pinhole sẽ SAI RẤT NẶNG ở rìa. Cách xử lý ngày thi: "
                "`colmap image_undistorter` -> PINHOLE rồi train trên bản đó."
            )
        raise NotImplementedError(f"chưa hỗ trợ méo cho model {self.model}")


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # (w,x,y,z), world-to-camera rotation
    tvec: np.ndarray  # world-to-camera translation
    camera_id: int
    name: str
    xys: np.ndarray  # (N,2) 2D keypoints (distorted pixel coords)
    point3D_ids: np.ndarray  # (N,), -1 if unmatched

    def R(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)

    def w2c(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.R()
        m[:3, 3] = self.tvec
        return m

    def center(self) -> np.ndarray:
        return -self.R().T @ self.tvec


@dataclass
class Points3D:
    ids: np.ndarray  # (M,)
    xyz: np.ndarray  # (M,3)
    rgb: np.ndarray  # (M,3) uint8
    error: np.ndarray  # (M,)
    id_to_row: dict = field(default_factory=dict)


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def _read(f, fmt):
    sz = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(sz))


def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "<Q")
        for _ in range(n):
            cid, model_id, w, h = _read(f, "<iiQQ")
            name, num_params = CAMERA_MODELS[model_id]
            params = np.array(_read(f, "<" + "d" * num_params))
            cams[cid] = Camera(cid, name, int(w), int(h), params)
    return cams


def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "<Q")
        for _ in range(n):
            iid = _read(f, "<i")[0]
            qvec = np.array(_read(f, "<dddd"))
            tvec = np.array(_read(f, "<ddd"))
            cam_id = _read(f, "<i")[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (npts,) = _read(f, "<Q")
            data = np.frombuffer(f.read(24 * npts), dtype=np.float64).reshape(npts, 3)
            xys = data[:, :2].copy()
            p3d = data[:, 2].copy().view(np.int64)
            images[iid] = Image(iid, qvec, tvec, cam_id, name.decode(), xys, p3d)
    return images


def write_images_bin(path, images):
    """Write a COLMAP images.bin file from an iterable of Image records."""
    records = list(images)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(records)))
        for image in records:
            f.write(struct.pack("<i", int(image.id)))
            f.write(struct.pack("<dddd", *map(float, image.qvec)))
            f.write(struct.pack("<ddd", *map(float, image.tvec)))
            f.write(struct.pack("<i", int(image.camera_id)))
            f.write(image.name.encode("utf-8") + b"\x00")
            if len(image.xys) != len(image.point3D_ids):
                raise ValueError(
                    f"image {image.name}: xys/point3D_ids length mismatch"
                )
            f.write(struct.pack("<Q", len(image.xys)))
            for (x, y), point_id in zip(image.xys, image.point3D_ids):
                f.write(
                    struct.pack(
                        "<ddq", float(x), float(y), int(point_id)
                    )
                )


def read_points3d_bin(path):
    with open(path, "rb") as f:
        (n,) = _read(f, "<Q")
        ids = np.empty(n, dtype=np.int64)
        xyz = np.empty((n, 3), dtype=np.float64)
        rgb = np.empty((n, 3), dtype=np.uint8)
        err = np.empty(n, dtype=np.float64)
        for i in range(n):
            (pid,) = _read(f, "<Q")
            ids[i] = pid
            xyz[i] = _read(f, "<ddd")
            rgb[i] = _read(f, "<BBB")
            err[i] = _read(f, "<d")[0]
            (tlen,) = _read(f, "<Q")
            f.seek(8 * tlen, 1)  # skip track
    pts = Points3D(ids, xyz, rgb, err)
    pts.id_to_row = {int(p): i for i, p in enumerate(ids)}
    return pts


# --------------------------------------------------------------- văn bản (.txt) ---
# Ba hàm dưới đọc đúng ba file COLMAP dạng text. Quy ước: dòng bắt đầu bằng '#' là chú
# thích, và trong images.txt MỖI ẢNH CHIẾM HAI DÒNG — dòng pose rồi dòng điểm 2D. Dòng
# điểm 2D có thể RỖNG (ContextCapture xuất vậy: có pose, không có track), nên không được
# giả định cứ hai dòng không rỗng là một ảnh.

def _txt_rows(path):
    """Sinh từng dòng đã bỏ chú thích, GIỮ NGUYÊN dòng rỗng (images.txt cần)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                yield line


def read_cameras_txt(path):
    """CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]"""
    cams = {}
    for line in _txt_rows(path):
        if not line:
            continue
        t = line.split()
        cid, model, w, h = int(t[0]), t[1], int(t[2]), int(t[3])
        cams[cid] = Camera(cid, model, w, h, np.array([float(v) for v in t[4:]]))
    return cams


def read_images_txt(path):
    """Hai dòng một ảnh: pose, rồi điểm 2D (có thể rỗng)."""
    images = {}
    rows = list(_txt_rows(path))
    i = 0
    while i < len(rows):
        if not rows[i]:
            i += 1
            continue
        t = rows[i].split()
        iid = int(t[0])
        qvec = np.array([float(v) for v in t[1:5]])
        tvec = np.array([float(v) for v in t[5:8]])
        cam_id = int(t[8])
        # Tên file có thể chứa dấu cách -> ghép lại phần đuôi, đừng lấy t[9].
        name = " ".join(t[9:])
        pts_line = rows[i + 1] if i + 1 < len(rows) else ""
        if pts_line:
            v = pts_line.split()
            # bộ ba (X, Y, POINT3D_ID)
            xys = np.array([[float(v[k]), float(v[k + 1])] for k in range(0, len(v), 3)])
            p3d = np.array([int(v[k + 2]) for k in range(0, len(v), 3)], dtype=np.int64)
        else:
            xys = np.zeros((0, 2), dtype=np.float64)
            p3d = np.zeros((0,), dtype=np.int64)
        images[iid] = Image(iid, qvec, tvec, cam_id, name, xys, p3d)
        i += 2
    return images


def read_points3d_txt(path):
    """POINT3D_ID X Y Z R G B ERROR TRACK[]"""
    ids, xyz, rgb, err = [], [], [], []
    for line in _txt_rows(path):
        if not line:
            continue
        t = line.split()
        ids.append(int(t[0]))
        xyz.append([float(t[1]), float(t[2]), float(t[3])])
        rgb.append([int(t[4]), int(t[5]), int(t[6])])
        err.append(float(t[7]))
    pts = Points3D(
        np.array(ids, dtype=np.int64),
        np.array(xyz, dtype=np.float64).reshape(-1, 3),
        np.array(rgb, dtype=np.uint8).reshape(-1, 3),
        np.array(err, dtype=np.float64),
    )
    pts.id_to_row = {int(p): i for i, p in enumerate(pts.ids)}
    return pts


def read_sparse(sparse_dir):
    """Đọc cameras/images/points3D từ `sparse_dir`, tự nhận .bin hay .txt.

    Ưu tiên .bin khi có cả hai (đọc nhanh hơn, và là thứ BTC ship ở vòng 1–2).
    Trả về (cameras, images, points3D) — cùng kiểu dữ liệu với đường .bin, nên chỗ gọi
    không cần biết định dạng nguồn.
    """
    def pick(stem):
        b = os.path.join(sparse_dir, stem + ".bin")
        t = os.path.join(sparse_dir, stem + ".txt")
        if os.path.exists(b):
            return b, "bin"
        if os.path.exists(t):
            return t, "txt"
        raise FileNotFoundError(
            f"không thấy {stem}.bin lẫn {stem}.txt trong {sparse_dir} — "
            f"kiểm lại đường dẫn sparse (BTC đặt ở train/sparse/0/)"
        )

    cpath, ckind = pick("cameras")
    ipath, ikind = pick("images")
    ppath, pkind = pick("points3D")
    cams = read_cameras_bin(cpath) if ckind == "bin" else read_cameras_txt(cpath)
    imgs = read_images_bin(ipath) if ikind == "bin" else read_images_txt(ipath)
    pts = read_points3d_bin(ppath) if pkind == "bin" else read_points3d_txt(ppath)
    if ckind == "txt" or ikind == "txt" or pkind == "txt":
        print(f"[colmap] đọc sparse dạng TEXT: cameras={ckind} images={ikind} "
              f"points3D={pkind} ({len(pts.ids)} điểm)", flush=True)
    return cams, imgs, pts
