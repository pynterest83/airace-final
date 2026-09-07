#!/bin/bash
# 60_submit.sh — đóng gói + KIỂM TRA bài nộp. Chạy cái này, đừng tự zip bằng tay.
# Kiểm đúng checklist §14 đề bài: đủ ảnh, đúng tên theo CSV, đúng kích thước, PNG, không thừa.
source "$(dirname "$0")/lib.sh"
SRC_DIR=${1:-$VT_RUNS/final}
OUT=${2:-$VT_SUBMIT/submission_$(date -u +%d%b_%H%M).zip}
[ -d "$SRC_DIR" ] || die "không thấy $SRC_DIR"
log "đóng gói $SRC_DIR → $OUT"

"$PY" - "$SRC_DIR" "$VT_SCENE/test/test_poses.csv" "$OUT" <<'PYEOF'
import csv, os, sys, zipfile
import cv2

src, csv_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
rows = list(csv.DictReader(open(csv_path)))
have = {os.path.splitext(f)[0]: f for f in os.listdir(src)}
problems, packed = [], 0

with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
    for r in rows:
        name = r["image_name"]
        w, h = int(r["width"]), int(r["height"])
        stem = os.path.splitext(name)[0]
        if stem not in have:
            problems.append(f"THIẾU  {name}")
            continue
        p = os.path.join(src, have[stem])
        im = cv2.imread(p)
        if im is None:
            problems.append(f"HỎNG   {name} — không đọc được")
            continue
        if (im.shape[1], im.shape[0]) != (w, h):
            problems.append(f"SAI CỠ {name} — có {im.shape[1]}x{im.shape[0]}, cần {w}x{h}")
            continue
        # tên trong zip phải KHỚP CHÍNH XÁC image_name của CSV
        if have[stem].lower().endswith(".png"):
            zf.write(p, arcname=name)
        else:                       # nguồn không phải PNG → chuyển, đề bài yêu cầu PNG
            ok, buf = cv2.imencode(".png", im)
            if not ok:
                problems.append(f"HỎNG   {name} — mã hoá PNG thất bại"); continue
            zf.writestr(name, buf.tobytes())
        packed += 1
    extra = set(have) - {os.path.splitext(r["image_name"])[0] for r in rows}
    for e in sorted(extra):
        problems.append(f"THỪA   {e} — không có trong test_poses.csv")

print(f"\nCSV yêu cầu : {len(rows)} ảnh")
print(f"Đã đóng gói : {packed} ảnh")
print(f"File zip    : {out}  ({os.path.getsize(out)/2**20:.1f} MB)")
if problems:
    print(f"\n!!! {len(problems)} VẤN ĐỀ — KHÔNG NỘP FILE NÀY:")
    for p in problems[:40]:
        print("   " + p)
    sys.exit(1)
print("\n[OK] đủ ảnh, đúng tên, đúng kích thước, đúng định dạng PNG.")
PYEOF
[ $? -eq 0 ] || die "kiểm tra bài nộp THẤT BẠI — sửa rồi chạy lại"

log "SHA256: $(sha256sum "$OUT" | cut -d' ' -f1)"
"$PY" - "$OUT" <<'PYEOF'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
bad = z.testzip()
print(f"[LỖI] file hỏng trong zip: {bad}" if bad else f"[OK] zip giải nén thử được, {len(z.namelist())} mục")
sys.exit(1 if bad else 0)
PYEOF
[ $? -eq 0 ] || die "zip hỏng"
log "SẴN SÀNG NỘP: $OUT"
