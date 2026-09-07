#!/bin/bash
# 00_probe.sh — 2 giờ đầu ngày thi. CPU thuần, 0 GPU. Chạy TRƯỚC MỌI THỨ.
# Trả lời 4 câu quyết định recipe. Đọc kỹ phần ➜ QUYẾT ĐỊNH rồi mới sửa recipe.env.
source "$(dirname "$0")/lib.sh"

[ -d "$VT_SCENE/train/images" ] || die "không thấy $VT_SCENE/train/images — sửa VT_SCENE trong config/recipe.env"
log "probe scene: $VT_SCENE"

echo; echo "############ 1-3. CAMERA / TRACK / KIỂU CHIA TEST ############"
"$PY" "$VT_SRC/probe_data.py" --scene "$VT_SCENE" --out "$VT_RUNS/probe.json" \
  2>&1 | tee "$VT_LOGS/probe.log"

echo; echo "############ 4. CỔNG HÌNH HỌC (quyết định 5 giờ công việc) ############"
# Audit SfM của BTC từ chính file sparse: residual tái chiếu, độ ràng buộc, mẫu lỗi theo ô ảnh.
# Mẫu lỗi tăng dần từ tâm ra góc = camera model sai (thiếu méo) ⇒ phải chạy giai đoạn 0.
run probe_sfm "$PY" "$VT_SRC/geom/sfm_audit.py" --scene_dir "$VT_SCENE" --out "$VT_RUNS/sfm_audit.json"
"$PY" - "$VT_RUNS/sfm_audit.json" "$VT_SAMPSON_GATE" <<'PYEOF'
import json, sys
import numpy as np
d = json.load(open(sys.argv[1])); gate = float(sys.argv[2])
per = d["per_image"]
tr = [v for v in per.values() if v.get("train")]
te = [v for v in per.values() if v.get("test")]
res = np.array([v["res_med"] for v in tr if v.get("res_med") is not None])
p90 = np.array([v["res_p90"] for v in tr if v.get("res_p90") is not None])
gt3 = np.array([v["res_gt3"] for v in tr if v.get("res_gt3") is not None])
print(f"   ảnh train {len(tr)} · pose test trong sparse {len(te)} · điểm 3D {d['n_points']:,}")
print(f"   residual tái chiếu dưới pose BTC: trung vị {np.median(res):.2f} px · p90 {np.median(p90):.2f} px"
      f" · %match>3px {100*np.mean(gt3):.1f}")
if np.median(res) < gate:
    print(f"   ➜ residual < {gate} px — hình học BTC NHÌN có vẻ ổn.")
else:
    print(f"   ➜ residual ≥ {gate} px — gần như chắc phải CHẠY GIAI ĐOẠN 0.")
print("   ⚠ QUAN TRỌNG: con số này chỉ đo cặp ảnh CÓ điểm chung, nên nó LUÔN đẹp —")
print("     ở vòng trước residual 1,2 px mọi view mà hình học vẫn hỏng 13 px ở cặp")
print("     KHÔNG chia điểm chung. Cổng THẬT là Sampson trong 10_geom.sh bước 4.")
print("     Đừng bỏ giai đoạn 0 chỉ vì số này thấp.")
PYEOF

echo; echo "############ TÓM TẮT ############"
log "probe xong. Sửa config/recipe.env theo các dòng ➜ rồi chạy scripts/run_all.sh"
