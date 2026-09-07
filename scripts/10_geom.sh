#!/bin/bash
# 10_geom.sh — GIAI ĐOẠN 0: chuẩn hoá hình học (dựng lại SfM với camera CÓ MÉO).
# Đây là đòn lớn nhất đã đo được: +3,90 điểm. Chỉ chạy khi cổng Sampson > VT_SAMPSON_GATE.
# Chạy MỘT lần cho mỗi scene. ~4 h khớp (chia mảnh) + ~75 phút SfM (CPU).
source "$(dirname "$0")/lib.sh"
need_disk 80
G="$(echo "$VT_GPUS" | tr ',' ' ')"; NG=$(echo "$G" | wc -w)
M="$VT_RUNS/geom"; mkdir -p "$M"

# LightGlue hay nằm ngoài venv. Kiểm TRƯỚC khi phóng 4 tiến trình 4 giờ rồi mới chết.
export PYTHONPATH="${VT_LIGHTGLUE:-}${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c "import lightglue" 2>/dev/null \
  || die "không import được lightglue. Đặt VT_LIGHTGLUE=<thư mục> trong config/recipe.env, hoặc chạy: bash env/setup_env.sh --with-lightglue"

# ── 1. Khớp đặc trưng dày SuperPoint + LightGlue trên MỌI cặp ảnh chồng phủ ──
if ! is_done geom_match; then
  # Kiểm TRƯỚC khi phóng $NG tiến trình — không thì hỏng ở phút thứ 8 mới biết.
  # Khớp dày cần ~6 GB/tiến trình; KHÔNG cần card rảnh, chạy chung job khác được.
  for g in $G; do need_vram "$g" "${VT_MATCH_VRAM_GB:-8}"; done
  log "khớp dày: res=$VT_MATCH_RES kp=$VT_MATCH_KP min_ov=$VT_MATCH_MIN_OV, $NG mảnh song song"
  for g in $G; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    [ "$used" -gt 20000 ] && log "  ⚠ card $g đang có job khác dùng ${used} MiB — vẫn chạy được, nhưng chậm hơn"
  done
  i=0
  for g in $G; do
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$PYTHONPATH" "$PY" "$VT_SRC/geom/match_dense.py" \
      --scene_dir "$VT_SCENE" --audit "$VT_RUNS/sfm_audit.json" --out "$M/match" \
      --res "$VT_MATCH_RES" --kp "$VT_MATCH_KP" --min_ov "$VT_MATCH_MIN_OV" \
      --ransac "$VT_RANSAC" \
      --shard "$i/$NG" --save_every 1000 >"$VT_LOGS/geom_match_$i.log" 2>&1 &
    i=$((i+1))
  done
  wait
  mark_done geom_match
fi

# ── 2. Dựng DB + tách 10% cặp GIỮ LẠI (không đưa vào tối ưu) làm cổng kiểm chứng ──
if ! is_done geom_db; then
  # GLOMAP chỉ cần DB + cặp giữ lại ⇒ --db_only bỏ tam giác hoá + BA từ pose BTC (~55' CPU)
  DBFLAG=""; [ "$VT_SFM_MODE" = "glomap" ] && DBFLAG="--db_only 1"
  run geom_db "$PY" "$VT_SRC/geom/pairs_db.py" --scene_dir "$VT_SCENE" \
    --match_dir "$M/match" --out "$M/db" --cam_model "$VT_SFM_CAM" \
    --tri_reproj 30 --tri_angle 0.5 $DBFLAG
  mark_done geom_db
fi

# ── 3. SfM dựng lại, camera tự hiệu chuẩn méo — ĐÂY là đòn ──
if ! is_done geom_sfm; then
  if [ "$VT_SFM_MODE" = "glomap" ]; then
    # GLOMAP (global_mapping trong pycolmap 4.2): ~25' thay 75', nghiệm trùng (2,53 px, k1 -0,104)
    run geom_sfm "$PY" "$VT_SRC/geom/sfm_glomap.py" --scene_dir "$VT_SCENE" \
      --match_dir "$M/match" --ba_dir "$M/db" --out "$M/sfm" \
      --audit "$VT_RUNS/sfm_audit.json" --cam_model "$VT_SFM_CAM" --threads "$VT_CPU_THREADS"
  else
    run geom_sfm "$PY" "$VT_SRC/geom/sfm_rebuild.py" --scene_dir "$VT_SCENE" \
      --match_dir "$M/match" --ba_dir "$M/db" --out "$M/sfm" \
      --threads "$VT_CPU_THREADS" --refine_extra 1
  fi
  mark_done geom_sfm
fi

# ── 4. CỔNG: Sampson trên cặp giữ lại phải < 3 px, và đủ số ảnh đăng ký ──
log "CỔNG KIỂM CHỨNG — Sampson trên cặp giữ lại (không hề đưa vào tối ưu)"
"$PY" "$VT_SRC/geom/check_geometry.py" --scene_dir "$VT_SCENE" \
  --match_dir "$M/match" --ba_dir "$M/db" --sparse "$M/sfm/sparse/0" \
  2>&1 | tee "$VT_LOGS/geom_gate.log"
echo
log "⚠ ĐỌC SỐ TRÊN. Nếu Sampson mới KHÔNG < ${VT_SAMPSON_GATE} px, hoặc thiếu ảnh đăng ký,"
log "  thì DỪNG và train trên scene gốc — nền hình học hỏng còn tệ hơn nền BTC."
# AUTO_GATE=1 (hoặc chạy không có terminal) → đi tiếp không hỏi.
# Mặc định tương tác thì vẫn hỏi, vì đây là quyết định 5 giờ công việc.
if [ "${AUTO_GATE:-0}" = "1" ] || [ ! -t 0 ]; then
  log "AUTO_GATE — đi tiếp không hỏi. TỰ ĐỌC $VT_LOGS/geom_gate.log để xác nhận cổng."
else
  read -rp "Cổng đã đạt? tiếp tục dựng scene [y/N] " ok
  [ "$ok" = "y" ] || die "dừng theo yêu cầu — train thẳng bằng VT_TRAIN_SCENE=\$VT_SCENE"
fi

# ── 5. Dựng scene mới + chuyển pose test sang hệ toạ độ mới ──
# build_scene.py đọc CẢ train_w2c_old.json lẫn train_w2c_new.json từ một --ba_dir.
# Đường GLOMAP để chúng ở hai chỗ (old ở db/, new ở sfm/) — gom lại trước khi gọi.
[ -f "$M/sfm/train_w2c_old.json" ] || cp "$M/db/train_w2c_old.json" "$M/sfm/" \
  || die "không thấy $M/db/train_w2c_old.json — pairs_db chưa chạy xong?"
run geom_scene "$PY" "$VT_SRC/geom/build_scene.py" --scene_dir "$VT_SCENE" \
  --ba_dir "$M/sfm" --out_scene "$VT_SCENE_FIXED_RAW"
run geom_transfer "$PY" "$VT_SRC/geom/transfer_test_poses.py" --orig_scene "$VT_SCENE" \
  --new_scene "$VT_SCENE_FIXED_RAW" --out_scene "$VT_SCENE_FIXED" --pts kp
mark_done geom
log "GIAI ĐOẠN 0 XONG → scene để train: $VT_SCENE_FIXED"
