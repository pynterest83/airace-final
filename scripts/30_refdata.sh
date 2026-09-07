#!/bin/bash
# 30_refdata.sh — sinh DỮ LIỆU HUẤN LUYỆN cho refiner, CHỈ từ ảnh train của BTC vòng này.
#
# Cách làm: train 4 model 3DGS, mỗi model GIẤU 1/4 số view train. View bị giấu là view
# model CHƯA THẤY nhưng ta CÓ ảnh thật ⇒ cặp (render lỗi thật, ảnh thật) đúng phân bố lỗi
# của pose test. Bốn model phủ đủ 100% view train.
#
# ⚠ KHÔNG đọc một pixel ground-truth test nào. Không dùng dữ liệu vòng nào khác.
#   Đây chính là lý do refiner train theo cách này là hợp lệ theo §11.1.
source "$(dirname "$0")/lib.sh"
need_disk 120
G=($(echo "$VT_GPUS" | tr ',' ' ')); NG=${#G[@]}
HE=$VT_HOLDOUT_EVERY

# ─────────────────────────────────────────────────────────────────────────────
# CHẾ ĐỘ p2 — không train model nào cả. Ba model seed (đã train ở 20_train.sh với
# --holdout_every 6) mỗi cái giấu 1/6 view train; ta chỉ dump từ chúng.
# Tiết kiệm ~7 GPU-giờ so với chế độ separate.
# ⚠ 20_train.sh PHẢI chạy TRƯỚC bước này (run_all.sh đã xếp đúng thứ tự).
# ─────────────────────────────────────────────────────────────────────────────
if [ "${VT_REFDATA_MODE:-p2}" = "p2" ]; then
  need_disk 60
  dump_seed() {  # dump_seed <seed> <gpu>
    local s=$1 g=$2 tag="s$s"
    [ -f "$VT_RUNS/gs_s$s/ckpt.pt" ] || { echo "[LỖI] chưa có $VT_RUNS/gs_s$s/ckpt.pt — chạy 20_train.sh trước"; return 1; }
    is_done "dump_$tag" && { log "SKIP dump_$tag"; return 0; }
    need_gpu "$g"
    log "dump refiner data từ model seed $s (giấu 1/$VT_P2_EVERY view) trên card $g"
    CUDA_VISIBLE_DEVICES=$g "$PY" "$VT_SRC/refine/refiner_data.py" \
      --result_dir "$VT_RUNS/gs_s$s" --scene_dir "$VT_TRAIN_SCENE" \
      --dump "$VT_RUNS/refdata_$tag" --targets holdout \
      --holdout_every "$VT_P2_EVERY" --K "$VT_REF_K" --depth_fix 1 \
      ${VT_REFDATA_LIMIT:+--limit $VT_REFDATA_LIMIT} \
      >"$VT_LOGS/dump_$tag.log" 2>&1 || { echo "[LỖI] dump_$tag"; return 1; }
    mark_done "dump_$tag"
    log "xong dump_$tag ($(ls -d "$VT_RUNS/refdata_$tag"/DJI* 2>/dev/null | wc -l) view)"
  }
  i=0; for s in $VT_SEEDS; do queue_run "$i" dump_seed "$s"; i=$((i+1)); done
  fail=0; queue_wait || fail=1
  [ "$fail" = 0 ] || die "có dump hỏng — xem logs/"
  n=$(ls -d "$VT_RUNS"/refdata_s*/ 2>/dev/null | wc -l)
  log "DỮ LIỆU REFINER XONG (p2): $n dump từ $n model seed"
  # proxy khi không có GT test: chấm luôn view holdout của dump đầu tiên
  if [ -z "${VT_GT_DIR:-}" ]; then
    d=$(ls -d "$VT_RUNS"/refdata_s*/ 2>/dev/null | head -1)
    [ -n "$d" ] && score_holdout "H_raw_$(basename "$d")" "$d"
  fi
  exit 0
fi


holdout_one() {  # holdout_one <offset> <gpu>
  local O=$1 g=$2 tag="ho${HE}o$1"
  if ! is_done "gs_$tag"; then
    need_gpu "$g"
    log "train model holdout $tag (giấu residue $O) trên card $g"
    CUDA_VISIBLE_DEVICES=$g "$PY" -u "$VT_SRC/gs/trainer.py" \
      --scene_dir "$VT_TRAIN_SCENE" --result_dir "$VT_RUNS/gs_$tag" \
      $VT_GS_FLAGS --seed 42 --holdout_every "$HE" --holdout_offset "$O" \
      >"$VT_LOGS/gs_$tag.log" 2>&1
    # ⚠ trainer.py CHẮC CHẮN kết thúc bằng AssertionError ở eval_holdout khi scene có méo
    # ("holdout eval needs distorted mode"). Đây là hành vi ĐÃ BIẾT: xảy ra SAU khi ckpt.pt
    # đã ghi xong. Điểm holdout ta tính bằng đường riêng. Nghiệm thu theo CKPT, không theo exit code.
    [ -f "$VT_RUNS/gs_$tag/ckpt.pt" ] || { echo "[LỖI] gs_$tag — không có ckpt.pt"; return 1; }
    grep -q "holdout eval needs distorted mode" "$VT_LOGS/gs_$tag.log" \
      && log "  ($tag: eval_holdout của trainer bỏ qua như dự kiến — ckpt vẫn tốt)"
    rm -f "$VT_RUNS/gs_$tag/ckpt_mid.pt"; mark_done "gs_$tag"
  fi
  if ! is_done "dump_$tag"; then
    log "dump refiner data $tag (rasterize + chọn nguồn + depth-fix + warp + mask)"
    CUDA_VISIBLE_DEVICES=$g "$PY" "$VT_SRC/refine/refiner_data.py" \
      --result_dir "$VT_RUNS/gs_$tag" --scene_dir "$VT_TRAIN_SCENE" \
      --dump "$VT_RUNS/refdata_$tag" --targets holdout \
      --holdout_every "$HE" --K "$VT_REF_K" --depth_fix 1 ${VT_REFDATA_LIMIT:+--limit $VT_REFDATA_LIMIT} \
      >"$VT_LOGS/dump_$tag.log" 2>&1 || { echo "[LỖI] dump_$tag"; return 1; }
    mark_done "dump_$tag"
  fi
  log "xong $tag"
}

for O in $(seq 0 $((HE-1))); do queue_run "$O" holdout_one "$O"; done
fail=0; queue_wait || fail=1
[ "$fail" = 0 ] || die "có nhánh holdout hỏng — xem logs/"
log "DỮ LIỆU REFINER XONG: $(ls -d "$VT_RUNS"/refdata_ho* 2>/dev/null | wc -l) dump"
