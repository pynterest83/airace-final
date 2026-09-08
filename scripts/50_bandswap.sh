#!/bin/bash
# 50_bandswap.sh — GIAI ĐOẠN 3: trộn theo TẦN SỐ giữa các seed.
#   LF = trung bình các seed  (nhiễu cấu trúc thô triệt tiêu nhau → PSNR lên)
#   HF = 100% từ MỘT seed     (trung bình băng cao làm mờ → thước phạt qua LPIPS/SSIM)
# Một seed cũng chạy được (chỉ là chép), nhưng đòn chỉ ăn khi có ≥2 seed.
source "$(dirname "$0")/lib.sh"
DIRS=(); for s in $VT_SEEDS; do d="$VT_RUNS/pred_s$s"; [ -d "$d" ] && DIRS+=("$s:$d"); done
[ ${#DIRS[@]} -gt 0 ] || die "chưa có kết quả seed nào — chạy scripts/40_infer.sh"
log "band-swap trên ${#DIRS[@]} seed, sigma=$VT_BANDSWAP_SIGMA"

# HF lấy từ seed điểm cao nhất nếu chấm được, không thì seed đầu tiên.
best="${DIRS[0]%%:*}"
if [ -n "${VT_GT_DIR:-}" ] && [ -d "$VT_GT_DIR" ]; then
  bv=-1
  for e in "${DIRS[@]}"; do
    s="${e%%:*}"; [ -f "$VT_SCORES/pred_s$s.csv" ] || continue
    v=$(read_score "pred_s$s")
    awk "BEGIN{exit !($v > $bv)}" && { bv=$v; best=$s; }
  done
  log "seed tốt nhất = $best ($bv) → lấy làm nguồn tần số cao"
  else
    # Không có GT test (đúng tình huống ngày thi) → xếp hạng bằng proxy holdout
    # do 30_refdata đã chấm sẵn. Trước đây chỗ này lấy seed ĐẦU MẢNG, tức chọn mù:
    # đo được trên 3 seed thật, chọn nhầm seed kém mất 0,10 điểm — và band-swap
    # lúc đó còn TỆ HƠN dùng một mình seed tốt nhất (61,9113 vs 61,9166).
    bv=-1
    for e in "${DIRS[@]}"; do
      s="${e%%:*}"; [ -f "$VT_SCORES/H_s$s.csv" ] || continue
      v=$(read_score "H_s$s")
      awk "BEGIN{exit !($v > $bv)}" && { bv=$v; best=$s; }
    done
    if [ "$bv" = "-1" ]; then
      log "⚠ không có điểm holdout (scores/H_s*.csv) → HF đành lấy seed đầu: $best"
      log "   chạy 30_refdata.sh để sinh điểm holdout thì sẽ chọn có căn cứ hơn"
    else
      log "không có GT → xếp hạng bằng proxy holdout: seed $best ($bv) làm nguồn HF"
    fi
  fi

LF=$(printf "%s," "${DIRS[@]#*:}" | sed 's/,$//' | tr ',' '\n' | sed 's/$/:1/' | paste -sd, -)
run bandswap "$PY" "$VT_SRC/post/bandswap.py" \
  --hf "$VT_RUNS/pred_s$best" --lf "$LF" \
  --out "$VT_RUNS/final" --sigma "$VT_BANDSWAP_SIGMA"
score final "$VT_RUNS/final"
log "GIAI ĐOẠN 3 XONG → $VT_RUNS/final"
