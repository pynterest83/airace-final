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
  log "không chấm được (chưa có GT) → HF lấy seed đầu: $best"
fi

LF=$(printf "%s," "${DIRS[@]#*:}" | sed 's/,$//' | tr ',' '\n' | sed 's/$/:1/' | paste -sd, -)
run bandswap "$PY" "$VT_SRC/post/bandswap.py" \
  --hf "$VT_RUNS/pred_s$best" --lf "$LF" \
  --out "$VT_RUNS/final" --sigma "$VT_BANDSWAP_SIGMA"
score final "$VT_RUNS/final"
log "GIAI ĐOẠN 3 XONG → $VT_RUNS/final"
