#!/bin/bash
# 31_reftrain.sh — train refiner UNet TRÊN CHÍNH DATA BTC VÒNG NÀY. ~3,5 h một card.
#
# Đây là bước làm cho toàn bộ bài nộp tự chứa: sau bước này, mọi trọng số trong
# pipeline đều sinh ra từ dữ liệu BTC phát vòng này. Không mang gì từ vòng khác vào.
source "$(dirname "$0")/lib.sh"
g=${GPU:-$(echo "$VT_GPUS" | cut -d, -f1)}
DUMPS=$(ls -d "$VT_RUNS"/refdata_s* "$VT_RUNS"/refdata_ho* 2>/dev/null | paste -sd, -)
[ -n "$DUMPS" ] || die "chưa có dump — chạy scripts/30_refdata.sh trước"

if ! is_done refiner; then
  need_gpu "$g"; ram_gate
  log "train refiner trên: $DUMPS"
  log "  (~3,5 h ở $VT_REF_ITERS iter. Thiếu giờ: VT_REF_ITERS=48000, mất ~0,7 điểm)"
  CUDA_VISIBLE_DEVICES=$g "$PY" -u "$VT_SRC/refine/refiner_train.py" train \
    --data "$DUMPS" --out "$VT_RUNS/refiner.pt" $VT_REF_TRAIN_FLAGS \
    >"$VT_LOGS/refiner.log" 2>&1 || die "refiner train hỏng — xem $VT_LOGS/refiner.log"
  mark_done refiner
fi
[ -f "$VT_RUNS/refiner.pt" ] || die "không thấy $VT_RUNS/refiner.pt"
log "REFINER XONG: $VT_RUNS/refiner.pt ($(du -h "$VT_RUNS/refiner.pt" | cut -f1))"
log "⚠ val crop KHÔNG dự đoán được điểm test (đã sai 7 lần) — chỉ dùng để chọn ckpt."
