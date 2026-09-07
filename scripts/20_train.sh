#!/bin/bash
# 20_train.sh — GIAI ĐOẠN 1: train 3DGS. Một seed mỗi card, chạy song song.
#   bash scripts/20_train.sh              # mọi seed trong VT_SEEDS
#   SEED=42 GPU=0 bash scripts/20_train.sh   # một seed cụ thể
source "$(dirname "$0")/lib.sh"
need_disk 60
log "scene train: $VT_TRAIN_SCENE   cap=$VT_CAP steps=$VT_STEPS"
[ "$VT_TRAIN_SCENE" = "$VT_SCENE" ] && log "  (dùng hình học GỐC của BTC — giai đoạn 0 chưa chạy hoặc đã bỏ qua)"

train_one() {  # train_one <seed> <gpu>
  local s=$1 g=$2 tag="gs_s$s"
  is_done "$tag" && { log "SKIP $tag (đã xong)"; return 0; }
  need_gpu "$g"
  log "train $tag trên card $g"
  CUDA_VISIBLE_DEVICES=$g "$PY" -u "$VT_SRC/gs/trainer.py" \
    --scene_dir "$VT_TRAIN_SCENE" --result_dir "$VT_RUNS/$tag" \
    $VT_GS_FLAGS --seed "$s" >"$VT_LOGS/$tag.log" 2>&1 \
    || true   # eval_holdout có thể ném assert sau khi đã lưu ckpt — nghiệm thu theo ckpt
  [ -f "$VT_RUNS/$tag/ckpt.pt" ] || { echo "[LỖI] $tag — không có ckpt.pt, xem $VT_LOGS/$tag.log"; return 1; }
  rm -f "$VT_RUNS/$tag/ckpt_mid.pt"      # chỉ để resume; ổ đầy đã giết 4 lô
  mark_done "$tag"
  log "xong $tag"
}

if [ -n "${SEED:-}" ]; then
  train_one "$SEED" "${GPU:-$(echo "$VT_GPUS" | cut -d, -f1)}"
else
  i=0
  for s in $VT_SEEDS; do queue_run "$i" train_one "$s"; i=$((i+1)); done
  fail=0; queue_wait || fail=1
  [ "$fail" = 0 ] || die "có seed train hỏng — xem logs/"
fi
log "GIAI ĐOẠN 1 XONG"
