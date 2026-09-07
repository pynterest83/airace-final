#!/bin/bash
# run_all.sh — chạy toàn bộ pipeline từ data thô tới file zip.
# Idempotent: mỗi bước xong ghi một mark, chạy lại thì bỏ qua bước đã xong.
#
#   bash scripts/run_all.sh            # đầy đủ
#   SKIP_GEOM=1 bash scripts/run_all.sh   # bỏ giai đoạn 0 (hình học BTC đã ổn)
#   VT_SEEDS=42 bash scripts/run_all.sh   # chế độ gấp: 1 seed
source "$(dirname "$0")/lib.sh"
H="$(dirname "$0")"
t0=$(date +%s)

log "════ 0. ĐO DỮ LIỆU ════"
bash "$H/00_probe.sh" || die "00_probe.sh hỏng — dừng, đừng chạy tiếp trên nền hỏng"

if [ "${SKIP_GEOM:-0}" != "1" ]; then
  log "════ 1. CHUẨN HOÁ HÌNH HỌC ════"
  bash "$H/10_geom.sh" || die "10_geom.sh hỏng — dừng, đừng chạy tiếp trên nền hỏng"
else
  log "════ 1. HÌNH HỌC — BỎ QUA theo SKIP_GEOM=1 ════"
fi

if [ "${VT_REFDATA_MODE:-p2}" = "p2" ]; then
  # p2: model seed vừa là nền vừa sinh data refiner ⇒ phải train seed TRƯỚC.
  log "════ 2. TRAIN 3DGS ${VT_SEEDS} (giấu 1/$VT_P2_EVERY view cho refiner) ════"
  bash "$H/20_train.sh" || die "20_train.sh hỏng — dừng, đừng chạy tiếp trên nền hỏng"
  log "════ 3. DỮ LIỆU + TRAIN REFINER ════"
  bash "$H/30_refdata.sh" || die "30_refdata.sh hỏng — dừng, đừng chạy tiếp trên nền hỏng"
  bash "$H/31_reftrain.sh" || die "31_reftrain.sh hỏng — dừng, đừng chạy tiếp trên nền hỏng"
else
  # separate: 4 model holdout riêng, độc lập với 3 seed ⇒ chạy song song được.
  log "════ 2. DỮ LIỆU + TRAIN REFINER ════"
  bash "$H/30_refdata.sh" || die "30_refdata.sh hỏng — dừng, đừng chạy tiếp trên nền hỏng"
  bash "$H/31_reftrain.sh" || die "31_reftrain.sh hỏng — dừng, đừng chạy tiếp trên nền hỏng"
  log "════ 3. TRAIN 3DGS ${VT_SEEDS} ════"
  bash "$H/20_train.sh" || die "20_train.sh hỏng — dừng, đừng chạy tiếp trên nền hỏng"
fi

log "════ 4. SUY LUẬN ════"
for s in $VT_SEEDS; do SEED=$s bash "$H/40_infer.sh" || die "40_infer seed $s hỏng"; done

log "════ 5. BAND-SWAP ════"
bash "$H/50_bandswap.sh" || die "50_bandswap.sh hỏng — dừng, đừng chạy tiếp trên nền hỏng"

log "════ 6. ĐÓNG GÓI ════"
bash "$H/60_submit.sh" || die "60_submit.sh hỏng — dừng, đừng chạy tiếp trên nền hỏng"

log "TOÀN BỘ XONG sau $(( ($(date +%s) - t0) / 60 )) phút"
