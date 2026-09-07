#!/bin/bash
# 40_infer.sh — GIAI ĐOẠN 2: suy luận TRONG MỘT TIẾN TRÌNH.
#
# rasterize trên canvas đệm → chọn K nguồn train → depth-fix → warp → UNet fp16 tiled
# → ánh xạ ngược về khung có méo. Tất cả trong một tiến trình, KHÔNG ghi PNG trung gian.
#   đo được: 4,4 s/view ≈ 7,5'/seed  (đường 3 bước cũ: 41'/seed) — điểm y hệt (61,9166 vs 61,9148)
#
#   SEED=42 GPU=0 bash scripts/40_infer.sh
source "$(dirname "$0")/lib.sh"
need_disk 20
S=${SEED:?đặt SEED=<seed>}; g=${GPU:-$(echo "$VT_GPUS" | cut -d, -f1)}
MODEL="$VT_RUNS/gs_s$S"
[ -f "$MODEL/ckpt.pt" ] || die "không thấy $MODEL/ckpt.pt — chạy 20_train.sh trước"
[ -f "$VT_REFINER" ]    || die "không thấy refiner $VT_REFINER — chạy 31_reftrain.sh trước"

if ! is_done "infer_$S"; then
  need_gpu "$g"
  # --pad 1        canvas đệm (nội hoá audit_padded): refiner chạy cả ở góc ảnh, +0,73đ
  # --render_aa 1  kênh render rasterize antialiased, khớp base train --antialiased 1 (+0,03)
  # --no_dump 1    không ghi PNG trung gian — đây là chỗ tiết kiệm phần lớn thời gian
  # --apply_ckpt   áp UNet ngay trong tiến trình, ảnh ra đã redistort về khung GT
  run "infer_$S" env CUDA_VISIBLE_DEVICES=$g "$PY" "$VT_SRC/refine/refiner_data.py" \
    --result_dir "$MODEL" --scene_dir "$VT_TRAIN_SCENE" --targets test \
    --dump "$VT_RUNS/.dump_s$S" --no_dump 1 \
    --pad 1 --render_aa 1 --K "$VT_REF_K" --depth_fix 1 \
    --apply_ckpt "$VT_REFINER" --apply_out "$VT_RUNS/pred_s$S" \
    --apply_ch "$VT_REF_CH" --apply_fp16 1
  mark_done "infer_$S"
fi
n=$(ls "$VT_RUNS/pred_s$S" 2>/dev/null | wc -l)
log "SUY LUẬN seed $S XONG → $VT_RUNS/pred_s$S ($n ảnh)"
score "pred_s$S" "$VT_RUNS/pred_s$S"
