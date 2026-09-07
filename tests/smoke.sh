#!/bin/bash
# smoke.sh — chạy TOÀN BỘ pipeline ở quy mô tí hon để chứng minh dây chuyền thông.
# Không nhằm ra điểm cao; nhằm bắt lỗi đường dẫn / cờ / định dạng TRƯỚC ngày thi.
# ~10-15 phút một card.
#
#   VT_SCENE=<scene> GPU=4 bash tests/smoke.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
export VT_ROOT="${VT_ROOT:-$HOME/vt_smoke}"
export VT_SCENE="${VT_SCENE:?đặt VT_SCENE=<thư mục scene>}"
export VT_VENV="${VT_VENV:?đặt VT_VENV=<venv>}"
export VT_GPUS="${GPU:-0}"
# quy mô tí hon
export VT_CAP=200000 VT_STEPS=200 VT_SEEDS="42" VT_HOLDOUT_EVERY=4
export VT_REF_ITERS=200 VT_REF_EVAL_EVERY=100 VT_REF_MAXVIEWS=8
export VT_TRAIN_SCENE="$VT_SCENE"
export FORCE=1
source "$HERE/config/recipe.env"
rm -rf "$VT_ROOT"; mkdir -p "$VT_RUNS" "$VT_LOGS" "$VT_SCORES" "$VT_SUBMIT"

pass=0; fail=0
step() {  # step <tên> <lệnh...>
  echo; echo "──────── $1 ────────"
  if "${@:2}"; then echo "✔ $1"; pass=$((pass+1)); else echo "✘ $1 THẤT BẠI"; fail=$((fail+1)); fi
}

step "probe dữ liệu" "$PY" "$VT_SRC/probe_data.py" --scene "$VT_SCENE" --out "$VT_RUNS/probe.json"

step "train 3DGS tí hon" env CUDA_VISIBLE_DEVICES=$VT_GPUS "$PY" -u "$VT_SRC/gs/trainer.py" \
  --scene_dir "$VT_SCENE" --result_dir "$VT_RUNS/gs_s42" $VT_GS_FLAGS --seed 42

step "model holdout + dump refiner" bash -c "
  CUDA_VISIBLE_DEVICES=$VT_GPUS '$PY' -u '$VT_SRC/gs/trainer.py' --scene_dir '$VT_SCENE' \
    --result_dir '$VT_RUNS/gs_ho4o0' $VT_GS_FLAGS --seed 42 --holdout_every 4 --holdout_offset 0 &&
  CUDA_VISIBLE_DEVICES=$VT_GPUS '$PY' '$VT_SRC/refine/refiner_data.py' \
    --result_dir '$VT_RUNS/gs_ho4o0' --scene_dir '$VT_SCENE' --dump '$VT_RUNS/refdata_ho4o0' \
    --targets holdout --holdout_every 4 --K 2 --depth_fix 1 --limit 8"

step "train refiner" env CUDA_VISIBLE_DEVICES=$VT_GPUS "$PY" -u "$VT_SRC/refine/refiner_train.py" train \
  --data "$VT_RUNS/refdata_ho4o0" --out "$VT_RUNS/refiner.pt" $VT_REF_TRAIN_FLAGS

step "render test pinhole" env CUDA_VISIBLE_DEVICES=$VT_GPUS "$PY" "$VT_SRC/refine/render_test_views.py" \
  --scene_dir "$VT_SCENE" --result_dir "$VT_RUNS/gs_s42" --out_dir "$VT_RUNS/gs_s42_pin"

step "đóng gói + kiểm bài nộp" bash "$HERE/scripts/60_submit.sh" \
  "$VT_RUNS/gs_s42_pin" "$VT_SUBMIT/smoke.zip"

echo; echo "════════════════════════════════"
echo "SMOKE: $pass đậu, $fail trượt"
echo "════════════════════════════════"
exit $((fail > 0))
