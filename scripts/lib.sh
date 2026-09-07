# lib.sh — helper dùng chung. Mọi script khác source file này (nó tự source recipe.env).
set -uo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_HERE/../config/recipe.env"

log()  { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$VT_LOGS/run.log"; }

# Kiểm scene có đủ thứ cần TRƯỚC khi đốt hàng giờ GPU.
check_scene() {  # check_scene [thư_mục]
  local s=${1:-$VT_TRAIN_SCENE}
  [ -d "$s/train/images" ] || die "không thấy $s/train/images"
  [ -d "$s/train/sparse/0" ] || die "không thấy $s/train/sparse/0"
  [ -f "$s/test/test_poses.csv" ] || die "không thấy $s/test/test_poses.csv"
  local n=$(ls "$s/train/images" | wc -l)
  [ "$n" -ge 10 ] || die "$s/train/images chỉ có $n ảnh — sai thư mục?"
  log "scene OK: $(basename "$s") — $n ảnh train, $(( $(wc -l < "$s/test/test_poses.csv") - 1 )) pose test"
}

# Chốt chặn: giai đoạn 0 đã chạy xong mà lại sắp train trên scene GỐC = mất +3,9 điểm.
# Đây từng là lỗi thật (VT_TRAIN_SCENE bị đóng băng lúc source). Đừng để nó im lặng nữa.
check_train_scene() {
  if is_done geom && [ "$VT_TRAIN_SCENE" = "$VT_SCENE" ]; then
    die "giai đoạn 0 ĐÃ XONG (có $VT_SCENE_FIXED) nhưng VT_TRAIN_SCENE vẫn trỏ scene GỐC.
       Sẽ mất toàn bộ đòn sửa hình học. Kiểm biến VT_TRAIN_SCENE_FORCE, hoặc bỏ export VT_TRAIN_SCENE."
  fi
}
die()  { echo "[LỖI] $*" | tee -a "$VT_LOGS/run.log" >&2; exit 1; }

# Chạy một lệnh, đo thời gian, ghi log riêng theo tag.
run() {  # run <tag> <lệnh...>
  local tag=$1; shift
  local t0=$(date +%s)
  log "BẮT ĐẦU $tag"
  "$@" >>"$VT_LOGS/$tag.log" 2>&1 || die "$tag thất bại — xem $VT_LOGS/$tag.log"
  log "XONG    $tag  ($(( $(date +%s) - t0 ))s)"
}

# Bỏ qua bước đã xong (idempotent — chạy lại script bao nhiêu lần cũng được).
done_mark() { echo "$VT_RUNS/.done_$1"; }
is_done()   { [ -f "$(done_mark "$1")" ]; }
mark_done() { touch "$(done_mark "$1")"; }

# Kiểm card trống trước khi phóng. FORCE=1 để bỏ qua.
need_gpu() {  # need_gpu <index>
  local g=${1:?cần index card} used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null) \
    || die "không đọc được nvidia-smi cho card $g"
  if [ "${used:-0}" -gt 20000 ] && [ "${FORCE:-0}" != "1" ]; then
    die "card $g đang dùng ${used} MiB. Đặt FORCE=1 nếu vẫn muốn chạy."
  fi
}

# Khác need_gpu: chỉ đòi ĐỦ VRAM TRỐNG, không đòi card rảnh hẳn.
# Dùng cho khâu nhẹ (khớp dày ~6 GB/tiến trình) chạy chung máy với job khác.
need_vram() {  # need_vram <index> <GB cần>
  local g=${1:?} need_gb=${2:?} free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null) \
    || die "không đọc được nvidia-smi cho card $g"
  if [ "$(( free / 1024 ))" -lt "$need_gb" ]; then
    die "card $g chỉ còn $(( free / 1024 )) GB trống, cần ${need_gb} GB. Đợi job khác xong, hoặc bớt card trong VT_GPUS."
  fi
}

# Chấm một thư mục render so GT. Ngày thi THƯỜNG KHÔNG CÓ GT test — khi đó dùng
# score_holdout (chấm trên view holdout, nơi ta có ảnh thật). Đừng im lặng bỏ qua:
# không đo được gì cả ngày thi là hỏng.
score() {  # score <tag> <thư_mục_render>
  local tag=$1 dir=$2
  if [ -n "${VT_GT_DIR:-}" ] && [ -d "$VT_GT_DIR" ]; then
    "$PY" "$VT_SRC/post/score_btc.py" --pred "$dir" --gt "$VT_GT_DIR" \
        --csv "$VT_SCORES/$tag.csv" --json "$VT_SCORES/$tag.json" --lpips_tile 1024 \
        2>&1 | tee -a "$VT_LOGS/score.log" | grep -E "SCORE|PSNR|SSIM|LPIPS" || true
  else
    log "chấm $tag: KHÔNG có VT_GT_DIR → dùng proxy holdout (score_holdout.sh)"
    log "  ⚠ proxy holdout KHÔNG bằng điểm test: đòn BASE chuyển 1:1, đòn REFINER bị"
    log "    phóng đại ~4× (đo paired 2 fold). Xem docs/02_NGAY_1_DO_DAC.md §3."
  fi
}

# Chấm proxy trên view holdout — dùng khi không có GT test.
#   score_holdout <tag> <dump_dir> [thư_mục_ảnh_đã_refine]
# Không truyền thư mục thứ 3 = chấm chính render thô trong dump.
score_holdout() {
  local tag=$1 dump=$2 pred=${3:-}
  [ -d "$dump" ] || { log "score_holdout $tag: không thấy dump $dump"; return 0; }
  "$PY" "$VT_SRC/post/score_holdout.py" --dump "$dump" ${pred:+--pred "$pred"} \
      --out "$VT_SCORES/$tag.csv" 2>&1 | tee -a "$VT_LOGS/score.log" | tail -3
}

# Đọc lại điểm đã chấm từ csv (dùng cho so sánh giữa các bước).
read_score() {  # read_score <tag>
  "$PY" - "$VT_SCORES/$1.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
v = [100*(0.4*(1-float(r['lpips'])) + 0.3*float(r['ssim']) + 0.3*min(float(r['psnr'])/50, 1)) for r in rows]
print(f"{sum(v)/len(v):.4f}")
PYEOF
}

# Hàng đợi: chạy nhiều job nhưng KHÔNG BAO GIỜ quá 1 job/card.
# Trước đây các script phóng hết job cùng lúc rồi chia card bằng "i % NG" — với 1 card
# thì mọi job đổ lên card 0, need_gpu chặn job thứ 2 và giết cả script. Đề bài §13 ghi
# cấu hình tham khảo là 1 GPU, nên đường 1 card phải chạy được.
declare -a _SLOT_PID=()
queue_run() {  # queue_run <chỉ_số_job> <hàm> <tham số...>
  local i=$1; shift
  local gpus=($(echo "$VT_GPUS" | tr ',' ' ')); local ng=${#gpus[@]}
  local slot=$(( i % ng )); local g=${gpus[$slot]}
  # chờ job trước ĐANG dùng đúng card này xong rồi mới phóng
  [ -n "${_SLOT_PID[$slot]:-}" ] && wait "${_SLOT_PID[$slot]}" 2>/dev/null
  "$@" "$g" &
  _SLOT_PID[$slot]=$!
}
queue_wait() {  # chờ hết, trả 1 nếu có job hỏng
  local rc=0 p
  for p in "${_SLOT_PID[@]}"; do [ -n "$p" ] && { wait "$p" || rc=1; }; done
  _SLOT_PID=(); return $rc
}

# Chờ các tiến trình refiner khác nạp xong dataset (chúng ngốn ~100 GB RAM mỗi cái).
ram_gate() {
  sleep 5
  until [ -z "$(ps -eo etimes,cmd | awk '$1<1500 && /refiner_train[.]py train/')" ]; do
    log "ram_gate: đợi refiner khác nạp dữ liệu xong…"; sleep 60
  done
}

# Kiểm ổ trước mỗi lô nặng — đã mất kết quả 4 lần vì đầy ổ giữa chừng.
need_disk() {  # need_disk <GB>
  local need=${1:-50} free
  free=$(df -BG --output=avail "$VT_ROOT" | tail -1 | tr -dc '0-9')
  [ "${free:-0}" -ge "$need" ] || die "chỉ còn ${free}G trống, cần ${need}G. Dọn bớt runs/ trước."
}
