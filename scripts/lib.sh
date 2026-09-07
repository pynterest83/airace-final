# lib.sh — helper dùng chung. Mọi script khác source file này (nó tự source recipe.env).
set -uo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_HERE/../config/recipe.env"

log()  { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$VT_LOGS/run.log"; }
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

# Chấm một thư mục render. Không có GT thì báo và bỏ qua, KHÔNG chết.
score() {  # score <tag> <thư_mục_render>
  local tag=$1 dir=$2
  if [ -z "${VT_GT_DIR:-}" ] || [ ! -d "$VT_GT_DIR" ]; then
    log "chấm $tag: BỎ QUA (chưa có VT_GT_DIR — dùng holdout để đo, xem docs/02)"
    return 0
  fi
  "$PY" "$VT_SRC/post/score_btc.py" --pred "$dir" --gt "$VT_GT_DIR" \
      --csv "$VT_SCORES/$tag.csv" --json "$VT_SCORES/$tag.json" --lpips_tile 1024 \
      2>&1 | tee -a "$VT_LOGS/score.log" | grep -E "SCORE|PSNR|SSIM|LPIPS" || true
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
