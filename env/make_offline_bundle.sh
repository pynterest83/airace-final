#!/bin/bash
# make_offline_bundle.sh — chạy Ở MÁY NÀY (máy đã chạy được), TRƯỚC ngày thi.
# Gói mọi thứ máy thi cần mà KHÔNG tải được nếu không có Internet.
# Vòng 2 máy thi không có mạng — đừng để phát hiện điều này lúc 2 giờ sáng.
#
#   bash env/make_offline_bundle.sh [thư_mục_ra]
set -euo pipefail
OUT="${1:-$HOME/vt_offline}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"/{torch_hub,wheels,lg}

echo "═══ 1. trọng số torch.hub (LPIPS-VGG + RAFT) ═══"
SRC="$HOME/.cache/torch/hub/checkpoints"
# vgg16 = LPIPS (bộ chấm + loss refiner) · raft = depth-fix · *_lightglue = giai đoạn 0
for w in vgg16-*.pth raft_large_*.pth alexnet-*.pth superpoint*.pth *_lightglue_*.pth; do
  for f in $SRC/$w; do [ -f "$f" ] && cp -v "$f" "$OUT/torch_hub/"; done
done 2>/dev/null

echo "═══ 2. LightGlue + SuperPoint ═══"
[ -d "${VT_LIGHTGLUE:-$HOME/lg_pkgs}" ] \
  && cp -r "${VT_LIGHTGLUE:-$HOME/lg_pkgs}" "$OUT/lg/" && echo "  đã copy lg_pkgs" \
  || echo "  ⚠ không thấy lg_pkgs — giai đoạn 0 sẽ không chạy được offline"

echo "═══ 3. wheel của mọi phụ thuộc ═══"
PY="${VT_VENV:-$HOME/projects/vt-track1/venv_day1}/bin/python"
"$PY" -m pip download -q -d "$OUT/wheels" -r "$HERE/requirements.txt" 2>&1 | tail -2 \
  || echo "  ⚠ tải wheel thất bại (không mạng?) — chạy script này lúc CÒN mạng"

echo "═══ 4. mã nguồn ═══"
CDIR="$(dirname "$HERE")"
tar czf "$OUT/contest_code.tgz" --exclude=__pycache__ --exclude='.done_*' \
  -C "$(dirname "$CDIR")" "$(basename "$CDIR")" && echo "  contest_code.tgz ($(du -h "$OUT/contest_code.tgz" | cut -f1))"

cat > "$OUT/CAI_DAT.txt" <<'TXT'
Cài trên máy thi (không cần Internet)
════════════════════════════════════
1. tar xzf contest_code.tgz && cd contest
2. mkdir -p ~/.cache/torch/hub/checkpoints && cp ../torch_hub/* $_
3. cp -r ../lg/lg_pkgs ~/            # rồi đặt VT_LIGHTGLUE=~/lg_pkgs
4. PIP_OFFLINE=../wheels bash env/setup_env.sh --with-lightglue
5. bash env/verify_env.sh            # phải ĐẬU hết
TXT
echo; echo "XONG → $OUT  ($(du -sh "$OUT" | cut -f1))"
echo "Copy cả thư mục này sang máy thi. Hướng dẫn: $OUT/CAI_DAT.txt"
