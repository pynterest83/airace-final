#!/bin/bash
# fetch_models.sh — tải MỌI trọng số pretrained mà pipeline cần, vào contest/work/models.
# Chạy MỘT lần trên máy có Internet. Sau đó pipeline chạy offline được.
#
# Pipeline KHÔNG dùng model 3DGS hay diffusion pretrained nào — refiner train từ đầu
# trên data BTC vòng thi. Bốn thứ dưới đây đều là mạng thị giác đa dụng, không gắn scene.
set -uo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/config/recipe.env"
mkdir -p "$TORCH_HOME"
echo "TORCH_HOME = $TORCH_HOME"; echo

"$PY" - <<'PY'
import os, sys, torch
ok = True

def gate(name, fn, why):
    global ok
    try:
        fn(); print(f"  ✔ {name:34} {why}")
    except Exception as e:
        ok = False; print(f"  ✘ {name:34} {type(e).__name__}: {str(e)[:70]}")

# 1. VGG16 — LPIPS dùng, cho CẢ bộ chấm LẪN loss của refiner. Thiếu = đứng hình.
def _vgg():
    import lpips; lpips.LPIPS(net="vgg")
gate("VGG16 (LPIPS)", _vgg, "bộ chấm + loss refiner")

# 2. RAFT — depth-fix trong refiner_data
def _raft():
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    raft_large(weights=Raft_Large_Weights.DEFAULT)
gate("RAFT-large", _raft, "depth-fix (giai đoạn 2)")

# 3+4. SuperPoint + LightGlue — chỉ giai đoạn 0 (khớp dày)
def _lg():
    sys.path.insert(0, os.environ.get("VT_LIGHTGLUE", ""))
    from lightglue import LightGlue, SuperPoint
    SuperPoint(max_num_keypoints=256).eval(); LightGlue(features="superpoint").eval()
gate("SuperPoint + LightGlue", _lg, "khớp dày (giai đoạn 0)")

print()
sys.exit(0 if ok else 1)
PY
rc=$?

echo; echo "── đã tải vào $TORCH_HOME ──"
find "$TORCH_HOME" -name '*.pth' -printf '%s %f\n' 2>/dev/null | sort -rn \
  | awk '{printf "  %8.1f MB  %s\n", $1/1048576, $2}' | head -10
echo "  tổng: $(du -sh "$TORCH_HOME" 2>/dev/null | cut -f1)"
[ $rc -eq 0 ] && echo "  TẤT CẢ SẴN SÀNG — pipeline chạy offline được" \
              || echo "  ⚠ CÓ TRỌNG SỐ CHƯA TẢI ĐƯỢC — kiểm mạng rồi chạy lại"
exit $rc
