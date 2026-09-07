#!/bin/bash
# verify_env.sh — kiểm môi trường ĐỦ CHẠY chưa. Chạy sau setup_env.sh, và lại
# sau mỗi lần nâng torch/driver. Mỗi dòng là một cửa; TRƯỢT cửa nào thì sửa cửa đó.
source "$(cd "$(dirname "$0")/.." && pwd)/config/recipe.env"
fail=0
ck() { if eval "$2" >/dev/null 2>&1; then echo "  ĐẬU   $1"; else echo "  TRƯỢT $1"; fail=1; fi; }

echo "═══ thư viện ═══"
ck "python"          "'$PY' -V"
ck "torch + CUDA"    "'$PY' -c 'import torch;assert torch.cuda.is_available()'"
ck "gsplat"          "'$PY' -c 'import gsplat'"
ck "lpips"           "'$PY' -c 'import lpips'"
ck "pycolmap"        "'$PY' -c 'import pycolmap'"
ck "cv2"             "'$PY' -c 'import cv2'"
ck "torchvision RAFT" "'$PY' -c 'from torchvision.models.optical_flow import raft_large'"
ck "nvcc (gsplat JIT)" "command -v nvcc"
ck "ninja trên PATH"   "command -v ninja"      # thiếu = gsplat JIT chết giữa chừng
ck "lightglue (giai đoạn 0)" "PYTHONPATH='${VT_LIGHTGLUE:-}' '$PY' -c 'import lightglue'"

echo; echo "═══ mã nguồn nạp được ═══"
for m in gs/trainer refine/refiner_data refine/refiner_train refine/fuse_test \
         post/score_btc post/bandswap geom/sfm_audit probe_data; do
  ck "$m" "'$PY' -c \"
import importlib.util,sys
s=importlib.util.spec_from_file_location('p','$VT_SRC/$m.py')
m=importlib.util.module_from_spec(s); sys.argv=['p']
try: s.loader.exec_module(m)
except SystemExit: pass\""
done

echo; echo "═══ GPU ═══"
"$PY" - <<'PYEOF'
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gb = p.total_memory / 2**30
        print(f"  card {i}: {p.name}  {gb:.0f} GB   → cap_max an toàn ≈ {int(gb*380_000):,}")
else:
    print("  KHÔNG THẤY GPU")
PYEOF

echo; echo "═══ gsplat rasterize thật (bắt lỗi JIT) ═══"
"$PY" - <<'PYEOF'
import torch
try:
    from gsplat.rendering import rasterization
    n = 100
    out = rasterization(
        means=torch.randn(n, 3, device="cuda"),
        quats=torch.randn(n, 4, device="cuda"),
        scales=torch.rand(n, 3, device="cuda") * 0.1,
        opacities=torch.rand(n, device="cuda"),
        colors=torch.rand(n, 3, device="cuda"),
        viewmats=torch.eye(4, device="cuda")[None],
        Ks=torch.tensor([[[100., 0, 64], [0, 100., 64], [0, 0, 1.]]], device="cuda"),
        width=128, height=128)
    print(f"  ĐẬU   rasterization chạy, ra {tuple(out[0].shape)}")
except Exception as e:
    print(f"  TRƯỢT rasterization: {type(e).__name__}: {e}")
PYEOF

echo; echo "═══ bộ chấm ═══"
"$PY" "$VT_SRC/post/score_btc.py" --selftest 2>&1 | tail -12

echo
[ "$fail" = 0 ] && echo "MÔI TRƯỜNG SẴN SÀNG" || echo "CÒN CỬA TRƯỢT — sửa trước khi chạy pipeline"
