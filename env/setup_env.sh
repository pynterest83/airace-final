#!/bin/bash
# setup_env.sh — dựng môi trường trên MÁY MỚI HOÀN TOÀN.
# Mọi thứ cài vào contest/work/ — không đụng gì ngoài thư mục này.
#
#   bash env/setup_env.sh                 # môi trường cơ bản
#   bash env/setup_env.sh --with-lightglue  # + LightGlue (cần cho giai đoạn 0)
#   PIP_OFFLINE=<thư mục wheels> bash env/setup_env.sh   # máy không có Internet
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../config/recipe.env"
WITH_LG=0; [ "${1:-}" = "--with-lightglue" ] && WITH_LG=1

echo "═══ 1. kiểm máy ═══"
python3 -V || { echo "  ✘ không có python3"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null | sed 's/^/  /' || echo "  ⚠ KHÔNG THẤY GPU"
command -v nvcc >/dev/null && echo "  ✔ nvcc $(nvcc --version | grep -oE 'release [0-9.]+')" \
  || echo "  ⚠ KHÔNG THẤY nvcc — gsplat JIT sẽ hỏng. Thêm /usr/local/cuda/bin vào PATH."
echo "  ổ đĩa: $(df -h "$VT_CODE" | tail -1 | awk '{print $4}') trống  (cần ≥150 GB)"
echo "  RAM:   $(free -g | awk '/^Mem:/{print $2}') GB"

echo; echo "═══ 2. venv: $VT_VENV ═══"
[ -d "$VT_VENV" ] || python3 -m venv "$VT_VENV"
PY="$VT_VENV/bin/python"
PIPARGS=""; [ -n "${PIP_OFFLINE:-}" ] && PIPARGS="--no-index --find-links $PIP_OFFLINE"
"$PY" -m pip install -q --upgrade pip $PIPARGS

echo; echo "═══ 3. torch (khớp CUDA của máy) ═══"
if "$PY" -c "import torch" 2>/dev/null; then
  echo "  đã có: $("$PY" -c 'import torch;print(torch.__version__, torch.version.cuda)')"
else
  # Đổi cu128 cho khớp driver máy thi. Bản đã kiểm: torch 2.13.0 / torchvision 0.28.0.
  if [ -n "${PIP_OFFLINE:-}" ]; then
    "$PY" -m pip install $PIPARGS torch==2.13.0 torchvision==0.28.0
  else
    "$PY" -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu128
  fi
fi

echo; echo "═══ 4. phần còn lại ═══"
"$PY" -m pip install $PIPARGS -r "$HERE/requirements.txt"

if [ "$WITH_LG" = 1 ]; then
  echo; echo "═══ 5. LightGlue (không có trên pip) ═══"
  if [ ! -d "$VT_LIGHTGLUE/lightglue" ]; then
    mkdir -p "$(dirname "$VT_LIGHTGLUE")"
    if [ -n "${PIP_OFFLINE:-}" ]; then
      echo "  ⚠ chế độ offline — copy sẵn thư mục lightglue vào $VT_LIGHTGLUE"
    else
      git clone --depth 1 https://github.com/cvg/LightGlue "$VT_LIGHTGLUE" \
        && "$PY" -m pip install $PIPARGS -e "$VT_LIGHTGLUE" \
        || echo "  ✘ không cài được LightGlue — giai đoạn 0 sẽ không chạy"
    fi
  fi
  PYTHONPATH="$VT_LIGHTGLUE" "$PY" -c "import lightglue" 2>/dev/null \
    && echo "  ✔ lightglue nạp được" || echo "  ✘ lightglue KHÔNG nạp được"
fi

echo; echo "═══ XONG ═══"
echo "  Tiếp theo:  bash env/fetch_models.sh   # tải trọng số pretrained (~600 MB)"
echo "              bash env/verify_env.sh     # kiểm mọi cửa"
