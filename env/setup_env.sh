#!/bin/bash
# setup_env.sh — dựng môi trường trên MÁY MỚI HOÀN TOÀN.
# Mọi thứ cài vào contest/work/ — không đụng gì ngoài thư mục này.
#
#   bash env/setup_env.sh                 # môi trường cơ bản
#   bash env/setup_env.sh --with-lightglue  # + LightGlue (cần cho giai đoạn 0)
#   PIP_OFFLINE=<thư mục wheels> bash env/setup_env.sh   # máy không có Internet
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# Lấy Python hệ thống trước khi recipe.env đưa một venv cũ lên đầu PATH.
SYSTEM_PYTHON="${SYSTEM_PYTHON:-$(command -v python3)}"
source "$HERE/../config/recipe.env"
WITH_LG=0; [ "${1:-}" = "--with-lightglue" ] && WITH_LG=1

echo "═══ 1. kiểm máy ═══"
"$SYSTEM_PYTHON" -V || { echo "  ✘ không có python3"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null | sed 's/^/  /' || echo "  ⚠ KHÔNG THẤY GPU"
command -v nvcc >/dev/null && echo "  ✔ nvcc $(nvcc --version | grep -oE 'release [0-9.]+')" \
  || echo "  ⚠ KHÔNG THẤY nvcc — gsplat JIT sẽ hỏng. Thêm /usr/local/cuda/bin vào PATH."
echo "  ổ đĩa: $(df -h "$VT_CODE" | tail -1 | awk '{print $4}') trống  (cần ≥150 GB)"
echo "  RAM:   $(free -g | awk '/^Mem:/{print $2}') GB"

echo; echo "═══ 2. venv: $VT_VENV ═══"
[ -x "$VT_VENV/bin/python" ] || "$SYSTEM_PYTHON" -m venv "$VT_VENV"
PY="$VT_VENV/bin/python"
PIPARGS=(); [ -n "${PIP_OFFLINE:-}" ] && PIPARGS=(--no-index --find-links "$PIP_OFFLINE")

# Một lần `python -m venv` thất bại (thường do thiếu python3.x-venv trên
# Ubuntu) vẫn có thể để lại thư mục và interpreter, nhưng không có pip. Đừng
# coi venv nửa chừng đó là hợp lệ.
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "  ⚠ venv tồn tại nhưng thiếu pip; đang thử sửa bằng ensurepip..."
  if ! "$PY" -m ensurepip --upgrade; then
    PY_MM="$($SYSTEM_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    echo "  ✘ Python này không có ensurepip. Trên Ubuntu, chạy:"
    echo "      sudo apt-get install python${PY_MM}-venv"
    echo "    rồi chạy lại chính lệnh setup_env.sh (không cần xóa venv)."
    exit 1
  fi
fi
"$PY" -m pip install -q --upgrade pip "${PIPARGS[@]}"

echo; echo "═══ 3. torch (khớp CUDA của máy) ═══"
if "$PY" -c "import torch" 2>/dev/null; then
  echo "  đã có: $("$PY" -c 'import torch;print(torch.__version__, torch.version.cuda)')"
else
  # Wheel CUDA 12.8 mới nhất hiện có trên index chính thức của PyTorch.
  if [ -n "${PIP_OFFLINE:-}" ]; then
    "$PY" -m pip install "${PIPARGS[@]}" torch==2.10.0 torchvision==0.25.0
  else
    "$PY" -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
  fi
fi

echo; echo "═══ 4. phần còn lại ═══"
"$PY" -m pip install "${PIPARGS[@]}" -r "$HERE/requirements.txt"

if [ "$WITH_LG" = 1 ]; then
  echo; echo "═══ 5. LightGlue (không có trên pip) ═══"
  if [ ! -d "$VT_LIGHTGLUE/lightglue" ]; then
    mkdir -p "$(dirname "$VT_LIGHTGLUE")"
    if [ -n "${PIP_OFFLINE:-}" ]; then
      echo "  ⚠ chế độ offline — copy sẵn thư mục lightglue vào $VT_LIGHTGLUE"
    else
      git clone --depth 1 https://github.com/cvg/LightGlue "$VT_LIGHTGLUE"
    fi
  fi
  if [ -d "$VT_LIGHTGLUE/lightglue" ]; then
    "$PY" -m pip install "${PIPARGS[@]}" -e "$VT_LIGHTGLUE"
  fi
  if PYTHONPATH="$VT_LIGHTGLUE" "$PY" -c "import lightglue" 2>/dev/null; then
    echo "  ✔ lightglue nạp được"
  else
    echo "  ✘ lightglue KHÔNG nạp được — giai đoạn 0 sẽ không chạy"
    exit 1
  fi
fi

echo; echo "═══ XONG ═══"
echo "  Tiếp theo:  bash env/fetch_models.sh   # tải trọng số pretrained (~600 MB)"
echo "              bash env/verify_env.sh     # kiểm mọi cửa"
