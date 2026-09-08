# VT Track 1 — Vòng 3 (Chung khảo)

Bộ mã dự thi **AI RACE 2026 – Track 1**: Novel View Synthesis cảnh lớn từ ảnh drone.
Thư mục này **tự chứa** — mọi thứ cần để chạy từ dữ liệu BTC tới file zip nộp bài đều ở đây.

```
contest/
├── config/recipe.env      ⭐ MỌI tham số. Sửa file này = sửa cả pipeline.
├── env/                   dựng + kiểm môi trường trên máy thi
├── scripts/               chạy pipeline, đánh số theo thứ tự
├── src/                   mã nguồn
│   ├── gs/                3DGS (trainer, dataset, colmap io)
│   ├── refine/            refiner láng giềng (dump, train, apply)
│   ├── geom/              chuẩn hoá hình học (khớp dày, SfM lại)
│   ├── post/              chấm điểm, band-swap, đóng gói
│   ├── probe_data.py      đo dữ liệu ngày-1
│   └── infer_padded.py    suy luận trên canvas đệm
├── docs/                  luật · phương pháp · kế hoạch ngày thi · bẫy
├── tests/smoke.sh         chạy thử toàn dây chuyền ở quy mô tí hon
└── .claude/skills/luat-thi/   ràng buộc luật thi cho AI agent
```

---

## Chạy từ đầu trên một server mới

### 0. Server cần có sẵn

| | |
|---|---|
| GPU | ≥96 GB VRAM cho `VT_CAP=36000000`. Ít hơn thì hạ cap — `verify_env.sh` in ra mức an toàn |
| **nvcc + ninja** | gsplat biên dịch JIT lần chạy đầu. Thiếu là chết ngay, không có đường vòng |
| Ổ đĩa | ~150 GB |
| RAM | ≥128 GB (refiner nạp dataset lên bộ nhớ) |
| Mạng | cần **một lần** cho bước 2. Sau đó chạy offline được |

### 1. Lấy code (~700 KB)

```bash
git clone <repo-url> contest && cd contest
```

Mọi thứ sinh ra lúc chạy sẽ nằm trong `contest/work/` — venv, trọng số, data, kết quả.
Đừng đặt `VT_ROOT`; để mặc định thì thư mục này tự chứa và xoá `work/` là sạch hoàn toàn.

### 2. Môi trường (~15 phút, cần mạng)

```bash
bash env/setup_env.sh --with-lightglue   # venv + torch + gsplat + LightGlue
bash env/fetch_models.sh                 # 599 MB trọng số pretrained
bash env/verify_env.sh                   # PHẢI ĐẬU HẾT trước khi đi tiếp
```

`setup_env.sh` cài `torch==2.13.0` với `--index-url .../cu128`. **Nếu CUDA của server khác,
sửa dòng đó trong `env/setup_env.sh`** — đây là chỗ hay hỏng nhất trên máy lạ.

Pipeline không dùng model 3DGS hay diffusion pretrained nào. Bốn trọng số tải về đều là
mạng thị giác đa dụng: VGG16 (LPIPS — bộ chấm *và* loss refiner), RAFT (depth-fix),
SuperPoint + LightGlue (khớp dày).

### 3. Đưa data BTC vào (~4 GB)

Data **không** nằm trong git. Chuyển riêng:

```bash
mkdir -p work/data
rsync -avP <nguồn>/scene/ work/data/scene/
```

Cấu trúc phải đúng: `work/data/scene/train/{images,sparse/0}` + `work/data/scene/test/test_poses.csv`

### 4. Chạy thử tí hon (~15 phút) trước khi cam kết 9 tiếng

```bash
VT_SCENE=$PWD/work/data/scene VT_VENV=$PWD/work/venv GPU=0 bash tests/smoke.sh
```

Nó chạy đủ 6 khâu ở quy mô nhỏ và ra một file zip. Hỏng ở đây thì đừng chạy tiếp.

### 5. Đo dữ liệu — 2 giờ đầu ngày thi, CPU thuần

```bash
bash scripts/00_probe.sh
```

Đọc kỹ các dòng **➜ QUYẾT ĐỊNH**. Bốn câu nó trả lời quyết định recipe; sai một câu là
mất đúng một đòn. Sửa `config/recipe.env` theo đó — nhất là `VT_CAP` theo VRAM thật.

### 6. Chạy toàn bộ

```bash
bash scripts/run_all.sh          # ~9h trên 1 card, train TẤT CẢ từ đầu
```

Kết quả: `work/submit/submission_<ngày>.zip`, đã tự kiểm đủ ảnh / đúng tên / đúng cỡ / PNG.

Chạy nền thì nên dùng `tmux` để còn xem lại được:

```bash
tmux new -s duyet 'bash scripts/run_all.sh'   # Ctrl-B D để thoát, tmux attach -t duyet để vào lại
```

### Nếu máy thi KHÔNG có Internet

Vòng 2 đã không có. Chạy ở máy **có** mạng trước:

```bash
bash env/make_offline_bundle.sh ~/vt_offline    # ~4 GB
```

Copy cả `~/vt_offline` sang máy thi rồi làm theo `CAI_DAT.txt` trong đó.

### Chạy từng bước

| Script | Việc | Thời gian |
|---|---|---|
| `00_probe.sh` | 4 phép đo quyết định recipe | 5 phút, CPU |
| `10_geom.sh` | chuẩn hoá hình học (chỉ khi cổng trượt) | ~45 phút |
| `30_refdata.sh` | dump dữ liệu refiner (p2: từ chính model seed) | ~0,8 h |
| `31_reftrain.sh` | train refiner **trên data BTC vòng này** (bf16) | ~2 h |
| `20_train.sh` | train 3DGS, một seed mỗi card | ~105 phút/seed |
| `40_infer.sh` | suy luận canvas đệm, một tiến trình (`SEED=42`) | ~7,5 phút/seed |
| `50_bandswap.sh` | trộn tần số giữa các seed | ~1 phút |
| `60_submit.sh` | đóng gói + kiểm bài nộp | ~1 phút |

Mọi script **idempotent** — chạy lại thì bỏ qua bước đã xong. Hỏng ở đâu, chạy lại từ đó.

### Chế độ gấp

```bash
VT_SEEDS=42 VT_REF_ITERS=48000 SKIP_GEOM=1 bash scripts/run_all.sh    # ~4 h
```

---

## Ba điều phải biết trước khi sửa gì

1. **Đọc `.claude/skills/luat-thi/SKILL.md`** — luật §11 đề bài. Vi phạm là mất bài, không
   phải mất điểm. Đặc biệt: **không đưa trọng số train trên dữ liệu vòng khác vào bài nộp.**
   `31_reftrain.sh` train refiner từ đầu trên chính data vòng này, đó là lý do nó tồn tại.
2. **Đọc `docs/01_PHUONG_PHAP.md` §6** — danh sách ~40 hướng ĐÃ THỬ VÀ ĐÃ LOẠI kèm số đo.
   Đọc trước khi định thử lại bất cứ thứ gì.
3. **Điểm validation nội bộ không dự đoán được điểm test** — đã sai 7 lần. Val chỉ để chọn
   checkpoint; mọi quyết định phải chấm trên test hoặc holdout.

## Tài liệu

| File | Trả lời |
|---|---|
| [`.claude/skills/luat-thi/SKILL.md`](.claude/skills/luat-thi/SKILL.md) | Luật thi và ràng buộc cứng |
| [`docs/01_PHUONG_PHAP.md`](docs/01_PHUONG_PHAP.md) | Phương pháp: bốn giai đoạn, bằng chứng từng lựa chọn, hướng đã loại |
| [`docs/02_NGAY_1_DO_DAC.md`](docs/02_NGAY_1_DO_DAC.md) | Kế hoạch ngày thi, ngân sách giờ, câu hỏi cho BTC |
| [`docs/03_SU_CO.md`](docs/03_SU_CO.md) | Bẫy vận hành đã trả giá |

## Nguồn gốc

Phương pháp được phát triển và hiệu chuẩn từ trước ngày thi trên dữ liệu các vòng đã kết
thúc và bộ dữ liệu công khai. **Mọi trọng số đi vào bài nộp vòng này đều được train từ đầu
trên dữ liệu BTC phát cho vòng này** — `scripts/30_refdata.sh` và `scripts/31_reftrain.sh`
làm đúng việc đó, và chúng chỉ cần ảnh **train**, không cần ground-truth test.
