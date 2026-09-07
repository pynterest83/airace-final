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

## Trước ngày thi — đóng gói offline (BẮT BUỘC)

Vòng 2 máy thi **không có Internet**. LPIPS-VGG (553 MB), RAFT, SuperPoint/LightGlue đều
tải từ mạng lần đầu dùng. Chạy ở máy này, lúc **còn** mạng:

```bash
bash env/make_offline_bundle.sh ~/vt_offline    # ~4 GB
```

Copy cả `~/vt_offline` sang máy thi, làm theo `CAI_DAT.txt` trong đó.

## Bắt đầu

```bash
# 1. Trỏ vào dữ liệu BTC và dựng môi trường
export VT_ROOT=$HOME/vt-contest
mkdir -p $VT_ROOT/data && cp -r <thư mục data BTC> $VT_ROOT/data/scene

bash env/setup_env.sh
bash env/verify_env.sh          # PHẢI đậu hết trước khi làm gì tiếp

# 2. Đo dữ liệu — 2 giờ đầu, CPU thuần, quyết định recipe
bash scripts/00_probe.sh        # đọc kỹ các dòng ➜ QUYẾT ĐỊNH

# 3. Sửa config/recipe.env theo kết quả probe (nhất là VT_CAP theo VRAM thật)

# 4. Chạy
bash scripts/run_all.sh          # ~10h trên 1 card, train tất cả từ đầu
```

Kết quả: `$VT_ROOT/submit/submission_<ngày>.zip`, đã tự kiểm đủ ảnh / đúng tên / đúng cỡ.

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
