# Tích hợp A + P vào `submission/` — thay đổi tối thiểu

## 1. Áp 3 patch vào `solution/vt/` (áp sạch lên bản gốc, đã kiểm)

```bash
cd submission/solution/vt
patch -p0 < transfer_test_poses.py.patch   # A: --pre_save / --pre_load
patch -p0 < refiner_data.py.patch          # P: --src_depth_cache / --precompute_src_depth
patch -p0 < fuse_test.py.patch             # X: gieo hạt quantile theo pose (bắt buộc đi kèm P)
```

## 2. `pipeline.py` — 4 chỗ

```python
# hằng số, cạnh REFINER_PT
PRE_NPY   = os.path.join(WEIGHTS_DIR, "transfer_pre.npy")   # A: X1 + keypoint test, ~20 MB
SRC_DEPTH = os.path.join(WEIGHTS_DIR, "src_depth")          # P: 1 .npy / ảnh train, ~7,5 MB mỗi file

# trong run(), khối _need(...)
_need(PRE_NPY,   "precompute chuyển pose (A) — chạy chuan_bi.sh")
_need(SRC_DEPTH, "depth nguồn tính trước (P) — chạy chuan_bi.sh")

# bước ①: thêm --pre_load
_call("transfer_test_poses", ["--orig_scene", orig, "--new_scene", new,
                              "--out_scene", tr, "--pts", "kp", "--pre_load", PRE_NPY])

# bước ②: thêm --src_depth_cache vào argv của refiner_data
                                   "--apply_ch", APPLY_CH, "--apply_fp16", "1",
                                   "--src_depth_cache", SRC_DEPTH])
```

`--pre_load` bỏ qua hoàn toàn tam giác hoá, chỉ còn PnP. `--src_depth_cache` có file thì nạp thẳng,
không thì tính như cũ (an toàn nếu thiếu file lẻ).

## 3. `chuan_bi.sh` — thêm sau phần copy trọng số (cần GPU, chạy MỘT lần)

```bash
# ── A: precompute chuyển pose. Không phụ thuộc pose test — CSV nào cũng được, dùng public ──
PRE="$SUB/weights/transfer_pre.npy"; T=$(mktemp -d)
mkdir -p $T/orig/train $T/orig/test $T/new/train
ln -s "$SUB/scene/images" $T/orig/train/images; ln -s "$SUB/scene/sparse_btc" $T/orig/train/sparse
ln -s "$SUB/scene/images" $T/new/train/images;  ln -s "$SUB/scene/sparse_ba5" $T/new/train/sparse
cp "$DATA/scene/test/test_poses.csv" $T/orig/test/
( cd "$SUB/solution/vt" && python transfer_test_poses.py --orig_scene $T/orig --new_scene $T/new \
    --out_scene $T/tr --pts kp --pre_save "$PRE" ) && say "A: transfer_pre.npy" "$(du -h $PRE|cut -f1)"
rm -rf $T

# ── P: depth nguồn cho MỌI ảnh train, đúng canvas đệm + cờ depth như lúc suy luận ──
SD="$SUB/weights/src_depth"; rm -rf "$SD"
mkdir -p $T/ba5/train; ln -s "$SUB/scene/images" $T/ba5/train/images; ln -s "$SUB/scene/sparse_ba5" $T/ba5/train/sparse
( cd "$SUB/solution/vt" && TORCH_HOME="$SUB/weights/torch_hub" python refiner_data.py \
    --result_dir "$SUB/weights/gs" --scene_dir $T/ba5 --targets test --dump $T/d --no_dump 1 \
    --pad 1 --render_aa 1 --K 4 --depth_fix 1 \
    --depth_mode quant --depth_q 0.5 --depth_levels 192 --depth_smooth 40 --depth_smooth_sig 0.04 \
    --precompute_src_depth "$SD" ) && say "P: src_depth/" "$(ls $SD|wc -l) file · $(du -sh $SD|cut -f1)"
```

⚠ Cờ depth ở P **phải y hệt** `DEPTH_FLAGS` trong pipeline.py — khác một cờ là depth nguồn sai mà không báo.
⚠ P cần `scene_ba5` có `test/test_poses.csv`? Không — `--precompute_src_depth` thoát trước vòng view.
   Nhưng `--targets test` vẫn đọc CSV lúc nạp scene → tạo file rỗng có header nếu chưa có.

## 4. Nghiệm thu (theo đúng hồ sơ)

1. `python inference.py --pose_file <public csv> --output_file x.zip` chạy trọn, đủ ảnh
2. Nộp → điểm ± 0,01 so 67,2311
3. Bảng thời gian: kỳ vọng ① ~3 s, ② ~1,1 s/view
4. `PnP thất bại: 0`

Đo được ở đây (102 view /4, 404 ảnh train, cùng cờ): ① 46 → 1,2 s · ② 358 → 123 s.

---

## 5. Số đo — tất cả trên 1 card H200, scene /4 (1320×989), 102 view, 404 ảnh train, cờ y hệt thi

| Lượt | ② tổng | quantile/view | s/view | Điểm | Pixel |
|---|---|---|---|---|---|
| baseline (gốc) | 358 s | 3,34 | 3,42 | **63,9210** | — |
| baseline lượt 2 | 375 s | | | 63,9210 | **102/102 giống hệt** lượt 1 → sàn nhiễu = 0 |
| baseline seed 1 | | | | 63,9180 | → **nhiễu seed ≈ 0,003** |
| C (bỏ sync) | 357 s | | | 63,9210 | 102/102 → **đúng nhưng vô ích**, bỏ |
| P (cache, RNG cũ) | 123 s | 1,00 | 1,11 | 63,9123 | 0/102 — RNG toàn cục bị dịch |
| X (gieo hạt theo pose, không cache) | 299 s | | | 63,9147 | giá của X: **−0,006** ≈ 2× nhiễu seed |
| **P_X (cache + X, fp32)** | **112 s** | **1,00** | **1,01** | **63,9147** | **102/102 giống hệt X** ← khuyến nghị |
| P_X16 (cache fp16, 4 GB) | 116 s | | | 63,9116 | thêm −0,003 = 1× nhiễu seed |

Precompute lúc đóng gói: 404 ảnh → 358 s (0,89 s/ảnh) → **1101 ảnh ≈ 16 phút**, một lần.

## 6. Quy về thi (138 view, 1101 ảnh train, 1 card)

| | Hiện tại | Sau A + P_X |
|---|---|---|
| ① chuyển pose | 81 s | **~3 s** |
| ② nạp + vòng view | 477 s | ~10 + 138 × 1,01 ≈ **150 s** |
| ③ zip | 1 s | 1 s |
| **TỔNG** | **559 s** (biên 41 s) | **≈ 155 s** (biên ~445 s, 74 %) |

Máy chấm chậm hơn máy nhà **3,8×** vẫn kịp 600 s.

## 7. Điều phải nói thẳng

- **P bắt buộc đi kèm X.** Không có X, precompute lệch RNG so với inline (−0,009). Có X, precompute == inline
  từng bit — nhưng X tự nó đổi phép lấy mẫu quantile so với bản đã nộp 67,2311: đo được **−0,006**, cỡ
  2× nhiễu seed. Trong tiêu chí ±0,01 của hồ sơ. **Phải nộp thử một lần để xác nhận trên data thi.**
- **C không tiết kiệm gì** (357 vs 358 s). 193 lần rasterize mới là chi phí, không phải sync. Đừng áp C.
- **Dung lượng**: cache fp32 ≈ **8 GB** cho 1101 ảnh (7,5 MB/ảnh, không nén được — lossless chỉ 1,2×).
  fp16 → 4 GB, thêm −0,003. Nếu gói bị giới hạn cỡ thì fp16 là lựa chọn; nếu không, fp32.
- **Hướng B (nhiều tiến trình) không đo** — với P thì không cần, và giải pháp cam kết 1 card.
- Chưa kiểm trên **đúng scene thi** (1101 ảnh): số ảnh train nhiều hơn → cache miss nhiều hơn → P càng
  ăn hơn ở đây (hiện chỉ 3,34 lượt/view vì 404 ảnh). Chiều có lợi.
