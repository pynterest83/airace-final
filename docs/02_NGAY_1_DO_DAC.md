# Ngày thi — 2 giờ đầu và kế hoạch giờ

> 1,5 ngày **không đủ để phát minh**. Trong phòng thi chỉ làm được *chạy · đo · viết*.
> Mọi phần sáng tạo phải xong trước. Tiền lệ: mọi đòn thắng đều có sẵn từ tổng duyệt;
> mọi thứ nghĩ ra dưới áp lực deadline đều lỗ.

## 0. Ba mươi phút đầu — dựng máy, không đụng thuật toán

```bash
export VT_ROOT=$HOME/vt-contest
mkdir -p $VT_ROOT/data && cp -r <data BTC> $VT_ROOT/data/scene

bash env/setup_env.sh          # thêm --with-lightglue nếu định chạy giai đoạn 0
bash env/verify_env.sh         # PHẢI đậu hết trước khi làm gì tiếp
```

`verify_env.sh` in ra `cap_max an toàn` theo VRAM thật của máy. **Sửa `VT_CAP` trong
`config/recipe.env` theo số đó**, đừng để mặc định 36M nếu card nhỏ hơn.

⚠ Vòng trước máy thi **không có Internet**. Nếu lần này cũng vậy thì LPIPS-VGG, RAFT và
mọi wheel phải mang theo sẵn. Kiểm ngay trong 30 phút này, đừng để phát hiện lúc 2 giờ sáng.

## 1. Hai giờ đầu — bốn phép đo, CPU thuần, 0 GPU

```bash
bash scripts/00_probe.sh
```

Chạy **song song** với một baseline nhỏ đã phóng sẵn trên GPU. Bốn câu, và sai câu nào thì
mất đúng đòn tương ứng:

| # | Câu hỏi | Quyết định | Sai thì mất gì |
|---|---|---|---|
| 1 | Camera còn méo không? Tâm quang test có bị ép về W/2,H/2? | `--test_use_train_K`, có cần prewarp không | ảnh còn méo mà bỏ qua ⇒ mất trục prewarp/redistort |
| 2 | `points3D` có track 2D thật không? | `--depth_weight` | track rỗng mà vẫn bật ⇒ **null-test**, đã dính một lần |
| 3 | Test **xen kẽ trong chuyến bay** hay **ngoại suy theo vùng**? | có dùng refiner láng giềng không | ngoại suy ⇒ refiner TEO, đừng dồn giờ vào warp |
| 4 | Hình học BTC nhất quán không (Sampson)? | có chạy giai đoạn 0 không | bỏ qua khi cần ⇒ mất tới +3,9 điểm |

**Câu 3 là câu đáng tiền nhất.** Toàn bộ +6,7 điểm của refiner đứng trên giả định "mỗi pose
test luôn có ≥2 ảnh train láng giềng gần". Nếu vòng 3 chia test theo **vùng** thay vì xen kẽ,
giả định đó sập và kế hoạch phải đổi ngay trong giờ đầu — dồn vào dung lượng 3DGS và prior
hình học thay vì warp.

**Câu 4 quyết định 5 giờ công việc** và chỉ tốn vài phút để đo.

## 2. Ngân sách giờ (1 card; nhiều card thì chia seed ra chạy song song)

| Bước | Thời gian | Bắt buộc |
|---|---|---|
| Dựng môi trường + probe | 1 h | **Có** |
| Giai đoạn 0 — khớp dày | **~20′/card** (I/O ảnh chiếm phần lớn; khớp chỉ 0,03 s/cặp) | chỉ khi cổng trượt |
| Giai đoạn 0 — SfM GLOMAP | **~25′** CPU (incremental cũ: 75′) | chỉ khi cổng trượt |
| 4 model holdout + dump | ~5 h (song song 4 card ⇒ ~1,5 h) | **Có** (để train refiner) |
| Train refiner (bf16) | **~2 h** (fp32: 3,2 h) | **Có** |
| Train 3DGS, mỗi seed | ~85–105′ | **Có** |
| Suy luận, mỗi seed | **~7,5′** (đường 3 bước cũ: 41′) | **Có** |
| Band-swap + đóng gói | ~12′ | chỉ khi có nhiều seed |

**Trên 1 card, train tất cả từ đầu** (đúng cấu hình tham khảo §13 đề bài):

| Cấu hình | 1 card |
|---|---|
| `VT_REFDATA_MODE=separate` (4 model holdout riêng) | ~18h |
| **`VT_REFDATA_MODE=p2` (mặc định)** | **~10h** |
| p2 + `VT_SFM_MODE=glomap` | ~9h |

Ba dòng trên là ba lựa chọn **thay thế nhau**, không cộng vào nhau — cùng một luồng đầy đủ,
khác nhau ở chỗ áp bao nhiêu đòn tối ưu. Trên nhiều card thì chia cho số card ở các bước
train song song được (train seed, dump); `31_reftrain` không chia được.

### Đường gấp (nếu vỡ kế hoạch)

`VT_SEEDS=42 VT_REF_ITERS=48000 VT_STEPS=12000 SKIP_GEOM=1 bash scripts/run_all.sh`
→ ~2,5 giờ. Giá đã đo: 12k bước thay 15k = **−0,22 điểm** (đổi lấy −20…40′);
refiner 48k thay 192k ≈ −0,7; bỏ band-swap ≈ −0,25. Vẫn ra bài nộp hợp lệ.

**Luôn có một bài nộp hợp lệ trong tay sớm.** Chạy `60_submit.sh` trên render 3DGS thô ngay
khi có nó, rồi mới đi cải thiện. Điểm thấp còn hơn không nộp được.

## 3. Nếu vòng 3 KHÔNG cấp ground-truth test

Nhiều khả năng không có. Khi đó **không chấm local được** và phải quay lại chế độ holdout:

- Để `VT_GT_DIR` trống. `score()` sẽ nhắc dùng proxy thay vì im lặng bỏ qua.
- Đo bằng **`score_holdout`** (`src/post/score_holdout.py`): chấm đúng thước BTC trên view
  holdout trong dump, nơi `gt_und.png` là ảnh thật. Gọi tay:

  ```bash
  source scripts/lib.sh
  score_holdout H_thô  $VT_RUNS/refdata_s42                 # render thô
  score_holdout H_ref  $VT_RUNS/refdata_s42  <thư_mục_refined>   # sau refiner
  ```

  `30_refdata.sh` tự gọi nó khi `VT_GT_DIR` rỗng.
**Luật proxy đã đo (paired, 2 fold — dùng đúng luật này):**

| Loại quyết định | Holdout dùng được không |
|---|---|
| BASE / hình học / SfM / pose | **Chuyển 1:1.** So điểm **thô** trên holdout; lệch holdout−test ổn định +1,9…+2,4 |
| REFINER | **Phóng đại ~4×.** ΔH +2,96 → Δtest +0,74 (fold khác: +3,00 → +0,76). Chỉ tin **DẤU**, chia độ lớn cho 4, và chỉ tin khi ΔH > ~1 |
| Điểm tuyệt đối | Không suy ra được — hiệu chuẩn bằng 1 lần nộp sớm (bản 1 seed) rồi cộng offset |

- ⚠ **Đừng dùng VAL của refiner** (crop trên dump, cùng cảnh) — đã 7 lần không dự đoán test.
  Proxy hợp lệ là **điểm BTC đủ khung trên view holdout chưa từng vào refiner**.
- Chỉ cần chấm **24 view chọn đều theo tên**, không cần đủ 102: tái tạo HIỆU điểm giữa hai
  cấu hình với RMS 0,13 (ρ xếp hạng 0,996) — nhanh gấp 4.

## 4. Danh sách hỏi BTC ngay khi vào phòng

Rẻ nhất và đắt tiền nhất — 0 giây GPU, sai thì lệch cả điểm lẫn thứ hạng đòn:

1. **`psnr_max` bằng bao nhiêu?** LPIPS chạy backbone nào, `normalize` True hay False?
   (Ta đang giả định 50 / vgg / False — suy ngược từ vòng trước, **chưa được BTC xác nhận**.)
2. Có cấp ground-truth của một phần test không?
3. Quy mô: bao nhiêu ảnh, mấy scene, độ phân giải, có COLMAP sparse không, có track 2D-3D không?
4. Camera: đã khử méo hay còn méo? Một hay nhiều camera model? Nhiều chuyến bay?
5. Chấm: trọng số giữa leaderboard và báo cáo/thuyết trình? Bao nhiêu lượt nộp?
6. Môi trường: mấy GPU, VRAM bao nhiêu, **có Internet không**, được mang code/checkpoint sẵn không?

> Câu 6 quan trọng về mặt luật: nếu được mang checkpoint sẵn, **vẫn không** được mang trọng số
> train trên dữ liệu vòng khác vào bài nộp (§11.1). Xem skill `luat-thi`.
