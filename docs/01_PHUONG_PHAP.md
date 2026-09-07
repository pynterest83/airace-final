# Phương pháp — bốn giai đoạn

Bài toán: cho ~400 ảnh drone của một khu đô thị + camera pose + sparse COLMAP,
sinh ảnh RGB tại ~100 pose chưa từng chụp. Thước đo thưởng **độ sắc nét thật**
(xem §5), nên mọi đòn làm mượt là âm.

```
dữ liệu BTC
    │
    ├─ Giai đoạn 0  CHUẨN HOÁ HÌNH HỌC        (chỉ khi cổng Sampson > 3 px)
    │      khớp dày → SfM lại với camera CÓ MÉO → kiểm trên cặp giữ lại
    │      ⤷ đòn LỚN NHẤT đã đo: +3,90 điểm
    │
    ├─ Giai đoạn 1  3DGS                       (gsplat MCMC, 3 seed)
    │      ⤷ nền ~47 → ~54 điểm
    │
    ├─ Giai đoạn 2  REFINER LÁNG GIỀNG         (UNet 4,38M trên canvas đệm)
    │      warp 2 ảnh train thật về pose đích → UNet dán texture thật vào render mờ
    │      ⤷ +6,7 điểm — khối đắt giá nhất
    │
    └─ Giai đoạn 3  BAND-SWAP 3 SEED
           LF = trung bình seed · HF = 100% một seed
           ⤷ +0,15 điểm, gần như miễn phí (CPU)
```

---

## 1. Giai đoạn 0 — chuẩn hoá hình học

**Bệnh.** Reconstruction BTC phát thường dùng camera `SIMPLE_PINHOLE` **không méo**, và
chỉ nhất quán **cục bộ**: hai ảnh chia nhiều điểm chung thì khớp ~2 px, nhưng hai ảnh cùng
nhìn một chỗ mà **không** chia điểm chung nào thì lệch 13–40 px, và sai số tăng đơn điệu từ
tâm ảnh ra góc ảnh — chữ ký kinh điển của **thiếu tham số méo**.

**Vì sao chết người.** Mỗi điểm mặt đất được ~70 ảnh nhìn thấy. 3DGS buộc phải trung bình
hoá các hình học lệch nhau hàng chục pixel ⇒ ảnh mờ ở thang 8–16 px. Không đòn nào ở tầng
mô hình cứu được, vì hệ quy chiếu đã sai.

**Cách sửa.** Khớp đặc trưng dày (SuperPoint@3072 + LightGlue) trên mọi cặp chồng phủ ≥15%,
rồi dựng lại SfM với camera `OPENCV` **tự hiệu chuẩn méo**. Chuyển pose test sang hệ mới
bằng tam giác hoá lại + PnP với keypoint test mà BTC đã cấp.

**Cổng nghiệm thu** (`scripts/10_geom.sh` bước 4) — đây là cổng **độc lập với bộ chấm**:

| Kiểm tra | Ngưỡng | Vòng trước đạt |
|---|---|---|
| Sampson trên cặp GIỮ LẠI (không đưa vào tối ưu) | < 3 px | 13,03 → **2,53** |
| Nhóm không chia điểm chung | < 4 px | → 2,98 |
| Số ảnh đăng ký | đủ 100% | 404/404 |
| Leave-one-out của phép chuyển pose test | < 3 px | 1,2 |

> ⚠ **Bỏ qua giai đoạn 0 nếu cổng đã đạt sẵn.** Phép đo tốn vài phút và quyết định 5 giờ
> công việc. Nền hình học dựng hỏng còn tệ hơn nền gốc của BTC.

**Bằng chứng đây là đòn thật, không phải mua điểm đường vòng:** ba phép đo độc lập với bộ chấm
đều cải thiện — Sampson 13,03→2,53 px; khớp quang trắc cụm 27 view 22,61→26,66 dB;
model holdout trên 101 view **chưa từng thấy** 18,87→22,58 dB. Phép thứ ba quan trọng nhất:
nó chứng minh nền mới **tổng quát hoá** tốt hơn, không phải overfit.

---

## 2. Giai đoạn 1 — 3DGS

`src/gs/trainer.py`, gsplat MCMC. Recipe ở `config/recipe.env` (`VT_GS_FLAGS`).

| Thành phần | Giá trị | Bằng chứng |
|---|---|---|
| Densify | **MCMC**, không ADC | MCMC +0,4 ở cùng cap; ADC bị batch-view làm hỏng ngưỡng gradient |
| `--cap_max` | **36M** (theo VRAM) | 12M→24M→36M cho 60,19→60,32→60,38. Bão hoà, nhưng vẫn dương |
| `--max_steps` | **15k** | 15k = 60,38; **30k = 60,02 — HẠI**. Đừng train lâu hơn |
| `--batch_views 4 --lr_scale 2` | gộp gradient 4 view rồi mới bước Adam | +0,23; thu hẹp gap train→test. B=8 không hơn |
| `--wd_weight 0.10` | Wasserstein Distortion trên feature VGG16 | +0,2 so LPIPS-crop; 0,20 không hơn 0,10 |
| `--lpips_weight 0` | tắt LPIPS trong trainer | WD-R đã gánh; LPIPS full-frame 21 MP thì OOM |
| `--depth_weight 0.05` | L1 depth vs track COLMAP | +0,3. **Tắt tay nếu track rỗng** (xem probe) |
| `--geo_consist_weight 0.5` | | +0,16…+0,21 qua refiner, và **xoá nhiễu seed** |
| `--opacity_reg 0` | | 0,01 làm sụp cảnh |

**Cái model nền làm được và không làm được.** PSNR mất ở **tần số thấp**; độ phân giải hiệu
dụng chỉ ~1/12 native. Base đang ở **trần tổng quát hoá**, không phải trần dung lượng: bớt
view thì fit train tăng (18,8→28 dB) nhưng test đứng im. Kết luận: tiền **không** nằm ở
train 3DGS tốt hơn, mà ở việc đem **texture thật** từ ảnh train vào ảnh test — tức giai đoạn 2.

---

## 3. Giai đoạn 2 — refiner láng giềng

Ý tưởng: 3DGS làm mờ texture, nhưng **ảnh train gốc có texture thật**. Warp ảnh train láng
giềng về pose đích qua depth của render, rồi để một UNet nhỏ quyết định **dán chỗ nào**.

### 3.1 Dữ liệu huấn luyện — không cần GT test

Train 4 model 3DGS, mỗi model `--holdout_every 4 --holdout_offset O`, O ∈ {0,1,2,3}.
Mỗi model giấu 1/4 view train; 4 model phủ đủ 100%. View bị giấu = model **chưa thấy**
nhưng ta **có ảnh thật** ⇒ cặp (render lỗi thật, ảnh thật) ở đúng phân bố lỗi của pose test.

> Đây là lý do pipeline này **tự chứa và hợp lệ**: refiner học từ ảnh train của chính vòng
> thi, không đụng một pixel GT test nào, không mượn dữ liệu vòng khác. Xem skill `luat-thi`.

### 3.2 Dump mỗi view đích (`src/refine/refiner_data.py`)

1. **Rasterize** tại pose đích → RGB, expected depth, alpha.
2. **Chọn K nguồn train**: lọc góc trục quang < 25°, lọc overlap ≥ 40%, xếp theo baseline tăng dần.
   **K = 4** (mặc định hiện tại). ⚠ Kết luận cũ "K ≥ 3 là âm" đo trên nền hình học CŨ và **đã bị lật**:
   trên nền SfM mới, đổi refiner K2 → K4 cho **+0,74 điểm test** (đo paired trên 2 fold: +0,74 / +0,75).
3. **Depth-fix (GT-free)** — đòn tinh tế nhất. Hai warp từ hai ảnh *thật* lệch nhau median
   34 px vì depth render sai. Thay vì dịch 2D, giải thẳng sai số độ sâu: RAFT ở 1/4 res giữa
   hai warp → `f01`; Jacobian giải tích `k_s = ∂(u,v)/∂z`; mô hình tuyến tính
   `f01 = (k0 − k1)·Δz` ⇒ `Δz = ((k0−k1)·f01)/|k0−k1|²`; làm trơn σ=8, kẹp |Δz|/z ≤ 0,15,
   một vòng. Rồi **warp lại cả hai nguồn** bằng depth đã sửa. **+0,49 điểm.**
   (Lặp 2–3 vòng thì phân kỳ. Một vòng thôi.)
4. **Warp + căn tile ±4 px + mask** che khuất. Thiếu nguồn → `warp = render, mask = 0`.

### 3.3 Mạng (`src/refine/refiner_train.py`)

- Vào **13 kênh**: `[render(3) | warp0(3) | warp1(3) | mask0 | mask1 | depth8 | alpha]`.
- **UNet 4 tầng** ch 48-96-192-384, **4,38M tham số** (17 MB). Không transformer, không
  attention, không align học được — đều đã thử, đều ≤ 0. Đầu vào đã pixel-aligned nên conv là đủ.
- **Đầu ra có cổng**: 4 kênh = residual(3) + gate(1, sigmoid);
  `out = clamp(render + gate·residual, 0, 1)`. Mạng chỉ sửa chỗ nó tin.
- Loss = `0,6·L1 + 0,3·(1−SSIM) + 0,6·LPIPS-VGG`, cùng quy ước bộ chấm.
- **Source-dropout 0,3**: xác suất 0,3 thay một nguồn bằng render+mask 0. Vừa regularize
  (+0,14) vừa cho suy giảm êm khi thiếu láng giềng.

### 3.4 Canvas đệm khi suy luận — **+0,73 điểm**, và chạy trong MỘT tiến trình

Rasterize trên canvas rộng hơn ảnh (đệm ~448 px mỗi phía), chạy refiner ở đó, rồi mới ánh xạ
ngược về khung có méo bằng `cv2.remap`. Lý do: ở khung thường, vùng góc ảnh bị cắt cụt nên
refiner không có ngữ cảnh để làm việc. Đệm xong thì góc ảnh cũng được refine.

Toàn bộ chuỗi này chạy **trong một tiến trình** (`refiner_data.py --pad 1 --no_dump 1 --apply_ckpt`),
không ghi PNG trung gian: **4,4 s/view ≈ 7,5′/seed**, so với 41′/seed của đường 3 bước cũ — điểm y hệt.
Thêm `--render_aa 1` để kênh render dùng rasterize antialiased, khớp với base train `--antialiased 1` (+0,03).

---

## 4. Giai đoạn 3 — band-swap theo tần số

`out = LF_mean + (HF_a − LF_a)`, `LF = GaussianBlur(σ=8)`.

- **LF = trung bình nhiều seed** — nhiễu cấu trúc thô triệt tiêu nhau, PSNR +0,14 dB.
- **HF = 100% từ MỘT seed** — trung bình băng cao làm **lapvar tụt 42%**: chi tiết của các
  member triệt tiêu nhau, và thước phạt nặng qua LPIPS/SSIM.

Đây là luật chung: *distortion và perception sống ở hai băng tần khác nhau; trung bình hoá ở
băng thấp thì cộng dồn, ở băng cao thì triệt tiêu.* σ=8 > σ=4. Ensemble bão hoà ở 3 member.

---

## 5. Thước đo quyết định điều gì đáng làm

```
Score = 0.4·(1 − LPIPS) + 0.3·SSIM + 0.3·clamp(PSNR/50, 0, 1)
Quy đổi: +1 dB PSNR = +0,6đ · −0,01 LPIPS = +0,4đ · +0,01 SSIM = +0,3đ
```

Bản đồ trần điểm đo trên vòng trước: fit/capacity ≈ 10đ (túi lớn nhất) · tổng quát hoá ≈ 3đ ·
**thêm data cùng phân bố: +11% ảnh chỉ được +0,15 ⇒ đừng đốt giờ đi xin thêm ảnh.**
Metric thưởng sắc nét tới tận 89,6 ⇒ **mọi đòn làm mượt là âm.**

⚠ **Dấu của một đòn phụ thuộc độ phân giải của thước.** Cùng một phép biến đổi cho dấu **âm**
ở /4 và **dương** ở native. Chọn res của thước trước khi tin dấu của bất kỳ kết quả nào.

---

## 6. Đã thử và ĐÃ LOẠI — đừng thử lại

| Nhóm | Đã đóng |
|---|---|
| 3DGS nền | cap > 36M, train > 15k, ADC, SH-reg, bilagrid, pose-opt (4 kiểu), coarse-to-fine, Mip-Splatting, partition 2×2, hierarchical, prune hậu kỳ, chuyên gia cục bộ, distill-back ảnh refined, oversample, mono-depth, **absgrad (hại mọi ô)** |
| Nguồn / warp | K = 3…5, xếp hạng nguồn, forward-splat, mono-fwd, plane-sweep, MVS COLMAP, RAFT căn về render, AlignNet, CorrAlign, tile ±48 px, coverage-K6 |
| Refiner | ch32/ch64, UNet 5 tầng, transformer, crop 768, ½ res, coarse-to-fine 2 thang, cond, w_lpips ≠ 0,6, GAN, EMA, hard-mining, **TTA (−0,4)**, fine-tune thêm |
| Sau refiner | fusion tay, diffusion (SD-Turbo, Difix), ensemble > 3 seed, pool > 12 member, chọn member theo view |

**Quy luật đã lặp 6 lần:** mọi cải thiện ở tầng 3DGS thuộc loại "diện mạo" đều bị refiner
**hấp thụ gần hết** (ensemble 7 model: thô +0,8 → sau refiner +0,17). Ngoại lệ duy nhất là
**sửa hình học**: thô +5,71 → sau refiner **+6,73**. Nó *nở ra* vì cải thiện đúng thứ refiner
cần — chất lượng depth dùng để warp.

Hệ quả thực dụng: **đừng đánh giá một đòn ở tầng 3DGS bằng điểm thô.** Luôn đo sau refiner.
