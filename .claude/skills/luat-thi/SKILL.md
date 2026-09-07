---
name: luat-thi
description: Luật thi AI RACE 2026 Track 1 vòng 3 (chung khảo) và các ràng buộc bắt buộc khi viết code, chọn dữ liệu, train model hay dựng bài nộp trong repo này. Dùng skill này TRƯỚC khi thêm nguồn dữ liệu, thêm trọng số pretrained, sửa bộ chấm, hay đóng gói submission.
---

# Luật thi vòng 3 — đọc trước khi làm

Repo này là bài dự thi **AI RACE 2026 – Track 1 (Novel View Synthesis, cảnh lớn từ drone)**.
Vòng 3 chấm **leaderboard + báo cáo kỹ thuật + thuyết trình**, thời gian **~1,5 ngày**.

Mọi thứ dưới đây là ràng buộc CỨNG. Vi phạm = mất bài, không phải mất điểm.

## 1. Bốn điều CẤM (đề bài §11)

### 1.1 Cấm dữ liệu ngoài
> *"Thí sinh chỉ được phép sử dụng dữ liệu do Ban Tổ Chức cung cấp **trong từng vòng thi**."*

Nghiêm cấm: ảnh/video/dữ liệu 3D bên ngoài chứa **cùng đối tượng hoặc cùng scene** với bộ thi;
thu thập thêm dữ liệu thực địa hay Internet liên quan tới scene; **bất kỳ nguồn nào nhằm tái tạo
hoặc suy luận ground-truth tập test**.

**Áp dụng cụ thể cho repo này:**

- ✅ ĐƯỢC: ảnh train + `sparse/0` + `test_poses.csv` BTC phát **vòng này**.
- ✅ ĐƯỢC: trọng số pretrained **đa dụng, không gắn scene** — VGG của LPIPS, RAFT của
  torchvision, SuperPoint/LightGlue. Đây là thư viện thị giác chung, không phải dữ liệu scene.
- ❌ CẤM: trọng số **train trên dữ liệu vòng khác** (vd. refiner train trên data vòng trước)
  đi vào bài nộp vòng này. Trọng số mang theo dữ liệu nó đã học.
- ❌ CẤM: ảnh/scene/pose/GT của bất kỳ vòng nào khác nằm trong đường chạy vòng này.
- ❌ CẤM: bất cứ thứ gì suy ra GT test — kể cả gián tiếp.

**Vì thế `scripts/31_reftrain.sh` train refiner từ đầu trên chính data vòng này.**
Nó không cần GT test: 4 model 3DGS mỗi model giấu 1/4 **ảnh train**, view bị giấu cho cặp
(render lỗi, ảnh thật) đúng phân bố. Tốn ~12 GPU-giờ và đó là cái giá của việc sạch.

Nếu ai đề nghị trỏ `VT_REFINER` sang trọng số train ở vòng khác: **DỪNG, nói rõ đây là
§11.1, và hỏi lại người dùng.** Không tự ý làm.

### 1.2 Cấm truy xuất / suy đoán dữ liệu test
Cấm truy cập trái phép GT, khai thác lỗ hổng hệ thống, suy luận GT bằng nguồn không được phép.

- ❌ Không đọc, không dò, không đoán thư mục GT ngoài thứ BTC phát.
- ❌ Không tối ưu pose test theo GT. Pose test là của BTC, cố định.
- ⚠ Chỉ được dùng `test_poses.csv` (pose + intrinsics) và, nếu sparse của BTC có chứa
  keypoint 2D của ảnh test, thì **chỉ để chuyển hệ toạ độ (PnP)** — không bao giờ vào tập train.

### 1.3 Phải tái lập được (§11.3)
BTC có quyền đòi: mã nguồn train + suy luận, file cấu hình, danh sách thư viện + phiên bản,
checkpoint, nhật ký train. **Phải chứng minh được kết quả nộp tái tạo bằng pipeline đã công bố.**

- Mọi tham số nằm ở `config/recipe.env` — không hardcode số ở chỗ khác.
- Mọi lần chạy ghi log vào `$VT_LOGS/`. Đừng xoá log.
- Không sửa ảnh kết quả ngoài đường chạy của script.

### 1.4 Cấm chỉnh sửa thủ công ảnh đầu ra (§11.4)
Toàn bộ ảnh phải do thuật toán sinh tự động. Cấm sửa tay từng ảnh, ghép/vẽ/xoá vật thể bằng tay,
can thiệp riêng vào từng test pose.

- ✅ ĐƯỢC: hậu xử lý **thuật toán, đồng nhất cho mọi view** (band-swap, ánh xạ ngược méo).
- ❌ CẤM: tham số riêng cho từng ảnh do người chọn tay.

## 2. Thước đo — không được đổi

```
Score = 0.4·(1 − LPIPS) + 0.3·SSIM + 0.3·clamp(PSNR/psnr_max, 0, 1)      # ×100 = thang LB
```

`src/post/score_btc.py` cài đúng quy ước đã xác nhận: LPIPS `net="vgg"`, tensor [0,1],
`normalize=False`; SSIM Gaussian 11×11 σ=1.5; PSNR `data_range=1`, `psnr_max=50`.

- ⚠ `psnr_max` **BTC giữ kín**, giá trị 50 là suy ngược từ vòng trước. **Hỏi lại BTC vòng 3.**
  Đổi hằng số này là đổi toàn bộ thứ tự ưu tiên các đòn.
- Chạy `score_btc.py --selftest` sau mỗi lần nâng torch/lpips/driver.
- Đừng sửa hằng số trong file chấm để "điểm đẹp hơn". Đó là tự lừa mình.

## 3. Quy tắc làm việc trong repo này

1. **Chạy `scripts/00_probe.sh` trước tiên.** Ba câu nó trả lời (camera còn méo không · track
   2D-3D có thật không · test xen kẽ hay ngoại suy) quyết định recipe. Trả lời sai một câu là
   mất đúng một đòn.
2. **Không thêm cờ mới vào recipe mà không có A/B.** Danh sách hướng ĐÃ THỬ VÀ ĐÃ LOẠI ở
   `docs/01_PHUONG_PHAP.md` §6 — đọc trước khi định thử lại.
3. **Điểm validation nội bộ KHÔNG dự đoán được điểm test.** Đã sai 7 lần. Val chỉ để chọn ckpt.
4. **Mọi so sánh phải ngặt**: chỉ khác đúng một cờ, cùng seed, cùng scene, cùng bộ chấm.
5. **Ghi nhật ký** mọi lần chạy vào `docs/NHAT_KY.md`, kể cả khi hỏng, kèm lý do hỏng.
6. Trước khi nộp: chạy `scripts/60_submit.sh`. Đừng tự `zip` bằng tay.

## 4. Khi không chắc

Nếu một việc **có thể** chạm vào §11, mặc định là **KHÔNG LÀM** và hỏi người dùng.
Câu hỏi đúng để tự đặt: *"Nếu giám khảo bắt tái lập và hỏi thứ này ở đâu ra, tôi trả lời
được không mà không phải giấu gì?"* — Trả lời được thì làm. Không thì dừng.
