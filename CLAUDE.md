# Hướng dẫn cho AI agent làm việc trong thư mục này

Đây là bài dự thi **AI RACE 2026 – Track 1 vòng 3**. Đọc kỹ trước khi sửa gì.

## Bắt buộc đọc trước

1. **`.claude/skills/luat-thi/SKILL.md`** — luật thi. Bốn điều cấm ở §11 đề bài.
   **Dùng skill này trước khi** thêm nguồn dữ liệu, thêm trọng số pretrained, sửa bộ chấm,
   hay đóng gói submission.
2. **`docs/01_PHUONG_PHAP.md`** — phương pháp và §6 danh sách hướng đã loại kèm số đo.
   Đừng đề xuất lại thứ đã có kết luận âm trong đó.
3. **`docs/03_SU_CO.md`** — bẫy vận hành đã trả giá.

## Ranh giới cứng

- ⛔ **Không** đưa dữ liệu, scene, pose hay **trọng số train ở vòng thi khác** vào đường chạy
  vòng này. Trọng số mang theo dữ liệu nó đã học. Nếu ai đề nghị trỏ `VT_REFINER` sang trọng
  số của vòng khác: **dừng, nói rõ đây là §11.1, hỏi lại người dùng.**
- ⛔ **Không** đọc/dò/đoán ground-truth test ngoài thứ BTC phát. Không tối ưu pose test theo GT.
- ⛔ **Không** sửa hằng số trong `src/post/score_btc.py` để điểm đẹp hơn. Đó là tự lừa mình.
- ⛔ **Không** sửa ảnh kết quả bằng tay hay bằng tham số chọn riêng cho từng ảnh (§11.4).
- ⛔ **Không** dùng `/tmp` cho dữ liệu chạy — máy khởi động lại là mất. Dùng `$VT_ROOT`.
- ⛔ **Không** `pkill -f` theo tên script — đã tự giết phiên làm việc 5 lần. Kill theo PID.

## Quy ước trong repo

- **Mọi tham số ở `config/recipe.env`.** Đừng hardcode số vào script hay code.
  Cần một giá trị mới thì thêm biến `VT_*` ở đó, có kèm dòng bình luận nói *vì sao* giá trị đó.
- **Script đánh số theo thứ tự chạy** và đều idempotent (ghi `$VT_RUNS/.done_<tag>`).
  Thêm bước mới thì giữ đúng quy ước này.
- **Mọi bình luận và tài liệu viết bằng tiếng Việt**, giống phần còn lại của repo.
- Code trong `src/` là mã đã chạy được và đã đo — **sửa tối thiểu**, mỗi thay đổi phải kèm
  lý do đo được. Đây không phải chỗ dọn dẹp cho đẹp.

## Khi được nhờ chạy thí nghiệm

1. So sánh phải **ngặt**: chỉ khác đúng một cờ, cùng seed, cùng scene, cùng bộ chấm.
2. **Đăng ký ngưỡng trước khi chạy** — "đòn này phải hơn ≥ X điểm mới đi tiếp". Không thì
   sẽ tự thuyết phục mình rằng +0,03 là có ý nghĩa.
3. Đòn ở tầng 3DGS phải đo **sau refiner**, không đo bằng điểm thô — refiner hấp thụ gần hết
   cải thiện loại "diện mạo" (đã lặp 6 lần).
4. Ghi kết quả vào `docs/NHAT_KY.md`, **kể cả khi hỏng**, kèm lý do hỏng.

## Khi không chắc

Mặc định là **không làm** và hỏi người dùng. Câu tự hỏi:
*"Nếu giám khảo bắt tái lập và hỏi thứ này ở đâu ra, tôi trả lời được không mà không phải giấu gì?"*
Trả lời được thì làm. Không thì dừng.
