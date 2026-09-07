# Bẫy vận hành — đã trả giá rồi, đừng trả lại

| Bẫy | Hậu quả thật | Cách tránh |
|---|---|---|
| `pkill -f "<mẫu>"` khi mẫu nằm trong chính lệnh ssh | **tự giết phiên làm việc, 5 lần** | kill theo PID lấy từ `pgrep`, kiểm `/proc/<pid>/cmdline` trước |
| Nhiều tiến trình refiner cùng nạp dữ liệu | vượt hạn mức RAM, tiến trình bị huỷ | `ram_gate` trong `lib.sh` — đã gọi sẵn ở `31_reftrain.sh` |
| Không kiểm ổ trước lô nặng | ổ đầy giữa chừng, **mất kết quả 4 lần** | `need_disk` — đã gọi sẵn. 1 lượt apply ghi 2,7 GB; 1 dump 8–12 GB |
| Áp refiner thiếu `--K 2 --fp16 1` | số kênh không khớp, tải trọng số thất bại | luôn đi qua `scripts/40_infer.sh` |
| Chấm ảnh pinhole so GT có méo | lệch tới 240 px ở góc — **số vô hiệu** | luôn chấm thư mục `redistort/`, không chấm render trực tiếp |
| Hàm tạo bản đồ méo chạy trên scene đã khử méo | méo hai lần, lệch 35–70 px | kiểm lệch khối trước khi tin số |
| Chia mảnh khớp đặc trưng quá nhỏ | mỗi mảnh 13–14 GB RAM, 7/12 mảnh bị huỷ | `match_dense.py` giữ đặc trưng trên GPU; đừng chia quá số card |
| Chạy > 2 tiến trình apply refiner cùng lúc | nghẽn đĩa: 400 s → **3.945 s** | tối đa 2 song song |
| Máy chủ khởi động lại | `/tmp` trống, mọi chuỗi chết | đặt mọi thứ trong `$VT_ROOT`, **không dùng `/tmp`** |
| `--holdout_offset` không được đọc từ `config.json` | dump lấy nhầm view train (render overfit), refiner tệ đi 0,2 | **đã vá** trong `refine/refiner_data.py` — đừng revert |
| Tin điểm validation của refiner | **đã sai 7 lần** | val chỉ để chọn ckpt; quyết định phải chấm trên test/holdout |

## Khi một bước hỏng

Mọi script đều **idempotent**: mỗi bước xong ghi `$VT_RUNS/.done_<tag>`. Chạy lại
`run_all.sh` thì nó bỏ qua bước đã xong và làm tiếp từ chỗ hỏng.

Muốn chạy lại một bước: `rm $VT_RUNS/.done_<tag>` rồi chạy lại script đó.

Log của mỗi bước ở `$VT_LOGS/<tag>.log`. Log tổng ở `$VT_LOGS/run.log`.

## Kiểm nhanh khi nghi ngờ

```bash
bash env/verify_env.sh                  # môi trường + bộ chấm còn đúng không
python src/post/score_btc.py --selftest # riêng bộ chấm
bash tests/smoke.sh                     # toàn dây chuyền ở quy mô tí hon (~15 phút)
```

Chạy `--selftest` sau **mỗi** lần nâng torch, lpips hoặc driver. Nếu thước lệch, nó báo ngay
thay vì âm thầm trả số sai suốt cả ngày.
