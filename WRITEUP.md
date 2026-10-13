# Bài viết ngắn — Lab 25: Tối ưu chi phí GPU (NimbusAI)

**Sinh viên:** Nguyễn Văn An · **Mã:** 2A202602782 · **Hình thức:** cá nhân

Mọi con số dưới đây lấy từ output của các mission và `outputs/report.md` (dữ liệu mô phỏng seed=25, giá snapshot tháng 6/2026).

## 1. Baseline và Optimized

| | Baseline | Optimized | Tiết kiệm |
|---|---|---|---|
| Tổng chi phí hằng tháng | $27,133 | $14,627 | **$12,506 (46.1%)** |
| Inference, đơn giá | $6.488 / 1M-token | $1.128 / 1M-token | 82.6% |

Baseline inference giả định mọi request chạy model lớn, không cache, không batch, nên 82.6% là cận trên. Sau khi hiệu chỉnh hai đòn bẩy (xem mục 4), tổng tiết kiệm là $8,355 trên baseline $18,006, tức **46.4%**, gần như không đổi về tỷ lệ, vì vậy mức giảm khoảng 46% đáng tin.

## 2. Phân tích từng đòn bẩy

| Đòn bẩy | Tiết kiệm / tháng | Tỷ trọng |
|---|---|---|
| Purchasing (spot, reserved) | $10,040 | 80% |
| Inference (cascade, cache, batch) | $1,211 | 10% |
| Right-size GPU lie | $655 | 5% |
| Tắt GPU nằm không | $600 | 5% |

**Purchasing đóng góp nhiều nhất** vì tiền thuê GPU chiếm khoảng 95% baseline ($25,667 trên $27,133); đòn bẩy nằm trên khoản chi lớn nhất thắng, dù không phải đòn bẩy mạnh nhất về phần trăm. Trong inference, **cascade là chủ lực** (một mình giảm 76.5%), batch giảm 16.9%, prompt cache chỉ 9.4% vì chỉ khoảng 32% input cache được và cache chỉ giảm giá input trong khi output mới là phần đắt của model lớn. Các đòn bẩy nhân vào nhau nên không cộng lại.

## 3. GPU-Util "nói dối"

`gpu-h100-4` báo GPU-Util 98.2% nhưng MFU chỉ 0.194, trong khi các H100 khỏe có MFU khoảng 0.41. Telemetry cho thấy SM active 93% nhưng Tensor active chỉ 19%: các khối tính toán bận, còn Tensor Core (nơi nhân ma trận thật sự) chỉ chạy khoảng một phần năm thời gian. Nguyên nhân thường gặp là chờ bộ nhớ hoặc dữ liệu, batch nhỏ, kernel nhỏ. Tác động tài chính: GPU này tốn $1,800/tháng nhưng chỉ cho ra khoảng một nửa công việc của GPU cùng loại, tức khoảng **$955/tháng** tiền FLOPs không dùng. `gpu-a10g-1` (util 96.9%, MFU 0.268, $720/tháng) cũng bị đánh dấu; Guide chỉ nhắc một GPU nhưng dữ liệu cho hai. Ngoài ra `gpu-h100-5` nằm không 8 giờ mỗi ngày, mất $600/tháng.

## 4. Các phần mở rộng đã làm (đủ 5/5)

| Extension | Kết quả đo | Insight quan trọng nhất |
|---|---|---|
| 1. `recommend_tier` nâng cao (tỷ lệ thu hồi theo GPU, reserved 1 năm vs 3 năm) | Tiết kiệm mua GPU 39.1% (gốc) → 38.5% (quy tắc cũ, tính tiền đúng) → 38.7% (chính sách mới) | Tier không đổi trên 8 job; số đổi chủ yếu do sửa cách tính tiền (cột `days`, reserved bị tính 24/7: ví dụ `job-infer-search` thật là $1,296 chứ không phải $972). Vòng đời workload quyết định: dưới 12 tháng thì on-demand, 12 đến 24 tháng thì reserved 1 năm, 36 tháng mới là 3 năm |
| 2. Right-sizing theo MBU | An toàn chỉ $144/tháng (A10G → L4); $2,520/tháng nếu chấp nhận AMD MI300X | Rẻ theo giờ chưa đủ: GPU mới phải đủ TFLOPs, băng thông và VRAM. Hai GPU lie không hạ cấp được (H100 → A100 thiếu tính toán, A10G → L4 thiếu băng thông), nên sửa bằng tăng MFU chứ không hạ phần cứng |
| 3. `cache_is_worth_it` | Hòa vốn model small 1.67 lần đọc, large 0.11; thực tế 18.8 và 4.2 lần; chỉ 4/2,400 request mất chiết khấu | Traffic dày nên cache gần như luôn có lợi; hàm chặn "tiết kiệm ảo" khi traffic thưa. Model rẻ cần nhiều lần đọc hơn vì phí ghi cố định chiếm tỷ lệ lớn so với giá input |
| 4. Ngân sách reasoning | 8.4% traffic = 16.4% chi phí = **94% năng lượng**; trần 2% traffic tiết kiệm $0.78/ngày và 71% Wh | Reasoning là vấn đề năng lượng nhiều hơn vấn đề tiền. Trần 10% không có tác dụng vì mức hiện tại chỉ 8.4% |
| 5. Lập lịch carbon | 1,789 kWh: carbon 680 → 54 kg CO2e (giảm 92%) ở `europe-north1`; `us-east-wa` rẻ nhất và cân bằng nhất (tiền điện $215 → $98) | "Tối ưu" phụ thuộc tiêu chí; `europe-north1` chỉ thắng về tổng chi phí khi giá carbon vượt khoảng $583/tấn |

Mỗi extension có test riêng trong `tests/test_extensions.py` (không sửa test gốc). Tổng 29 test pass.

## 5. Ba hành động đầu tiên nếu là FinOps lead

1. **Đổi cách mua GPU** (khoảng $6,400 đến $10,040/tháng): spot cho job train, dev, batch; reserved cho inference chạy 24/7, nhưng chỉ chọn kỳ hạn 3 năm khi chắc dịch vụ sống đủ 36 tháng. Lớn nhất, không cần đổi code.
2. **Tắt GPU nằm không bằng lịch tự động** ($600/tháng): rủi ro bằng 0, làm trong ngày.
3. **Bật cascade, batch, cache cho inference và đặt trần reasoning 2%** (đơn giá $6.488 → $1.128/1M-token, giảm 71% tổng năng lượng), đồng thời **profiling hai GPU lie** để tăng MFU thay vì hạ cấp. Việc này cần kiểm tra chất lượng model nhỏ trước khi mở rộng.

## Giới hạn cần lưu ý

Dữ liệu mô phỏng, baseline inference lạc quan, baseline trộn tiền token với tiền thuê GPU, và nhiều giả định của extension (cache sống 1 giờ, phí lưu $0.30/1M token, tỷ lệ thu hồi spot theo GPU, giá carbon $100/tấn, công suất TDP, mục tiêu tải 70%). Chi tiết ở mục "Giả định và giới hạn" của `outputs/report.md`.
