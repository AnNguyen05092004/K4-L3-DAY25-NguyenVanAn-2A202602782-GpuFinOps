# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,627  
**Projected savings:** $12,506  (**46%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,211 |
| Purchasing (spot/reserved) | $10,040 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest region (electricity $): us-east-wa
- Cleanest region (gCO2/kWh): europe-north1
- Best balanced region (cost + carbon price): us-east-wa

## Đơn giá $/1M-token (inference)

Đây là thước đo chính của lab: tiền trả cho mỗi triệu token phục vụ được, baseline = mọi request chạy model lớn, không cache, không batch.

| Kịch bản | $/ngày | $/1M-token | Tiết kiệm |
|---|---|---|---|
| Baseline (model lớn, không cache, không batch) | $48.87 | $6.488 | 0.0% |
| Chỉ cascade | $11.48 | $1.523 | 76.5% |
| Chỉ prompt cache | $44.27 | $5.877 | 9.4% |
| Chỉ batch | $40.64 | $5.394 | 16.9% |
| Cả ba (có cache_is_worth_it) | $8.50 | $1.128 | 82.6% |

Các đòn bẩy nhân vào nhau nên tiết kiệm riêng lẻ không cộng lại bằng mức khi bật cả ba. Cascade chiếm phần lớn vì 80% request chuyển sang model nhỏ rẻ hơn 15–37 lần. Các đòn bẩy hạ tầng (purchasing, right-size, idle) đo bằng USD/tháng vì dữ liệu telemetry và workloads không gắn với số token.

## Hiệu chỉnh số liệu (độ tin cậy của con số tiết kiệm)

Con số headline ở trên đến từ mission gốc. Sau khi làm Ext 1 và Ext 2, hai đòn bẩy được tính lại cho đúng hơn:

| Đòn bẩy | Headline (USD/tháng) | Hiệu chỉnh (USD/tháng) | Lý do |
|---|---|---|---|
| Inference (cascade/cache/batch) | $1,211 | $1,211 | Không đổi |
| Purchasing (spot/reserved) | $10,040 | $6,400 | Dùng cột days thật của từng job, reserved bị tính 24/7, tỷ lệ thu hồi spot theo loại GPU (Ext 1) |
| Right-size util-lies | $655 | $144 | Hạ cấp một bậc không qua kiểm tra tính toán, băng thông, VRAM; chỉ giữ GPU hạ được an toàn (Ext 2) |
| Kill idle GPUs | $600 | $600 | Không đổi |
| **Baseline** | $27,133 | $18,006 | Baseline on-demand tính theo số ngày thật |
| **Tổng tiết kiệm** | $12,506 (46.1%) | $8,355 (46.4%) | |

Kết luận: tuyệt đối số tiền giảm nhưng tỷ lệ tiết kiệm gần như không đổi (46.1% so với 46.4%), nên mức giảm khoảng 46% đáng tin.

## GPU-Util "nói dối": cơ chế và tác động chi phí

`GPU-Util` của nvidia-smi chỉ là tỷ lệ thời gian có ít nhất một kernel đang chạy. Một kernel nhỏ, chạy chậm hoặc đang chờ dữ liệu vẫn được tính là "bận". MFU mới đo phần FLOPs thực sự sinh ra so với đỉnh lý thuyết. Nguyên nhân thường gặp: chờ bộ nhớ (memory stall), batch quá nhỏ, chờ nạp dữ liệu hoặc giao tiếp giữa các GPU, quá nhiều kernel nhỏ làm tốn thời gian khởi chạy.

| GPU | Loại | GPU-Util | MFU | SM active | Tensor active | Tiền thuê/tháng | FLOPs trả tiền nhưng không dùng |
|---|---|---|---|---|---|---|---|
| gpu-h100-4 | H100 | 98.2% | 0.194 | 93% | 19% | $1,800 | ≈ $955/tháng (so với MFU 0.41 của các H100 khỏe) |
| gpu-a10g-1 | A10G | 96.9% | 0.268 | 92% | 27% | $720 | không có GPU cùng loại để so sánh |

Bằng chứng cơ chế: SM active rất cao nhưng Tensor active thấp nghĩa là các khối tính toán "bận" mà Tensor Core, bộ phận thực sự nhân ma trận, chỉ chạy một phần thời gian. Cách xử lý đúng là profiling để tăng MFU (batch lớn hơn, fuse kernel, nạp dữ liệu song song), không phải hạ cấp phần cứng (xem Ext 2).

## Khuyến nghị theo thứ tự ưu tiên (ROI)

| # | Hành động | Tiết kiệm ước tính | Vì sao ưu tiên ở mức này |
|---|---|---|---|
| 1 | Mua spot cho job train/dev/batch, reserved cho inference chạy 24/7 | $6,400/tháng (hiệu chỉnh) | Khoản tiết kiệm lớn nhất, chỉ đổi cách mua, không đổi code. Chỉ chọn reserved 3 năm khi chắc workload sống đủ 36 tháng (Ext 1) |
| 2 | Tắt GPU nằm không (lịch tắt/autoscale) | $600/tháng | Rủi ro bằng 0, làm trong ngày |
| 3 | Bật cascade, batch, prompt cache cho inference | $1,211/tháng, $6.488 → $1.128/1M-token | Giảm đơn giá lớn nhất, nhưng khoản chi inference chỉ là phần nhỏ của baseline và cần kiểm tra chất lượng model nhỏ |
| 4 | Profiling các GPU lie để tăng MFU | ≈ $955/tháng (chỉ tính GPU có đối chứng) | Sửa nguyên nhân gốc thay vì hạ cấp, nhưng cần công kỹ sư |
| 5 | Trần reasoning 2% traffic | $23/tháng và 71% năng lượng | Tiền ít nhưng reasoning chiếm 94% điện; cần đo chất lượng |
| 6 | Dời job train/batch sang us-east-wa | 519 kg CO2e và $116 tiền điện cho các job ngắt được | Không ảnh hưởng người dùng; cần tính phí egress và quota |
| 7 | Right-size theo băng thông và VRAM | $144/tháng (NVIDIA), $2,520/tháng nếu chấp nhận AMD MI300X | Đòn bẩy nhỏ nếu giữ NVIDIA; đổi AMD cần chuyển stack phần mềm |

## Nhận xét về tính bền vững (carbon, vùng triển khai)

Với 1,789 kWh của 5 job có thể ngắt, giả định công suất định mức (TDP) và giá carbon nội bộ $100/tấn:

| Vùng | $/kWh | gCO2/kWh | Tiền điện | CO2 (kg) | Tổng gồm tiền carbon | Nhãn |
|---|---|---|---|---|---|---|
| us-east-wa | 0.055 | 90 | $98.39 | 161.0 | $114.50 | rẻ nhất, cân bằng |
| us-west-2 | 0.070 | 120 | $125.23 | 214.7 | $146.70 |  |
| europe-north1 | 0.090 | 30 | $161.01 | 53.7 | $166.38 | sạch nhất |
| us-east-1 | 0.120 | 380 | $214.68 | 679.8 | $282.66 |  |
| europe-central2 | 0.180 | 660 | $322.02 | 1,180.7 | $440.09 |  |

- Vùng rẻ nhất và cân bằng nhất là **us-east-wa**: tiền điện $214.68 → $98.39, carbon 680 → 161 kg.
- Vùng sạch nhất **europe-north1** giảm carbon 92% nhưng chỉ thắng về tổng chi phí khi giá carbon vượt khoảng $583/tấn. Chọn vùng nào tùy ưu tiên của công ty: tiền, hay cam kết giảm phát thải.
- Reasoning chỉ chiếm 8.4% traffic và 16.4% chi phí nhưng tới 94% năng lượng, nên quản lý reasoning là đòn bẩy carbon lớn nhất ở phía inference.
- Đánh đổi: dời job sang vùng xa không ảnh hưởng job train/batch nhưng có phí egress, quota GPU, quy định lưu trữ dữ liệu và độ trễ nếu là dịch vụ phục vụ người dùng; inference realtime không nên dời.

## Phần mở rộng đã thực hiện (đo lường)

| Extension | Kết quả đo | Insight chính |
|---|---|---|
| 1. recommend_tier nâng cao | Tiết kiệm mua GPU: 39.1% (gốc) → 38.5% (quy tắc cũ, tính tiền đúng) → 38.7% (chính sách mới) | Tier không đổi trên 8 job; thay đổi chủ yếu do tính tiền đúng (days, reserved 24/7). Vòng đời workload quyết định reserved 1 năm hay 3 năm |
| 2. Right-sizing theo MBU | An toàn: $144/tháng; nếu chấp nhận AMD: $2,520/tháng; 2 GPU lie không hạ cấp được | Rẻ theo giờ chưa đủ: phải đủ TFLOPs, băng thông và VRAM |
| 3. cache_is_worth_it | Hòa vốn small 1.67 / large 0.11 lần đọc; thực tế 18.8 / 4.2; 4 request mất chiết khấu | Traffic dày nên cache gần như luôn có lợi; hàm chặn tiết kiệm ảo khi traffic thưa |
| 4. Ngân sách reasoning | 8.4% traffic = 16.4% chi phí = 94% năng lượng; trần 2%: -$0.78/ngày, -71% Wh | Reasoning là vấn đề năng lượng nhiều hơn vấn đề tiền; trần 10% chưa có tác dụng vì hiện chỉ 8.4% |
| 5. Lập lịch carbon | 680 → 54 kg CO2e (-92%) ở europe-north1; rẻ và cân bằng nhất: us-east-wa | "Tối ưu" phụ thuộc tiêu chí: giá carbon phải rất cao mới đổi kết quả |

## Giả định và giới hạn

- Dữ liệu là mô phỏng tất định (seed=25); giá là snapshot tháng 6/2026.
- Baseline inference giả định mọi request chạy model lớn nên mức giảm 82.6% là cận trên; cascade chưa đo chất lượng.
- Baseline gộp tiền token (giá API) với tiền thuê GPU của 8 job; hai đòn bẩy idle và right-size tính trên đội telemetry 11 GPU, không nằm trong baseline.
- Energy per query dùng giả định 800 token/query; trung bình thật của dữ liệu là 3,139 token/request.
- Giả định của extension: cache sống 1 giờ và phí lưu $0.30/1M token (Ext 3); tỷ lệ thu hồi spot theo GPU và vòng đời workload (Ext 1); giá carbon $100/tấn, công suất TDP, bỏ qua PUE (Ext 5); size theo nhu cầu đỉnh với mục tiêu tải 70% (Ext 2).

_Figures are June-2026 as-of snapshots; re-baseline before acting._