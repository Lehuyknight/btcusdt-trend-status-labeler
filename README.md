## BTCUSDT Trend Regime Lab

Repo này là một “laboratory” nhỏ để nghiên cứu bài toán **trend detection cho BTCUSDT Futures** với nhãn:

- `status = 1`: đang ở vùng **uptrend** (mở/giữ long),
- `status = -1`: đang ở vùng **downtrend** (mở/giữ short),
- `status = 0`: vùng **swing/uncertain** (đứng ngoài, đóng hết vị thế).

Pipeline gồm:

- `sample_trend_detection.py`: lấy dữ liệu kline từ Binance Futures, tính indicator, gán nhãn `status`, xuất CSV.
- `judgement_trend_detection.py`: đọc CSV, backtest logic theo `status`, in `net_pnl` theo từng năm, vẽ chart điểm nhận diện trend sai.

### 1) Cài đặt

- Python 3.10+ (khuyến nghị 3.11/3.12)
- Kết nối internet để gọi Binance Futures API

Cài dependencies:

```bash
python -m pip install pandas numpy matplotlib requests
```

Clone repo:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2) Cấu hình input

Trong `sample_trend_detection.py`, phần `USER CONFIG`:

- `SYMBOL`: mặc định `BTCUSDT`
- `INTERVAL`: khung thời gian, ví dụ `2h`, `4h`, `6h`
- `START_TIME`, `END_TIME`: chuỗi thời gian UTC (tối thiểu 1 năm, nên >= 4 năm để đánh giá 3 năm gần nhất)
- `OUTPUT_CSV`: tên file CSV output
- `OUTPUT_PARAMS_JSON`: file lưu tham số tối ưu tìm được

Ví dụ:

```python
SYMBOL = "BTCUSDT"
INTERVAL = "2h"
START_TIME = "2021-01-01 00:00:00"
END_TIME = "2025-12-31 23:59:59"
```

### 3) Chạy sinh nhãn trend

```bash
python sample_trend_detection.py
```

Sau khi chạy xong sẽ tạo:

- `output_trend_status.csv`
- `best_trend_params.json`

Schema CSV:

- `open_time, open, high, low, close, volume, status`

Ý nghĩa `status`:

- `0`: swing / không có edge rõ ràng → đứng ngoài hoặc đóng hết vị thế
- `1`: vùng giá được coi là uptrend → vào/giữ long
- `-1`: vùng giá được coi là downtrend → vào/giữ short

### 4) Chấm điểm chiến lược

```bash
python judgement_trend_detection.py --input_csv output_trend_status.csv
```

Tham số tùy chọn:

```bash
python judgement_trend_detection.py \
  --input_csv output_trend_status.csv \
  --fee_rate 0.0004 \
  --slippage_rate 0.0002 \
  --leverage 1.0 \
  --chart_file judgement_plot.png
```

Script sẽ in:

- `net_pnl` tổng
- `net_pnl` theo từng năm
- `max_drawdown`
- `turnover`
- `winrate`

Và lưu biểu đồ highlight các bar có vị thế nhưng bị lỗ vào:

- `judgement_plot.png`

### 5) Chiến lược bên trong (tóm tắt)

- **Regime filter**: ADX + ATR chuẩn hóa để chỉ trade khi xu hướng đủ mạnh và biến động đủ lớn.
- **Trend direction**: EMA nhanh/chậm + slope EMA + so sánh +DI / -DI.
- **Breakout filter**: yêu cầu giá phá vỡ vùng high/low rolling trước đó (tránh bị nhiễu trong range).
- **Hysteresis**: giữ trạng thái tối thiểu `hold_bars` nến trước khi đổi, nhằm giảm flip liên tục.
- **ATR stop**: nếu sau khi vào lệnh, giá đi ngược > `k * ATR` thì ép quay về `status = 0` (cắt lỗ logic).

Mục tiêu tối ưu là cải thiện **net_pnl từng năm** cho các năm gần nhất (ví dụ 2022–2025) trong khi kiểm soát drawdown và turnover.

### 6) Gợi ý hướng nghiên cứu tiếp

- Thử thêm các khung thời gian khác (`1h`, `12h`, `1d`) và so sánh.
- Thay đổi penalty trong hàm objective để ép **năm tệ nhất** tốt hơn (thay vì chỉ tối đa hóa trung bình).
- Kết hợp thêm filter về volume / funding rate (nếu có dữ liệu).

### 7) Lỗi thường gặp

- `ModuleNotFoundError`: chưa cài đủ thư viện → chạy lại lệnh `pip install`.
- Không có dữ liệu: kiểm tra internet hoặc `START_TIME/END_TIME`.
- PnL thấp: thử interval khác, tăng chiều dài dữ liệu, hoặc siết điều kiện trend (ADX/ATR/breakout). 
