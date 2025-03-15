import requests
import pandas as pd
import time

# Thông số
symbol = "BTCUSDT"
interval = "5m"
start_time = int(time.mktime(time.strptime("2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"))) * 1000  # 01/05/2021
end_time = int(time.mktime(time.strptime("2025-03-13 00:00:00", "%Y-%m-%d %H:%M:%S"))) * 1000    # 01/05/2022
limit = 1000  # Giới hạn tối đa 1 request

# Hàm lấy dữ liệu
def get_klines(symbol, interval, start_time, end_time, limit):
    url = "https://api.binance.com/api/v3/klines"
    data = []
    current_time = start_time
    
    while current_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_time,
            "endTime": end_time,
            "limit": limit
        }
        response = requests.get(url, params=params)
        klines = response.json()
        if not klines:
            break
        data.extend(klines)
        current_time = int(klines[-1][6]) + 1  # Thời gian đóng nến cuối cùng + 1ms
        time.sleep(0.1)  # Tránh vượt giới hạn API
    return data

# Lấy dữ liệu
klines = get_klines(symbol, interval, start_time, end_time, limit)

# Chuyển thành DataFrame
df = pd.DataFrame(klines, columns=[
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

# Lưu file
df.to_csv("binance_BTCUSDT_5m.csv", index=False)
print("Đã lưu dữ liệu vào binance_BTCUSDT_5m.csv")