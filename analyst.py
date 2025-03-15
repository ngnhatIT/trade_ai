import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import EMAIndicator, IchimokuIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, awesome_oscillator
from ta.volatility import AverageTrueRange
import seaborn as sns

# Load dữ liệu
def load_data(file_path="binance_BTCUSDT_5m.csv"):
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors='coerce')
    df = df.ffill().bfill()
    print(f"Kích thước dữ liệu: {df.shape}")
    return df

# Thêm đặc trưng kỹ thuật
def add_features(df):
    df = df.copy()
    # Cơ bản
    df["price_change"] = df["close"].pct_change().fillna(0)
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi().shift(1).fillna(50)
    df["ema_fast"] = EMAIndicator(close=df["close"], window=5).ema_indicator().shift(1).fillna(df["close"])
    df["ema_slow"] = EMAIndicator(close=df["close"], window=21).ema_indicator().shift(1).fillna(df["close"])
    df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range().shift(1).fillna(0)
    df["trend"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)
    df["volume_change"] = df["volume"].pct_change().fillna(0)
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek

    # Ichimoku Cloud
    ichimoku = IchimokuIndicator(high=df["high"], low=df["low"], window1=9, window2=26, window3=52)
    df["ichimoku_tenkan"] = ichimoku.ichimoku_conversion_line().shift(1).fillna(df["close"])
    df["ichimoku_kijun"] = ichimoku.ichimoku_base_line().shift(1).fillna(df["close"])
    df["ichimoku_span_a"] = ichimoku.ichimoku_a().shift(1).fillna(df["close"])
    df["ichimoku_span_b"] = ichimoku.ichimoku_b().shift(1).fillna(df["close"])
    df["ichimoku_signal"] = np.where(df["close"] > df["ichimoku_span_a"], 1, -1)

    # ADX
    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx.adx().shift(1).fillna(25)
    df["adx_pos"] = adx.adx_pos().shift(1).fillna(0)
    df["adx_neg"] = adx.adx_neg().shift(1).fillna(0)
    df["adx_signal"] = np.where(df["adx_pos"] > df["adx_neg"], 1, -1)

    # Fibonacci Retracement
    df["fib_high"] = df["high"].rolling(window=12).max().shift(1)
    df["fib_low"] = df["low"].rolling(window=12).min().shift(1)
    df["fib_range"] = df["fib_high"] - df["fib_low"]
    df["fib_38"] = df["fib_low"] + 0.382 * df["fib_range"]
    df["fib_50"] = df["fib_low"] + 0.5 * df["fib_range"]
    df["fib_61"] = df["fib_low"] + 0.618 * df["fib_range"]
    df["fib_signal"] = np.where(df["close"] > df["fib_61"], 1, np.where(df["close"] < df["fib_38"], -1, 0))

    # Chỉ báo bổ sung
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd().shift(1).fillna(0)
    df["macd_signal"] = macd.macd_signal().shift(1).fillna(0)
    df["macd_cross"] = np.where((df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1)), 1,
                                np.where((df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1)), -1, 0))
    
    df["stoch"] = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3).stoch().shift(1).fillna(50)
    df["stoch_signal"] = np.where(df["stoch"] > 80, -1, np.where(df["stoch"] < 20, 1, 0))
    
    df["momentum"] = df["close"].diff(5).shift(1).fillna(0)
    df["awesome"] = awesome_oscillator(high=df["high"], low=df["low"], window1=5, window2=34).shift(1).fillna(0)
    
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df

# Phân tích tổng quan
def analyze_overview(df):
    print("\n=== Tổng quan dữ liệu ===")
    print(df[["open", "high", "low", "close", "volume"]].describe())
    print("\nGiá trị thiếu:")
    print(df.isnull().sum())

# Phân tích biến động giá
def analyze_price_change(df):
    print("\n=== Phân tích biến động giá (price_change) ===")
    print(df["price_change"].describe())
    thresholds = [0.005, 0.002, 0.0015, 0.001, 0.0005]
    for thresh in thresholds:
        print(f"Tỷ lệ > {thresh}: {(df['price_change'] > thresh).mean() * 100:.2f}%")
        print(f"Tỷ lệ < -{thresh}: {(df['price_change'] < -thresh).mean() * 100:.2f}%")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df["price_change"], bins=100, kde=True)
    plt.title("Phân phối biến động giá (price_change)")
    plt.xlabel("Price Change")
    plt.ylabel("Tần suất")
    plt.savefig("price_change_histogram.png")
    plt.close()

# Phân tích xu hướng và chỉ báo
def analyze_indicators(df):
    print("\n=== Phân tích chỉ báo ===")
    print("Ichimoku Signal (1: LONG, -1: SHORT):")
    print(df["ichimoku_signal"].value_counts(normalize=True))
    print("ADX Signal (1: LONG, -1: SHORT):")
    print(df["adx_signal"].value_counts(normalize=True))
    print("Fibonacci Signal (1: LONG, -1: SHORT, 0: NEUTRAL):")
    print(df["fib_signal"].value_counts(normalize=True))
    print("MACD Crossover (1: LONG, -1: SHORT):")
    print(df["macd_cross"].value_counts(normalize=True))
    print("Stochastic Signal (1: LONG, -1: SHORT):")
    print(df["stoch_signal"].value_counts(normalize=True))
    print("Awesome Oscillator:")
    print(df["awesome"].describe())

    # Tương quan
    corr_features = ["price_change", "rsi", "atr", "ichimoku_tenkan", "ichimoku_kijun", "adx", "adx_pos", "adx_neg", 
                     "fib_38", "macd", "stoch", "momentum", "awesome"]
    corr = df[corr_features].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Tương quan giữa các đặc trưng và price_change")
    plt.savefig("correlation_heatmap.png")
    plt.close()

# Định nghĩa nhãn
def define_target(df, horizon=5, threshold=0.0015):
    df = df.copy()
    # Tính biến động giá trong 5 phút tiếp theo
    future_return = df["close"].pct_change(periods=horizon).shift(-horizon).fillna(0)
    # Nhãn dựa trên price_change và ADX signal
    conditions = [
        (future_return > threshold) & (df["adx_signal"] == 1),  # LONG: Tăng và +DI > -DI
        (future_return < -threshold) & (df["adx_signal"] == -1) # SHORT: Giảm và -DI > +DI
    ]
    df["target"] = np.select(conditions, [1, -1], default=0)  # 1: LONG, -1: SHORT, 0: NEUTRAL
    print("\nPhân phối nhãn:")
    print(df["target"].value_counts(normalize=True))
    return df

# Pipeline chính
def main():
    df = load_data()
    df = add_features(df)
    
    # Phân tích
    analyze_overview(df)
    analyze_price_change(df)
    analyze_indicators(df)
    df = define_target(df)

if __name__ == "__main__":
    main()