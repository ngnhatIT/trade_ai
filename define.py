import pandas as pd
import numpy as np
import logging
import optuna
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hàm tính các chỉ báo kỹ thuật
def calc_rsi(df):
    return RSIIndicator(close=df["close"], window=14).rsi().shift(1).fillna(50)

def calc_atr(df):
    return AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range().shift(1).fillna(0)

def calc_macd(df):
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    return macd.macd().shift(1).fillna(0), macd.macd_signal().shift(1).fillna(0)

def calc_adx(df):
    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    return adx.adx().shift(1).fillna(20)

def calc_bollinger_bands(df):
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    return bb.bollinger_hband().shift(1).fillna(0), bb.bollinger_lband().shift(1).fillna(0)

def calc_stochastic(df):
    stoch = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3)
    return stoch.stoch().shift(1).fillna(50)

def calc_vwap(df):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).rolling(window=24).sum() / df["volume"].rolling(window=24).sum()
    return vwap.shift(1).ffill()

def calc_obv(df):
    direction = np.where(df["close"] > df["close"].shift(1), 1, np.where(df["close"] < df["close"].shift(1), -1, 0))
    obv = (direction * df["volume"]).cumsum()
    return obv.shift(1).fillna(0)

def calc_rvi(df):
    # Tính Relative Vigor Index (RVI)
    numerator = ((df["close"] - df["open"]) + 2 * (df["close"].shift(1) - df["open"].shift(1)) + 
                 2 * (df["close"].shift(2) - df["open"].shift(2)) + (df["close"].shift(3) - df["open"].shift(3))) / 6
    denominator = ((df["high"] - df["low"]) + 2 * (df["high"].shift(1) - df["low"].shift(1)) + 
                   2 * (df["high"].shift(2) - df["low"].shift(2)) + (df["high"].shift(3) - df["low"].shift(3))) / 6
    rvi = (numerator / denominator).rolling(window=14).mean()
    return rvi.shift(1).fillna(0)

# Hàm gán nhãn cũ (từ mã gốc)
def define_target_old(df, horizon=9, atr_multiplier=2.0):
    logging.info("Bắt đầu định nghĩa nhãn (logic cũ)...")
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["atr"] = calc_atr(df)
    df["rsi"] = calc_rsi(df)
    
    df["threshold"] = df["atr"] * atr_multiplier
    df["breakout_up"] = (df["close"] > df["high"].shift(1).rolling(5).max()) & (df["volume"] > df["volume"].shift(1).rolling(5).mean())
    df["breakout_down"] = (df["close"] < df["low"].shift(1).rolling(5).min()) & (df["volume"] > df["volume"].shift(1).rolling(5).mean())
    
    df["target"] = 1  # Default là NEUTRAL
    for i in range(len(df)):
        curr_threshold = df["threshold"].iloc[i]
        curr_close = df["close"].iloc[i]
        curr_ema_50 = df["ema_50"].iloc[i]
        curr_breakout_up = df["breakout_up"].iloc[i]
        curr_breakout_down = df["breakout_down"].iloc[i]
        curr_rsi = df["rsi"].iloc[i]
        
        if (curr_close > curr_ema_50 + curr_threshold) or curr_breakout_up or (curr_rsi > 70 and curr_close > curr_ema_50):
            df.iloc[i, df.columns.get_loc("target")] = 2  # LONG
        elif (curr_close < curr_ema_50 - curr_threshold) or curr_breakout_down or (curr_rsi < 30 and curr_close < curr_ema_50):
            df.iloc[i, df.columns.get_loc("target")] = 0  # SHORT
    
    df = df.drop(columns=["threshold", "breakout_up", "breakout_down"], errors='ignore').dropna()
    
    label_counts = df["target"].value_counts()
    total_samples = len(df)
    label_ratios = label_counts / total_samples
    logging.info(f"Phân phối nhãn (logic cũ) - Số lượng: {label_counts}")
    logging.info(f"Phân phối nhãn (logic cũ) - Tỷ lệ: {label_ratios}")
    
    return df

# Hàm gán nhãn mới (cải tiến để tăng tỷ lệ và chất lượng tín hiệu SHORT)
def define_target_new(df, horizon=9, atr_multiplier=2.0, rsi_upper_threshold=60, rsi_lower_threshold=40, 
                      macd_threshold=0.1, adx_threshold_long=20, adx_threshold_short=20, price_volatility_threshold=0.02,
                      stoch_upper_threshold=75, stoch_lower_threshold=25):
    logging.info("Bắt đầu định nghĩa nhãn (logic mới)...")
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["ema_200"] = EMAIndicator(close=df["close"], window=200).ema_indicator()
    df["atr"] = calc_atr(df)
    df["rsi"] = calc_rsi(df)
    df["macd"], df["macd_signal"] = calc_macd(df)
    df["adx"] = calc_adx(df)
    df["bb_upper"], df["bb_lower"] = calc_bollinger_bands(df)
    df["stoch"] = calc_stochastic(df)
    df["vwap"] = calc_vwap(df)
    df["obv"] = calc_obv(df)
    df["rvi"] = calc_rvi(df)
    
    df["threshold"] = df["atr"] * atr_multiplier
    df["breakout_up"] = (df["close"] > df["high"].shift(1).rolling(5).max()) & (df["volume"] > df["volume"].shift(1).rolling(5).mean() * 1.5)
    df["breakout_down"] = (df["close"] < df["low"].shift(1).rolling(5).min()) & (df["volume"] > df["volume"].shift(1).rolling(5).mean() * 1.5)
    
    # Tính độ dốc của MACD
    df["macd_slope"] = df["macd"] - df["macd"].shift(1)
    
    # Tính độ biến động giá (dựa trên ATR trong 5 phiên trước)
    df["price_volatility"] = df["atr"].rolling(window=5).mean().shift(1).fillna(0) / df["close"]
    
    # Tính độ dốc của OBV và RVI
    df["obv_slope"] = df["obv"] - df["obv"].shift(1)
    df["rvi_slope"] = df["rvi"] - df["rvi"].shift(1)
    
    df["target"] = 1  # Default là NEUTRAL
    for i in range(len(df)):
        curr_threshold = df["threshold"].iloc[i]
        curr_close = df["close"].iloc[i]
        curr_ema_50 = df["ema_50"].iloc[i]
        curr_ema_200 = df["ema_200"].iloc[i]
        curr_breakout_up = df["breakout_up"].iloc[i]
        curr_breakout_down = df["breakout_down"].iloc[i]
        curr_rsi = df["rsi"].iloc[i]
        curr_macd = df["macd"].iloc[i]
        curr_macd_signal = df["macd_signal"].iloc[i]
        curr_macd_slope = df["macd_slope"].iloc[i]
        curr_adx = df["adx"].iloc[i]
        curr_bb_upper = df["bb_upper"].iloc[i]
        curr_bb_lower = df["bb_lower"].iloc[i]
        curr_price_volatility = df["price_volatility"].iloc[i]
        curr_stoch = df["stoch"].iloc[i]
        curr_vwap = df["vwap"].iloc[i]
        curr_obv_slope = df["obv_slope"].iloc[i]
        curr_rvi_slope = df["rvi_slope"].iloc[i]
        
        # Logic mới: Thắt chặt điều kiện cho LONG, nới lỏng điều kiện cho SHORT, thêm RVI
        macd_diff = curr_macd - curr_macd_signal
        # Điều kiện cho LONG
        if (curr_price_volatility < price_volatility_threshold and
            curr_adx > adx_threshold_long and
            ((curr_close > curr_ema_50 + curr_threshold and curr_close > curr_ema_200) or
             (curr_breakout_up and curr_close > curr_ema_200) or
             (curr_close > curr_bb_upper and curr_close > curr_ema_200) or
             (curr_rsi > rsi_upper_threshold and curr_close > curr_ema_50 and curr_close > curr_ema_200 and 
              macd_diff > macd_threshold and curr_macd_slope > 0.0) or  # Thắt chặt điều kiện MACD
             (curr_stoch < stoch_lower_threshold and curr_close > curr_ema_200 and curr_close > curr_vwap and 
              curr_obv_slope > 0 and curr_rvi_slope > 0))):  # OBV và RVI xác nhận
            df.iloc[i, df.columns.get_loc("target")] = 2  # LONG
        # Điều kiện cho SHORT
        elif (curr_price_volatility < price_volatility_threshold and
              curr_adx > adx_threshold_short and
              ((curr_close < curr_ema_50 - curr_threshold and curr_close < curr_ema_200) or
               (curr_breakout_down and curr_close < curr_ema_200) or
               (curr_close < curr_bb_lower and curr_close < curr_ema_200) or
               (curr_rsi < rsi_lower_threshold and curr_close < curr_ema_50 and curr_close < curr_ema_200 and 
                macd_diff < -macd_threshold and curr_macd_slope < 0.4) or  # Nới lỏng điều kiện MACD
               (curr_stoch > stoch_upper_threshold and curr_close < curr_ema_200 and curr_close < curr_vwap and 
                curr_obv_slope < 0 and curr_rvi_slope < 0))):  # OBV và RVI xác nhận
            df.iloc[i, df.columns.get_loc("target")] = 0  # SHORT
    
    df = df.drop(columns=["threshold", "breakout_up", "breakout_down", "macd", "macd_signal", "macd_slope", "adx", 
                          "bb_upper", "bb_lower", "price_volatility", "stoch", "vwap", "obv", "obv_slope", "rvi", "rvi_slope"], 
                 errors='ignore').dropna()
    
    label_counts = df["target"].value_counts()
    total_samples = len(df)
    label_ratios = label_counts / total_samples
    logging.info(f"Phân phối nhãn (logic mới) - Số lượng: {label_counts}")
    logging.info(f"Phân phối nhãn (logic mới) - Tỷ lệ: {label_ratios}")
    
    return df

# Hàm đánh giá chất lượng nhãn
def evaluate_label_quality(df, horizon=720):
    logging.info("Đánh giá chất lượng nhãn...")
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df = df.dropna()
    
    # Tính lợi nhuận trung bình, tỷ lệ thành công, và độ biến động
    long_returns = df[df["target"] == 2]["future_return"]
    short_returns = df[df["target"] == 0]["future_return"]
    neutral_returns = df[df["target"] == 1]["future_return"]
    
    long_avg_return = long_returns.mean() if not long_returns.empty else 0
    short_avg_return = -short_returns.mean() if not short_returns.empty else 0
    neutral_avg_return = neutral_returns.mean() if not neutral_returns.empty else 0
    
    long_success_ratio = (long_returns > 0).mean() if not long_returns.empty else 0
    short_success_ratio = (short_returns < 0).mean() if not short_returns.empty else 0
    long_volatility = long_returns.std() if not long_returns.empty else 0
    short_volatility = short_returns.std() if not short_returns.empty else 0
    
    logging.info(f"Lợi nhuận trung bình (sau {horizon} giờ):")
    logging.info(f"LONG: {long_avg_return:.4f}, Tỷ lệ thành công: {long_success_ratio:.4f}, Độ biến động: {long_volatility:.4f}")
    logging.info(f"SHORT: {short_avg_return:.4f}, Tỷ lệ thành công: {short_success_ratio:.4f}, Độ biến động: {short_volatility:.4f}")
    logging.info(f"NEUTRAL: {neutral_avg_return:.4f}")
    
    return (long_avg_return, short_avg_return, long_success_ratio, short_success_ratio, long_volatility, short_volatility)

# Hàm phân tích xu hướng thị trường
def analyze_market_trend(df):
    logging.info("Phân tích xu hướng thị trường...")
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["ema_200"] = EMAIndicator(close=df["close"], window=200).ema_indicator()
    
    # Xác định xu hướng
    df["trend"] = "NEUTRAL"
    df.loc[df["ema_50"] > df["ema_200"], "trend"] = "UPTREND"
    df.loc[df["ema_50"] < df["ema_200"], "trend"] = "DOWNTREND"
    
    # Tính tỷ lệ các giai đoạn
    trend_counts = df["trend"].value_counts()
    total_samples = len(df)
    trend_ratios = trend_counts / total_samples
    logging.info(f"Phân phối xu hướng thị trường - Số lượng: {trend_counts}")
    logging.info(f"Phân phối xu hướng thị trường - Tỷ lệ: {trend_ratios}")
    
    return df

# Hàm mục tiêu để tối ưu hóa ngưỡng bằng Optuna
def objective(trial, df):
    try:
        atr_multiplier = trial.suggest_float("atr_multiplier", 1.0, 3.0)
        rsi_upper_threshold = trial.suggest_float("rsi_upper_threshold", 60, 90)  # Thắt chặt ngưỡng RSI cho LONG
        rsi_lower_threshold = trial.suggest_float("rsi_lower_threshold", 10, 40)  # Nới lỏng ngưỡng RSI cho SHORT
        macd_threshold = trial.suggest_float("macd_threshold", 0.1, 0.7)
        adx_threshold_long = trial.suggest_float("adx_threshold_long", 20, 40)  # Thắt chặt ngưỡng ADX cho LONG
        adx_threshold_short = trial.suggest_float("adx_threshold_short", 10, 30)  # Nới lỏng ngưỡng ADX cho SHORT
        price_volatility_threshold = trial.suggest_float("price_volatility_threshold", 0.01, 0.05)
        stoch_upper_threshold = trial.suggest_float("stoch_upper_threshold", 50, 70)  # Giảm ngưỡng Stochastic cho SHORT
        stoch_lower_threshold = trial.suggest_float("stoch_lower_threshold", 10, 30)
        
        logging.info(f"Trial với atr_multiplier={atr_multiplier}, rsi_upper_threshold={rsi_upper_threshold}, "
                     f"rsi_lower_threshold={rsi_lower_threshold}, macd_threshold={macd_threshold}, "
                     f"adx_threshold_long={adx_threshold_long}, adx_threshold_short={adx_threshold_short}, "
                     f"price_volatility_threshold={price_volatility_threshold}, "
                     f"stoch_upper_threshold={stoch_upper_threshold}, stoch_lower_threshold={stoch_lower_threshold}")
        
        # Gán nhãn với logic mới
        df_trial = define_target_new(df.copy(), atr_multiplier=atr_multiplier, 
                                     rsi_upper_threshold=rsi_upper_threshold, 
                                     rsi_lower_threshold=rsi_lower_threshold, 
                                     macd_threshold=macd_threshold,
                                     adx_threshold_long=adx_threshold_long,
                                     adx_threshold_short=adx_threshold_short,
                                     price_volatility_threshold=price_volatility_threshold,
                                     stoch_upper_threshold=stoch_upper_threshold,
                                     stoch_lower_threshold=stoch_lower_threshold)
        
        # Tính toán phân phối nhãn
        label_counts = df_trial["target"].value_counts()
        total_samples = len(df_trial)
        label_ratios = label_counts / total_samples
        
        # Mục tiêu: Tỷ lệ LONG và SHORT ~20-22%, NEUTRAL ~53-54%
        target_long_short_ratio = 0.215  # Trung bình giữa 20% và 22%
        target_neutral_ratio = 0.535     # Trung bình giữa 53% và 54%
        long_ratio = label_ratios.get(2, 0)
        short_ratio = label_ratios.get(0, 0)
        neutral_ratio = label_ratios.get(1, 0)
        
        # Tính độ lệch
        long_deviation = abs(long_ratio - target_long_short_ratio)
        short_deviation = abs(short_ratio - target_long_short_ratio)
        neutral_deviation = abs(neutral_ratio - target_neutral_ratio)
        total_deviation = long_deviation + short_deviation + neutral_deviation
        
        # Phạt nếu tỷ lệ LONG hoặc SHORT không đạt mục tiêu, hoặc NEUTRAL ngoài khoảng
        penalty = 0
        if long_ratio > 0.22:
            penalty += 80 * (long_ratio - 0.22)  # Tăng phạt nếu LONG vượt 22%
        if short_ratio > 0.22:
            penalty += 80 * (short_ratio - 0.22)
        if long_ratio < 0.20:
            penalty += 80 * (0.20 - long_ratio)
        if short_ratio < 0.20:
            penalty += 80 * (0.20 - short_ratio)  # Tăng phạt nếu SHORT dưới 20%
        if neutral_ratio < 0.53:
            penalty += 80 * (0.53 - neutral_ratio)
        if neutral_ratio > 0.54:
            penalty += 80 * (neutral_ratio - 0.54)  # Tăng phạt nếu NEUTRAL vượt 54%
        
        # Đánh giá chất lượng nhãn
        (long_avg_return, short_avg_return, long_success_ratio, short_success_ratio, 
         long_volatility, short_volatility) = evaluate_label_quality(df_trial.copy(), horizon=9)
        
        # Tính quality_score: Tăng trọng số cho SHORT
        quality_score = (long_avg_return * 150 + short_avg_return * 300) + \
                        (long_success_ratio * 75 + short_success_ratio * 150) - \
                        (long_volatility + short_volatility) * 20
        
        # Tổng điểm
        score = -(total_deviation + penalty) + quality_score
        logging.info(f"Score: {score}, Long ratio: {long_ratio}, Short ratio: {short_ratio}, Neutral ratio: {neutral_ratio}, "
                     f"Long avg return: {long_avg_return:.4f}, Short avg return: {short_avg_return:.4f}, "
                     f"Long success ratio: {long_success_ratio:.4f}, Short success ratio: {short_success_ratio:.4f}")
        return score
    except Exception as e:
        logging.error(f"Error in trial: {str(e)}")
        return -float('inf')

# Hàm chính để chạy thử nghiệm
def run_label_engineering(df):
    # Kiểm tra dữ liệu đầu vào
    if df.empty:
        logging.error("Dữ liệu đầu vào rỗng!")
        raise ValueError("Dữ liệu đầu vào rỗng!")
    
    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Thiếu các cột cần thiết: {missing_columns}")
        raise ValueError(f"Thiếu các cột cần thiết: {missing_columns}")
    
    # Kiểm tra dữ liệu có giá trị NaN
    if df[required_columns].isna().any().any():
        logging.warning("Dữ liệu có giá trị NaN, sẽ được điền bằng phương pháp ffill và bfill.")
        df = df.ffill().bfill()
    
    # Kiểm tra và log thông tin dữ liệu ban đầu
    logging.info(f"Dữ liệu ban đầu - Số lượng mẫu: {len(df)}")
    logging.info(f"Dữ liệu ban đầu - Thời gian bắt đầu: {df.index.min()}, Thời gian kết thúc: {df.index.max()}")
    
    # Lọc dữ liệu 6 tháng gần nhất (từ 19/09/2024 đến 13/03/2025)
    end_date = pd.to_datetime("2025-03-13")
    start_date = end_date - pd.Timedelta(days=6*30)  # 6 tháng
    df_filtered = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    
    # Kiểm tra dữ liệu sau khi lọc
    if df_filtered.empty:
        logging.error(f"Không có dữ liệu trong khoảng từ {start_date} đến {end_date}!")
        raise ValueError(f"Không có dữ liệu trong khoảng từ {start_date} đến {end_date}!")
    
    logging.info(f"Dữ liệu sau khi lọc - Số lượng mẫu: {len(df_filtered)}")
    logging.info(f"Dữ liệu sau khi lọc - Thời gian bắt đầu: {df_filtered.index.min()}, Thời gian kết thúc: {df_filtered.index.max()}")
    
    # Phân tích xu hướng thị trường
    df_filtered = analyze_market_trend(df_filtered.copy())
    
    # Gán nhãn với logic cũ
    df_old = define_target_old(df_filtered.copy(), horizon=9)
    (long_avg_return_old, short_avg_return_old, long_success_ratio_old, short_success_ratio_old, 
     long_volatility_old, short_volatility_old) = evaluate_label_quality(df_old.copy(), horizon=9)
    
    # Tối ưu hóa ngưỡng với Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df_filtered), n_trials=50)
    
    best_params = study.best_params
    logging.info(f"Best Params: {best_params}")
    
    # Gán nhãn với logic mới và tham số tối ưu
    df_new = define_target_new(df_filtered.copy(), horizon=9, atr_multiplier=best_params["atr_multiplier"],
                               rsi_upper_threshold=best_params["rsi_upper_threshold"],
                               rsi_lower_threshold=best_params["rsi_lower_threshold"],
                               macd_threshold=best_params["macd_threshold"],
                               adx_threshold_long=best_params["adx_threshold_long"],
                               adx_threshold_short=best_params["adx_threshold_short"],
                               price_volatility_threshold=best_params["price_volatility_threshold"],
                               stoch_upper_threshold=best_params["stoch_upper_threshold"],
                               stoch_lower_threshold=best_params["stoch_lower_threshold"])
    
    # Đánh giá chất lượng nhãn
    (long_avg_return_new, short_avg_return_new, long_success_ratio_new, short_success_ratio_new, 
     long_volatility_new, short_volatility_new) = evaluate_label_quality(df_new.copy(), horizon=9)
    
    # So sánh phân phối nhãn
    label_counts_old = df_old["target"].value_counts()
    label_ratios_old = label_counts_old / len(df_old)
    label_counts_new = df_new["target"].value_counts()
    label_ratios_new = label_counts_new / len(df_new)
    
    logging.info(f"\nSo sánh phân phối nhãn:")
    logging.info(f"Logic cũ - Số lượng: {label_counts_old}")
    logging.info(f"Logic cũ - Tỷ lệ: {label_ratios_old}")
    logging.info(f"Logic cũ - Lợi nhuận trung bình: LONG={long_avg_return_old:.4f}, SHORT={short_avg_return_old:.4f}")
    logging.info(f"Logic cũ - Tỷ lệ thành công: LONG={long_success_ratio_old:.4f}, SHORT={short_success_ratio_old:.4f}")
    logging.info(f"Logic cũ - Độ biến động: LONG={long_volatility_old:.4f}, SHORT={short_volatility_old:.4f}")
    logging.info(f"Logic mới - Số lượng: {label_counts_new}")
    logging.info(f"Logic mới - Tỷ lệ: {label_ratios_new}")
    logging.info(f"Logic mới - Lợi nhuận trung bình: LONG={long_avg_return_new:.4f}, SHORT={short_avg_return_new:.4f}")
    logging.info(f"Logic mới - Tỷ lệ thành công: LONG={long_success_ratio_new:.4f}, SHORT={short_success_ratio_new:.4f}")
    logging.info(f"Logic mới - Độ biến động: LONG={long_volatility_new:.4f}, SHORT={short_volatility_new:.4f}")
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    label_ratios_old.plot(kind='bar', title="Phân phối nhãn (Logic cũ)")
    plt.xticks(ticks=[0, 1, 2], labels=["SHORT", "NEUTRAL", "LONG"], rotation=0)
    plt.ylabel("Tỷ lệ")
    
    plt.subplot(1, 2, 2)
    label_ratios_new.plot(kind='bar', title="Phân phối nhãn (Logic mới)")
    plt.xticks(ticks=[0, 1, 2], labels=["SHORT", "NEUTRAL", "LONG"], rotation=0)
    plt.ylabel("Tỷ lệ")
    
    plt.tight_layout()
    plt.savefig("label_distribution_comparison.png")
    plt.close()
    
    return df_new, best_params

# Hàm main để chạy thử nghiệm
if __name__ == "__main__":
    # Load dữ liệu
    try:
        df = pd.read_csv("binance_BTCUSDT_1h.csv", parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors='coerce')
        df = df.ffill().bfill()
        df = df.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        # Chạy thử nghiệm
        df_new, best_params = run_label_engineering(df)
        
        # Lưu dữ liệu với nhãn mới
        df_new.to_csv("data_with_new_labels.csv")
        logging.info("Dữ liệu với nhãn mới đã được lưu vào 'data_with_new_labels.csv'")
    except Exception as e:
        logging.error(f"Error in label engineering: {str(e)}")
        raise