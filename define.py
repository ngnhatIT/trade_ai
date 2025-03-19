import pandas as pd
import numpy as np
import logging
import optuna
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
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

# Hàm gán nhãn cũ
def define_target_old(df, horizon=2400, atr_multiplier=2.0):
    logging.info("Bắt đầu định nghĩa nhãn (logic cũ)...")
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["atr"] = calc_atr(df)
    df["rsi"] = calc_rsi(df)
    
    df["threshold"] = df["atr"] * atr_multiplier
    df["breakout_up"] = (df["close"] > df["high"].shift(1).rolling(5).max()) & (df["volume"] > df["volume"].shift(1).rolling(5).mean())
    df["breakout_down"] = (df["close"] < df["low"].shift(1).rolling(5).min()) & (df["volume"] > df["volume"].shift(1).rolling(5).mean())
    
    df["target"] = 1
    for i in range(len(df)):
        curr_threshold = df["threshold"].iloc[i]
        curr_close = df["close"].iloc[i]
        curr_ema_50 = df["ema_50"].iloc[i]
        curr_breakout_up = df["breakout_up"].iloc[i]
        curr_breakout_down = df["breakout_down"].iloc[i]
        curr_rsi = df["rsi"].iloc[i]
        
        if (curr_close > curr_ema_50 + curr_threshold) or curr_breakout_up or (curr_rsi > 70 and curr_close > curr_ema_50):
            df.iloc[i, df.columns.get_loc("target")] = 2
        elif (curr_close < curr_ema_50 - curr_threshold) or curr_breakout_down or (curr_rsi < 30 and curr_close < curr_ema_50):
            df.iloc[i, df.columns.get_loc("target")] = 0
    
    df = df.drop(columns=["threshold", "breakout_up", "breakout_down"], errors='ignore').dropna()
    label_counts = df["target"].value_counts()
    total_samples = len(df)
    label_ratios = label_counts / total_samples
    logging.info(f"Phân phối nhãn (logic cũ) - Số lượng: {label_counts}")
    logging.info(f"Phân phối nhãn (logic cũ) - Tỷ lệ: {label_ratios}")
    return df

# Hàm gán nhãn mới (cải tiến cho horizon 2400 giờ)
def define_target_new(df, horizon=2400, atr_multiplier=1.5, rsi_upper_threshold=60, rsi_lower_threshold=40, 
                      macd_threshold=0.05, adx_threshold_long=20, adx_threshold_short=15, volume_multiplier=1.3):
    logging.info("Bắt đầu định nghĩa nhãn (logic mới cải tiến)...")
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["ema_200"] = EMAIndicator(close=df["close"], window=200).ema_indicator()
    df["atr"] = calc_atr(df)
    df["rsi"] = calc_rsi(df)
    df["macd"], df["macd_signal"] = calc_macd(df)
    df["adx"] = calc_adx(df)
    df["bb_upper"], df["bb_lower"] = calc_bollinger_bands(df)
    
    df["threshold"] = df["atr"] * atr_multiplier
    df["volume_condition"] = df["volume"] > df["volume"].shift(1).rolling(20).mean() * volume_multiplier
    
    df["target"] = 1
    for i in range(len(df)):
        curr_threshold = df["threshold"].iloc[i]
        curr_close = df["close"].iloc[i]
        curr_ema_200 = df["ema_200"].iloc[i]
        curr_rsi = df["rsi"].iloc[i]
        curr_macd = df["macd"].iloc[i]
        curr_macd_signal = df["macd_signal"].iloc[i]
        curr_adx = df["adx"].iloc[i]
        curr_bb_upper = df["bb_upper"].iloc[i]
        curr_bb_lower = df["bb_lower"].iloc[i]
        curr_volume_condition = df["volume_condition"].iloc[i]
        
        macd_diff = curr_macd - curr_macd_signal
        # LONG: Tập trung vào xu hướng dài hạn với EMA 200
        if (curr_adx > adx_threshold_long and
            curr_close < curr_bb_upper * 0.98 and
            ((curr_close > curr_ema_200 + curr_threshold and curr_rsi > rsi_upper_threshold) or
             (curr_close > curr_ema_200 and macd_diff > macd_threshold and curr_volume_condition))):
            df.iloc[i, df.columns.get_loc("target")] = 2
        # SHORT: Tập trung vào xu hướng dài hạn với EMA 200
        elif (curr_adx > adx_threshold_short and
              curr_close > curr_bb_lower * 1.02 and
              ((curr_close < curr_ema_200 - curr_threshold and curr_rsi < rsi_lower_threshold) or
               (curr_close < curr_ema_200 and macd_diff < -macd_threshold and curr_volume_condition))):
            df.iloc[i, df.columns.get_loc("target")] = 0
    
    df = df.drop(columns=["threshold", "macd", "macd_signal", "adx", "bb_upper", "bb_lower", "volume_condition"], errors='ignore').dropna()
    
    label_counts = df["target"].value_counts()
    total_samples = len(df)
    label_ratios = label_counts / total_samples
    logging.info(f"Phân phối nhãn (logic mới cải tiến) - Số lượng: {label_counts}")
    logging.info(f"Phân phối nhãn (logic mới cải tiến) - Tỷ lệ: {label_ratios}")
    return df

# Hàm đánh giá chất lượng nhãn (sửa lỗi SettingWithCopyWarning)
def evaluate_label_quality(df, horizon=2400, risk_free_rate=0.0, transaction_cost=0.001):
    logging.info(f"Đánh giá chất lượng nhãn thực tế (horizon={horizon} giờ, chi phí giao dịch={transaction_cost*100:.2f}%)...")
    df = df.copy()  # Tạo bản sao để tránh sửa đổi DataFrame gốc
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df = df.dropna()
    
    long_returns = df[df["target"] == 2]["future_return"]
    short_returns = df[df["target"] == 0]["future_return"]
    neutral_returns = df[df["target"] == 1]["future_return"]
    
    long_returns_adj = long_returns - 2 * transaction_cost
    short_returns_adj = -short_returns - 2 * transaction_cost
    neutral_returns_adj = neutral_returns
    
    long_avg_return = long_returns_adj.mean() if not long_returns_adj.empty else 0
    short_avg_return = short_returns_adj.mean() if not short_returns_adj.empty else 0
    neutral_avg_return = neutral_returns_adj.mean() if not neutral_returns_adj.empty else 0
    long_success_ratio = (long_returns_adj > 0).mean() if not long_returns_adj.empty else 0
    short_success_ratio = (short_returns_adj > 0).mean() if not short_returns_adj.empty else 0
    long_volatility = long_returns_adj.std() if not long_returns_adj.empty else 0
    short_volatility = short_returns_adj.std() if not short_returns_adj.empty else 0
    
    long_sharpe = ((long_avg_return - risk_free_rate) / long_volatility) if long_volatility > 0 else 0
    short_sharpe = ((short_avg_return - risk_free_rate) / short_volatility) if short_volatility > 0 else 0
    
    def calc_max_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min() if not drawdown.empty else 0
    
    long_max_dd = calc_max_drawdown(long_returns_adj)
    short_max_dd = calc_max_drawdown(short_returns_adj)
    
    # Sử dụng .loc để tránh SettingWithCopyWarning
    df.loc[:, "trend"] = np.where(df["ema_50"] > df["ema_200"], "UPTREND", 
                                  np.where(df["ema_50"] < df["ema_200"], "DOWNTREND", "NEUTRAL"))
    long_uptrend_success = (df.loc[(df["target"] == 2) & (df["trend"] == "UPTREND"), "future_return"] - 2 * transaction_cost > 0).mean() if not df[(df["target"] == 2) & (df["trend"] == "UPTREND")].empty else 0
    short_downtrend_success = (-df.loc[(df["target"] == 0) & (df["trend"] == "DOWNTREND"), "future_return"] - 2 * transaction_cost > 0).mean() if not df[(df["target"] == 0) & (df["trend"] == "DOWNTREND")].empty else 0
    
    logging.info(f"Lợi nhuận trung bình (sau {horizon} giờ, đã trừ chi phí giao dịch):")
    logging.info(f"LONG: {long_avg_return:.4f}, Tỷ lệ thành công: {long_success_ratio:.4f}, Độ biến động: {long_volatility:.4f}")
    logging.info(f"SHORT: {short_avg_return:.4f}, Tỷ lệ thành công: {short_success_ratio:.4f}, Độ biến động: {short_volatility:.4f}")
    logging.info(f"NEUTRAL: {neutral_avg_return:.4f}")
    logging.info(f"Sharpe Ratio - LONG: {long_sharpe:.4f}, SHORT: {short_sharpe:.4f}")
    logging.info(f"Max Drawdown - LONG: {long_max_dd:.4f}, SHORT: {short_max_dd:.4f}")
    logging.info(f"Tỷ lệ thành công theo xu hướng - LONG trong UPTREND: {long_uptrend_success:.4f}, SHORT trong DOWNTREND: {short_downtrend_success:.4f}")
    
    return (long_avg_return, short_avg_return, long_success_ratio, short_success_ratio, 
            long_volatility, short_volatility, long_sharpe, short_sharpe, long_max_dd, short_max_dd,
            long_uptrend_success, short_downtrend_success)

# Hàm phân tích xu hướng thị trường
def analyze_market_trend(df):
    logging.info("Phân tích xu hướng thị trường...")
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["ema_200"] = EMAIndicator(close=df["close"], window=200).ema_indicator()
    
    df["trend"] = "NEUTRAL"
    df.loc[df["ema_50"] > df["ema_200"], "trend"] = "UPTREND"
    df.loc[df["ema_50"] < df["ema_200"], "trend"] = "DOWNTREND"
    
    trend_counts = df["trend"].value_counts()
    total_samples = len(df)
    trend_ratios = trend_counts / total_samples
    logging.info(f"Phân phối xu hướng thị trường - Số lượng: {trend_counts}")
    logging.info(f"Phân phối xu hướng thị trường - Tỷ lệ: {trend_ratios}")
    return df

# Hàm tối ưu hóa ngưỡng bằng Optuna
def objective(trial, df):
    try:
        atr_multiplier = trial.suggest_float("atr_multiplier", 1.0, 2.5)
        rsi_upper_threshold = trial.suggest_float("rsi_upper_threshold", 55, 70)
        rsi_lower_threshold = trial.suggest_float("rsi_lower_threshold", 30, 45)
        macd_threshold = trial.suggest_float("macd_threshold", 0.02, 0.10)
        adx_threshold_long = trial.suggest_float("adx_threshold_long", 15, 25)
        adx_threshold_short = trial.suggest_float("adx_threshold_short", 10, 20)
        volume_multiplier = trial.suggest_float("volume_multiplier", 1.2, 1.8)
        
        logging.info(f"Trial với atr_multiplier={atr_multiplier}, rsi_upper_threshold={rsi_upper_threshold}, "
                     f"rsi_lower_threshold={rsi_lower_threshold}, macd_threshold={macd_threshold}, "
                     f"adx_threshold_long={adx_threshold_long}, adx_threshold_short={adx_threshold_short}, "
                     f"volume_multiplier={volume_multiplier}")
        
        df_trial = define_target_new(df.copy(), horizon=2400, atr_multiplier=atr_multiplier, 
                                     rsi_upper_threshold=rsi_upper_threshold, 
                                     rsi_lower_threshold=rsi_lower_threshold, 
                                     macd_threshold=macd_threshold,
                                     adx_threshold_long=adx_threshold_long,
                                     adx_threshold_short=adx_threshold_short,
                                     volume_multiplier=volume_multiplier)
        
        label_counts = df_trial["target"].value_counts()
        total_samples = len(df_trial)
        label_ratios = label_counts / total_samples
        
        target_long_short_ratio = 0.215
        target_neutral_ratio = 0.535
        long_ratio = label_ratios.get(2, 0)
        short_ratio = label_ratios.get(0, 0)
        neutral_ratio = label_ratios.get(1, 0)
        
        long_deviation = abs(long_ratio - target_long_short_ratio)
        short_deviation = abs(short_ratio - target_long_short_ratio)
        neutral_deviation = abs(neutral_ratio - target_neutral_ratio)
        total_deviation = long_deviation + short_deviation + neutral_deviation
        
        penalty = 0
        if long_ratio > 0.22: penalty += 500 * (long_ratio - 0.22)
        if short_ratio > 0.22: penalty += 500 * (short_ratio - 0.22)
        if long_ratio < 0.20: penalty += 500 * (0.20 - long_ratio)
        if short_ratio < 0.20: penalty += 500 * (0.20 - short_ratio)
        if neutral_ratio < 0.53: penalty += 500 * (0.53 - neutral_ratio)
        if neutral_ratio > 0.54: penalty += 500 * (neutral_ratio - 0.54)
        
        (long_avg_return, short_avg_return, long_success_ratio, short_success_ratio, 
         long_volatility, short_volatility, long_sharpe, short_sharpe, long_max_dd, short_max_dd,
         long_uptrend_success, short_downtrend_success) = evaluate_label_quality(df_trial.copy(), horizon=2400)
        
        quality_score = (long_avg_return * 1000 + short_avg_return * 1000) + \
                        (long_success_ratio * 500 + short_success_ratio * 500) - \
                        (long_volatility + short_volatility) * 50
        if long_avg_return < 0: penalty += 600 * abs(long_avg_return)
        if short_avg_return < 0: penalty += 600 * abs(short_avg_return)
        if abs(long_max_dd) > 0.5: penalty += 300 * (abs(long_max_dd) - 0.5)
        if abs(short_max_dd) > 0.5: penalty += 300 * (abs(short_max_dd) - 0.5)
        
        score = -(total_deviation + penalty) + quality_score
        logging.info(f"Score: {score}, Long ratio: {long_ratio}, Short ratio: {short_ratio}, Neutral ratio: {neutral_ratio}")
        return score
    except Exception as e:
        logging.error(f"Error in trial: {str(e)}")
        return -float('inf')

# Hàm chính để chạy thử nghiệm
def run_label_engineering(df):
    if df.empty:
        logging.error("Dữ liệu đầu vào rỗng!")
        raise ValueError("Dữ liệu đầu vào rỗng!")
    
    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Thiếu các cột cần thiết: {missing_columns}")
        raise ValueError(f"Thiếu các cột cần thiết: {missing_columns}")
    
    if df[required_columns].isna().any().any():
        logging.warning("Dữ liệu có giá trị NaN, sẽ được điền bằng phương pháp ffill và bfill.")
        df = df.ffill().bfill()
    
    logging.info(f"Dữ liệu ban đầu - Số lượng mẫu: {len(df)}")
    logging.info(f"Dữ liệu ban đầu - Thời gian bắt đầu: {df.index.min()}, Thời gian kết thúc: {df.index.max()}")
    
    end_date = pd.to_datetime("2025-03-13")
    start_date = end_date - pd.Timedelta(days=6*30)
    df_filtered = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    
    if df_filtered.empty:
        logging.error(f"Không có dữ liệu trong khoảng từ {start_date} đến {end_date}!")
        raise ValueError(f"Không có dữ liệu trong khoảng từ {start_date} đến {end_date}!")
    
    logging.info(f"Dữ liệu sau khi lọc - Số lượng mẫu: {len(df_filtered)}")
    logging.info(f"Dữ liệu sau khi lọc - Thời gian bắt đầu: {df_filtered.index.min()}, Thời gian kết thúc: {df_filtered.index.max()}")
    
    df_filtered = analyze_market_trend(df_filtered.copy())
    
    df_old = define_target_old(df_filtered.copy(), horizon=2400)
    (long_avg_return_old, short_avg_return_old, long_success_ratio_old, short_success_ratio_old, 
     long_volatility_old, short_volatility_old, *_) = evaluate_label_quality(df_old.copy(), horizon=2400)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df_filtered), n_trials=50)
    
    best_params = study.best_params
    logging.info(f"Best Params: {best_params}")
    
    df_new = define_target_new(df_filtered.copy(), horizon=2400, atr_multiplier=best_params["atr_multiplier"],
                               rsi_upper_threshold=best_params["rsi_upper_threshold"],
                               rsi_lower_threshold=best_params["rsi_lower_threshold"],
                               macd_threshold=best_params["macd_threshold"],
                               adx_threshold_long=best_params["adx_threshold_long"],
                               adx_threshold_short=best_params["adx_threshold_short"],
                               volume_multiplier=best_params["volume_multiplier"])
    
    (long_avg_return_new, short_avg_return_new, long_success_ratio_new, short_success_ratio_new, 
     long_volatility_new, short_volatility_new, long_sharpe_new, short_sharpe_new, 
     long_max_dd_new, short_max_dd_new, long_uptrend_success_new, short_downtrend_success_new) = evaluate_label_quality(df_new.copy(), horizon=2400)
    
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
    logging.info(f"Logic mới - Sharpe Ratio: LONG={long_sharpe_new:.4f}, SHORT={short_sharpe_new:.4f}")
    logging.info(f"Logic mới - Max Drawdown: LONG={long_max_dd_new:.4f}, SHORT={short_max_dd_new:.4f}")
    logging.info(f"Logic mới - Tỷ lệ thành công theo xu hướng: LONG trong UPTREND={long_uptrend_success_new:.4f}, SHORT trong DOWNTREND={short_downtrend_success_new:.4f}")
    
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

# Hàm main
if __name__ == "__main__":
    try:
        df = pd.read_csv("binance_BTCUSDT_1h.csv", parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors='coerce')
        df = df.ffill().bfill()
        df = df.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        df_new, best_params = run_label_engineering(df)
        
        df_new.to_csv("data_with_new_labels.csv")
        logging.info("Dữ liệu với nhãn mới đã được lưu vào 'data_with_new_labels.csv'")
    except Exception as e:
        logging.error(f"Error in label engineering: {str(e)}")
        raise