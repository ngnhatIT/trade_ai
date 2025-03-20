import pandas as pd
import numpy as np
import logging
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calc_atr(df):
    return AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=5).average_true_range().shift(1).fillna(0)

def define_target_new(df, horizon=5, atr_multiplier=0.1):
    logging.info(f"Bắt đầu định nghĩa nhãn (horizon={horizon} giờ)...")
    df = df.copy()
    
    df["ema_10"] = EMAIndicator(close=df["close"], window=10).ema_indicator()
    df["atr"] = calc_atr(df)
    
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    long_threshold = 0.001  # +0.1%
    short_threshold = -0.001  # -0.1%
    
    df["threshold"] = df["atr"] * atr_multiplier
    
    df["target"] = 1
    for i in range(len(df)):
        curr_close = df["close"].iloc[i]
        curr_ema_10 = df["ema_10"].iloc[i]
        curr_threshold = df["threshold"].iloc[i]
        curr_future_return = df["future_return"].iloc[i]
        
        if (curr_close > curr_ema_10 + curr_threshold and curr_future_return >= long_threshold):
            df.iloc[i, df.columns.get_loc("target")] = 2
        elif (curr_close < curr_ema_10 - curr_threshold and curr_future_return <= short_threshold):
            df.iloc[i, df.columns.get_loc("target")] = 0
    
    df = df.drop(columns=["ema_10", "atr", "threshold", "future_return"], errors='ignore').dropna()
    return df

def evaluate_label_quality(df, horizon=5, transaction_cost=0.001):
    logging.info(f"Đánh giá chất lượng nhãn (horizon={horizon} giờ, chi phí giao dịch={transaction_cost*100:.2f}%)...")
    df = df.copy()
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
    
    long_sharpe = (long_avg_return / long_volatility) if long_volatility > 0 else 0
    short_sharpe = (short_avg_return / short_volatility) if short_volatility > 0 else 0
    
    def calc_max_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min() if not drawdown.empty else 0
    
    long_max_dd = calc_max_drawdown(long_returns_adj)
    short_max_dd = calc_max_drawdown(short_returns_adj)
    
    df["ema_10"] = EMAIndicator(close=df["close"], window=10).ema_indicator()
    df.loc[:, "trend"] = np.where(df["close"] > df["ema_10"], "UPTREND", 
                                  np.where(df["close"] < df["ema_10"], "DOWNTREND", "NEUTRAL"))
    long_uptrend_success = (df.loc[(df["target"] == 2) & (df["trend"] == "UPTREND"), "future_return"] - 2 * transaction_cost > 0).mean() if not df[(df["target"] == 2) & (df["trend"] == "UPTREND")].empty else 0
    short_downtrend_success = (-df.loc[(df["target"] == 0) & (df["trend"] == "DOWNTREND"), "future_return"] - 2 * transaction_cost > 0).mean() if not df[(df["target"] == 0) & (df["trend"] == "DOWNTREND")].empty else 0
    
    label_counts = df["target"].value_counts()
    total_samples = len(df)
    label_ratios = label_counts / total_samples
    
    logging.info(f"Phân phối nhãn - Số lượng: {label_counts}")
    logging.info(f"Phân phối nhãn - Tỷ lệ: {label_ratios}")
    logging.info(f"Lợi nhuận trung bình (sau {horizon} giờ, đã trừ chi phí giao dịch):")
    logging.info(f"LONG: {long_avg_return:.4f}, Tỷ lệ thành công: {long_success_ratio:.4f}, Độ biến động: {long_volatility:.4f}")
    logging.info(f"SHORT: {short_avg_return:.4f}, Tỷ lệ thành công: {short_success_ratio:.4f}, Độ biến động: {short_volatility:.4f}")
    logging.info(f"NEUTRAL: {neutral_avg_return:.4f}")
    logging.info(f"Sharpe Ratio - LONG: {long_sharpe:.4f}, SHORT: {short_sharpe:.4f}")
    logging.info(f"Max Drawdown - LONG: {long_max_dd:.4f}, SHORT: {short_max_dd:.4f}")
    logging.info(f"Tỷ lệ thành công theo xu hướng - LONG trong UPTREND: {long_uptrend_success:.4f}, SHORT trong DOWNTREND: {short_downtrend_success:.4f}")
    
    evaluation = {
        "label_distribution_ok": (0.20 <= label_ratios.get(2, 0) <= 0.25) and (0.20 <= label_ratios.get(0, 0) <= 0.25) and (0.50 <= label_ratios.get(1, 0) <= 0.60),
        "long_profit_ok": long_avg_return > 0,
        "short_profit_ok": short_avg_return > 0,
        "long_success_ok": long_success_ratio >= 0.6,
        "short_success_ok": short_success_ratio >= 0.6,
        "long_trend_ok": long_uptrend_success >= 0.7,
        "short_trend_ok": short_downtrend_success >= 0.7,
        "volatility_ok": long_volatility <= 0.1 and short_volatility <= 0.1,
        "sharpe_ok": long_sharpe >= 0.5 and short_sharpe >= 0,
        "drawdown_ok": abs(long_max_dd) <= 0.2 and abs(short_max_dd) <= 0.2
    }
    
    logging.info("\nĐánh giá tổng quan:")
    for key, value in evaluation.items():
        logging.info(f"{key}: {'OK' if value else 'NOT OK'}")
    
    return df, evaluation

def run_label_engineering(df, define_horizon=5, eval_horizon=5, train_end_date="2024-12-31"):
    logging.info(f"Dữ liệu ban đầu - Số lượng mẫu: {len(df)}")
    logging.info(f"Dữ liệu ban đầu - Thời gian bắt đầu: {df.index.min()}, Thời gian kết thúc: {df.index.max()}")
    
    train_df = df[df.index <= pd.to_datetime(train_end_date)].copy()
    test_df = df[df.index > pd.to_datetime(train_end_date)].copy()
    
    logging.info(f"Tập huấn luyện - Số lượng mẫu: {len(train_df)}, Thời gian: {train_df.index.min()} - {train_df.index.max()}")
    logging.info(f"Tập kiểm tra - Số lượng mẫu: {len(test_df)}, Thời gian: {test_df.index.min()} - {test_df.index.max()}")
    
    train_df_labeled = define_target_new(train_df, horizon=define_horizon)
    test_df_labeled = define_target_new(test_df, horizon=define_horizon)
    logging.info(f"\nĐánh giá nhãn trên tập kiểm tra với define_horizon={define_horizon} giờ và eval_horizon={eval_horizon} giờ:")
    test_df_evaluated, evaluation = evaluate_label_quality(test_df_labeled, horizon=eval_horizon)
    
    train_df_labeled.to_csv("train_data_with_labels.csv")
    test_df_evaluated.to_csv("test_data_with_labels.csv")
    logging.info("Dữ liệu huấn luyện với nhãn đã được lưu vào 'train_data_with_labels.csv'")
    logging.info("Dữ liệu kiểm tra với nhãn đã được lưu vào 'test_data_with_labels.csv'")
    
    return train_df_labeled, test_df_evaluated, evaluation

if __name__ == "__main__":
    try:
        df = pd.read_csv("binance_BTCUSDT_1h.csv", parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors='coerce')
        df = df.ffill().bfill()
        df = df.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        train_df_labeled, test_df_labeled, evaluation = run_label_engineering(df, define_horizon=5, eval_horizon=5)
    except Exception as e:
        logging.error(f"Error in label engineering: {str(e)}")
        raise