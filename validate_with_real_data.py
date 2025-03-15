import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, awesome_oscillator
from ta.volatility import AverageTrueRange
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from sklearn.model_selection import TimeSeriesSplit

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

symbol = "BTCUSDT"
interval = "5m"

# Định nghĩa lại focal loss
def focal_loss(gamma=1.0, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(loss, axis=-1)
    return focal_loss_fixed

# Load scaler và mô hình
scaler = pd.read_pickle("scaler.pkl")
model = load_model("optimized_model.keras", custom_objects={"focal_loss_fixed": focal_loss(gamma=1.0, alpha=0.5)})

# Hàm lấy dữ liệu thực tế
def get_historical_data(file_path="binance_BTCUSDT_5m.csv"):
    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors='coerce')
        df = df.ffill().bfill()
        logging.info(f"Kích thước dữ liệu: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        return None

# Thêm đặc trưng
def add_features(df):
    min_length = 34
    if len(df) < min_length:
        logging.warning(f"Data length {len(df)} is too short for feature calculation. Minimum required: {min_length}")
        return None
    df = df.copy()
    df["price_change"] = df["close"].pct_change().fillna(0)
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi().shift(1).fillna(50)
    df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range().shift(1).fillna(0)
    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx.adx().shift(1).fillna(25)
    df["adx_pos"] = adx.adx_pos().shift(1).fillna(0)
    df["adx_neg"] = adx.adx_neg().shift(1).fillna(0)
    df["adx_signal"] = np.where(df["adx_pos"] > df["adx_neg"], 1, -1)
    df["stoch"] = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3).stoch().shift(1).fillna(50)
    df["momentum"] = df["close"].diff(5).shift(1).fillna(0)
    df["momentum"] = np.log1p(df["momentum"].abs()) * np.sign(df["momentum"])
    df["awesome"] = awesome_oscillator(high=df["high"], low=df["low"], window1=5, window2=34).shift(1).fillna(0)
    df["awesome"] = np.log1p(df["awesome"].abs()) * np.sign(df["awesome"])
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df

# Định nghĩa nhãn thực tế
def define_real_target(df, threshold=0.001, horizon=5):
    df = df.copy()
    # Sử dụng chỉ báo kỹ thuật để xác định xu hướng thay vì giá trong tương lai
    df["trend"] = np.where(df["adx_signal"] == 1, 1, -1)  # Sử dụng tín hiệu ADX
    df["real_target"] = df["trend"].shift(1).fillna(0)  # Dịch chuyển để tránh thông tin từ tương lai
    return df

# Chuẩn bị dữ liệu cho dự đoán
def prepare_prediction_data(df, features, timesteps=40):
    X = []
    data = df[features].values
    data_scaled = scaler.transform(data)
    for i in range(timesteps, len(df)):
        X.append(data_scaled[i-timesteps:i])
    X = np.array(X)
    return X

# Mô phỏng tín hiệu và tính lợi nhuận
def simulate_signals(df, model, scaler, initial_capital=100, trade_percentage=0.10, fee_rate=0.0002, slippage_rate=0.0001, take_profit_percent=0.2, stop_loss_percent=0.1):
    features = ["price_change", "rsi", "atr", "adx", "adx_pos", "adx_neg", "stoch", "momentum", "awesome", "hour", "dayofweek"]
    X = prepare_prediction_data(df, features, timesteps=40)
    predictions = model.predict(X, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1) - 1
    confidences = np.max(predictions, axis=1)

    # Hàm xác định đòn bẩy động
    def determine_leverage(confidence):
        if confidence < 0.7:
            return 5
        elif confidence < 0.8:
            return 7
        elif confidence < 0.9:
            return 10
        else:
            return 10

    # Áp dụng bộ lọc và tính lợi nhuận
    signals = []
    capital = initial_capital
    total_profit = 0
    near_threshold_count = 0
    short_signals_by_date = []
    long_signals_by_date = []
    all_signals_by_date = []
    for i in range(len(predicted_classes)):
        threshold = 0.63
        reason = "PASS"
        profit_usd = 0

        latest_adx = df["adx"].values[i + 40]
        if latest_adx < 15:
            reason = "ADX too low"

        if confidences[i] <= threshold:
            reason = "Confidence too low"
            if confidences[i] > threshold - 0.05:
                near_threshold_count += 1

        timestamp = df.index[i + 40]
        date = timestamp.date()
        current_price = df["close"].values[i + 40]

        if confidences[i] > threshold and predicted_classes[i] != 0 and latest_adx >= 15:
            leverage = determine_leverage(confidences[i])
            amount_usd = capital * trade_percentage
            if amount_usd > capital:
                amount_usd = capital
            if amount_usd <= 0:
                signals.append((timestamp, "NO_SIGNAL", confidences[i], current_price, "Insufficient capital", 0))
                continue

            amount_btc = amount_usd / (current_price * leverage)
            adjusted_entry_price = current_price * (1 + slippage_rate if predicted_classes[i] == 1 else 1 - slippage_rate)
            signal = "LONG" if predicted_classes[i] == 1 else "SHORT"

            # Tính TP và SL
            tp_price = adjusted_entry_price * (1 + take_profit_percent) if signal == "LONG" else adjusted_entry_price * (1 - take_profit_percent)
            sl_price = adjusted_entry_price * (1 - stop_loss_percent) if signal == "LONG" else adjusted_entry_price * (1 + stop_loss_percent)

            # Kiểm tra nhãn thực tế để mô phỏng kết quả
            real_target = df["real_target"].values[i]
            if real_target == predicted_classes[i] and real_target != 0:
                exit_price = tp_price
                profit_percent = ((exit_price - adjusted_entry_price) / adjusted_entry_price) * 100 * leverage if signal == "LONG" else ((adjusted_entry_price - exit_price) / adjusted_entry_price) * 100 * leverage
                entry_fee = adjusted_entry_price * amount_btc * fee_rate
                exit_fee = exit_price * amount_btc * fee_rate
                slippage_in = adjusted_entry_price * amount_btc * slippage_rate
                slippage_out = exit_price * amount_btc * slippage_rate
                profit_usd = (profit_percent / 100) * (adjusted_entry_price * amount_btc) - (entry_fee + exit_fee + slippage_in + slippage_out)
                capital += profit_usd
                total_profit += profit_usd
            elif real_target != 0 and real_target != predicted_classes[i]:
                exit_price = sl_price
                profit_percent = ((exit_price - adjusted_entry_price) / adjusted_entry_price) * 100 * leverage if signal == "LONG" else ((adjusted_entry_price - exit_price) / adjusted_entry_price) * 100 * leverage
                entry_fee = adjusted_entry_price * amount_btc * fee_rate
                exit_fee = exit_price * amount_btc * fee_rate
                slippage_in = adjusted_entry_price * amount_btc * slippage_rate
                slippage_out = exit_price * amount_btc * slippage_rate
                profit_usd = (profit_percent / 100) * (adjusted_entry_price * amount_btc) - (entry_fee + exit_fee + slippage_in + slippage_out)
                capital += profit_usd
                total_profit += profit_usd

            signals.append((timestamp, signal, confidences[i], current_price, reason, profit_usd))
            if signal == "SHORT":
                short_signals_by_date.append(date)
            else:
                long_signals_by_date.append(date)
            all_signals_by_date.append(date)
        else:
            signals.append((timestamp, "NO_SIGNAL", confidences[i], current_price, reason, 0))

    logging.info(f"Number of signals near threshold (within 0.05): {near_threshold_count}")
    logging.info(f"Final capital: {capital:.2f} USD, Total profit: {total_profit:.2f} USD")
    return signals, predicted_classes, total_profit

# Đánh giá tín hiệu
def evaluate_simulation(signals, real_targets):
    predicted_signals = [s[1] for s in signals if s[1] != "NO_SIGNAL"]
    predicted_classes = [-1 if s == "SHORT" else 1 for s in predicted_signals]

    indices = [i for i, s in enumerate(signals) if s[1] != "NO_SIGNAL"]
    real_classes = [real_targets[i] for i in indices if real_targets[i] in [-1, 1]]
    predicted_classes = [predicted_classes[i] for i in range(len(predicted_classes)) if real_targets[indices[i]] in [-1, 1]]

    if len(predicted_classes) == 0 or len(real_classes) == 0:
        logging.warning("No valid signals predicted for evaluation after filtering NEUTRAL labels.")
        return None, None

    cm = confusion_matrix(real_classes, predicted_classes, labels=[-1, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["SHORT", "LONG"], yticklabels=["SHORT", "LONG"])
    plt.title("Confusion Matrix (Simulated Signals)")
    plt.savefig("confusion_matrix_simulated.png")
    plt.close()

    report = classification_report(real_classes, predicted_classes, target_names=["SHORT", "LONG"], labels=[-1, 1], zero_division=0)
    logging.info(f"Classification Report (Simulated Signals):\n{report}")
    return cm, report

# Pipeline chính với Walk-Forward Testing
def main():
    logging.info("Starting simulation test with initial capital 100 USD...")
    df = get_historical_data()
    if df is None or len(df) < 40:
        logging.error("Not enough historical data.")
        return

    df = add_features(df)
    if df is None:
        logging.error("Feature calculation failed.")
        return

    df = define_real_target(df, threshold=0.001)

    tscv = TimeSeriesSplit(n_splits=5)
    all_signals = []
    all_real_targets = []
    total_profit = 0

    for train_index, test_index in tscv.split(df):
        train_df, test_df = df.iloc[train_index].copy(), df.iloc[test_index].copy()

        # Chuẩn hóa lại scaler cho mỗi tập train
        train_features = ["price_change", "rsi", "atr", "adx", "adx_pos", "adx_neg", "stoch", "momentum", "awesome", "hour", "dayofweek"]
        train_data = train_df[train_features].values
        scaler.fit(train_data)

        # Mô phỏng trên tập test
        signals, predicted_classes, profit = simulate_signals(
            test_df,
            model,
            scaler,
            initial_capital=100,
            trade_percentage=0.10,
            fee_rate=0.0002,
            slippage_rate=0.0001,
            take_profit_percent=0.03,
            stop_loss_percent=0.02
        )
        total_profit += profit
        all_signals.extend(signals)
        all_real_targets.extend(test_df["real_target"].values[-len(signals):])

    logging.info(f"Total signals generated: {len(all_signals)}")
    logging.info(f"Valid signals (after filters): {len([s for s in all_signals if s[1] != 'NO_SIGNAL'])}")

    for timestamp, signal, confidence, price, reason, profit in all_signals:
        if signal != "NO_SIGNAL":
            logging.info(f"Time: {timestamp}, Signal: {signal}, Confidence: {confidence:.2f}, Price: {price:.2f}, Reason: {reason}, Profit: {profit:.2f}")
        else:
            logging.debug(f"Time: {timestamp}, Signal: {signal}, Confidence: {confidence:.2f}, Price: {price:.2f}, Reason: {reason}, Profit: 0.00")

    cm, report = evaluate_simulation(all_signals, all_real_targets)
    if cm is not None and report is not None:
        logging.info(f"Confusion Matrix saved as 'confusion_matrix_simulated.png'")
        logging.info(f"Total profit after simulation: {total_profit:.2f} USD with initial capital 100 USD")

if __name__ == "__main__":
    main()