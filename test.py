import requests
import pandas as pd
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, awesome_oscillator
from ta.volatility import AverageTrueRange
import logging
import ccxt
import csv
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

exchange = ccxt.binance({'enableRateLimit': True})
symbol = "BTC/USDT"
interval = "5m"
base_timesteps = 20
lookback_hours = 24
initial_capital = 1000
capital = initial_capital
trade_percentage = 0.10
fee_rate = 0.0002
slippage_rate = 0.0001
risk_per_trade = 0.02
confidence_threshold = 0.65  # Đã tăng từ 0.6 lên 0.65
base_confidence_threshold = 0.7
base_tp_ratio = 0.8
base_sl_ratio = 0.3
base_leverage = 10
max_leverage = 20
adx_threshold = 5
rsi_overbought = 70
rsi_oversold = 30
max_loss_pause_threshold = 100
max_drawdown_limit = 20
min_capital_threshold = initial_capital * 0.5

trade_history = []
capital_history = []
market_data_history = []
telegram_log = []
current_position = None
total_trades = 0
winning_trades = 0
daily_trades = 0
daily_profit = 0
last_day = None
is_trading_paused = False
last_drawdown_check = time.time()
hourly_stats = {f"{i:02d}h-{((i+6)%24):02d}h": {"trades": 0, "profit": 0} for i in range(0, 24, 6)}
total_loss = 0
daily_loss = 0  # Khởi tạo giá trị ban đầu cho daily_loss
short_signals_by_date = []
long_signals_by_date = []
all_signals_by_date = []
signals_log = []
last_candle_time = None
last_loss_reset_time = time.time()
last_reentry_time = None
hedge_position = None

def focal_loss(gamma=1.0, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(loss, axis=-1)
    return focal_loss_fixed

scaler = pd.read_pickle("scaler.pkl")
model = load_model("optimized_model.keras", custom_objects={"focal_loss_fixed": focal_loss(gamma=1.0, alpha=0.5)})
data_scaled_buffer = None

# Hàm lấy dữ liệu lịch sử
def get_historical_data(symbol, interval, lookback_hours=24, max_retries=5):
    url = "https://api.binance.com/api/v3/klines"
    data = []
    end_time = int(time.time() * 1000)
    start_time = int((datetime.fromtimestamp(end_time / 1000) - timedelta(hours=lookback_hours)).timestamp() * 1000)
    limit = 1000
    retries = 0
    while start_time < end_time and retries < max_retries:
        try:
            params = {"symbol": symbol.replace("/", ""), "interval": interval, "startTime": start_time, "endTime": end_time, "limit": limit}
            response = requests.get(url, params=params)
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            klines = response.json()
            if not klines:
                break
            data.extend(klines)
            current_time = int(klines[-1][6]) + 1
            if current_time <= start_time:
                break
            start_time = current_time
            time.sleep(1)
        except Exception as e:
            retries += 1
            wait_time = min(2 ** retries + random.uniform(0, 1), 60)
            logging.warning(f"Retry {retries}/{max_retries} after error: {e}. Waiting {wait_time}s")
            time.sleep(wait_time)
    if not data:
        logging.error("No historical data fetched.")
        return None
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors='coerce')
    df = df.ffill().bfill()
    logging.info(f"Fetched {len(df)} historical data points")
    return df

def get_current_price(symbol, signal=None, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            check_rate_limit()
            ticker = exchange.fetch_ticker(symbol)
            if signal == "LONG":
                price = ticker['ask']
            elif signal == "SHORT":
                price = ticker['bid']
            else:
                price = ticker['last']
            if price is None:
                raise ValueError("Could not fetch current price")
            logging.info(f"Fetched current price: {price:.2f} USDT")
            return price
        except Exception as e:
            retries += 1
            wait_time = min(2 ** retries + random.uniform(0, 1), 60)
            logging.warning(f"Retry {retries}/{max_retries} after error: {e}. Waiting {wait_time}s")
            time.sleep(wait_time)
    logging.error("Max retries reached. Failed to fetch current price.")
    return None

def get_realtime_data(symbol, interval, historical_df, max_retries=10):
    retries = 0
    while retries < max_retries:
        try:
            check_rate_limit()
            end_time = exchange.milliseconds()
            if len(historical_df) == 0:
                start_time = end_time - (5 * 60 * 1000)
            else:
                start_time = historical_df.index[-1].value + 1
            ohlcv = exchange.fetch_ohlcv(symbol, interval, since=start_time, limit=1)
            if ohlcv and len(ohlcv) > 0:
                new_data = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                new_data["timestamp"] = pd.to_datetime(new_data["timestamp"], unit="ms")
                new_data.set_index("timestamp", inplace=True)
                updated_df = pd.concat([historical_df.iloc[-100:], new_data]).drop_duplicates()
                market_data_history.append(new_data.iloc[-1].to_dict())
                return updated_df
            return historical_df
        except Exception as e:
            retries += 1
            wait_time = min(2 ** retries + random.uniform(0, 1), 60)
            logging.warning(f"Retry {retries}/{max_retries} after error: {e}. Waiting {wait_time}s")
            time.sleep(wait_time)
    logging.error("Max retries reached. Failed to fetch realtime data.")
    return historical_df

def check_rate_limit():
    rate_limit = exchange.rateLimit
    last_call = getattr(exchange, 'last_call', 0)
    current_time = time.time() * 1000
    if current_time - last_call < rate_limit:
        wait_time = (rate_limit - (current_time - last_call)) / 1000 + 1
        logging.info(f"Rate limit hit. Waiting {wait_time}s")
        time.sleep(wait_time)
    exchange.last_call = current_time

def add_features_incremental(df, prev_df=None):
    global data_scaled_buffer
    if prev_df is not None and len(df) > len(prev_df):
        new_df = df.iloc[len(prev_df):].copy()
    else:
        new_df = df.copy()
    if len(new_df) < 1:
        return None
    
    if prev_df is not None:
        new_df = pd.concat([prev_df.iloc[-34:], new_df]).drop_duplicates()
    min_length = 34
    if len(new_df) < min_length:
        logging.warning(f"Data length {len(new_df)} is too short for feature calculation. Minimum required: {min_length}")
        return None
    
    new_df["price_change"] = new_df["close"].pct_change().fillna(0)
    new_df["rsi"] = RSIIndicator(close=new_df["close"], window=14).rsi().shift(1).ffill().iloc[-1]
    new_df["atr"] = AverageTrueRange(high=new_df["high"], low=new_df["low"], close=new_df["close"], window=14).average_true_range().shift(1).ffill().iloc[-1]
    adx = ADXIndicator(high=new_df["high"], low=new_df["low"], close=new_df["close"], window=14)
    new_df["adx"] = adx.adx().shift(1).ffill().iloc[-1]
    new_df["adx_pos"] = adx.adx_pos().shift(1).ffill().iloc[-1]
    new_df["adx_neg"] = adx.adx_neg().shift(1).ffill().iloc[-1]
    new_df["adx_signal"] = np.where(new_df["adx_pos"] > new_df["adx_neg"], 1, -1)
    new_df["stoch"] = StochasticOscillator(high=new_df["high"], low=new_df["low"], close=new_df["close"], window=14, smooth_window=3).stoch().shift(1).ffill().iloc[-1]
    new_df["momentum"] = new_df["close"].diff(5).shift(1).ffill().iloc[-1]
    new_df["awesome"] = awesome_oscillator(high=new_df["high"], low=new_df["low"], window1=5, window2=34).shift(1).ffill().iloc[-1]
    new_df["awesome"] = np.log1p(new_df["awesome"].abs()) * np.sign(new_df["awesome"])
    new_df["hour"] = new_df.index.hour
    new_df["dayofweek"] = new_df.index.dayofweek
    new_df = new_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    features = ["price_change", "rsi", "atr", "adx", "adx_pos", "adx_neg", "stoch", "momentum", "awesome", "hour", "dayofweek"]
    if data_scaled_buffer is None or len(data_scaled_buffer) < base_timesteps:
        data_scaled_buffer = scaler.transform(new_df[features].values[-base_timesteps:])
    else:
        new_data = scaler.transform(new_df[features].values[-1:])
        data_scaled_buffer = np.roll(data_scaled_buffer, -1, axis=0)
        data_scaled_buffer[-1] = new_data[-1]
    
    if prev_df is not None:
        return pd.concat([prev_df.iloc[-base_timesteps:], new_df]).drop_duplicates()
    return new_df

def prepare_prediction_data(df, features, timesteps=20):
    global data_scaled_buffer
    if data_scaled_buffer is None or len(data_scaled_buffer) < timesteps:
        return None
    return data_scaled_buffer.reshape(1, timesteps, len(features))

def determine_leverage(confidence, latest_atr, entry_price, latest_adx):
    atr_ratio = latest_atr / entry_price
    adx_factor = latest_adx / 20
    leverage = base_leverage * (1 + atr_ratio * adx_factor)
    if confidence < 0.7:
        leverage = min(leverage, 10)
    elif confidence < 0.8:
        leverage = min(leverage, 15)
    elif confidence < 0.9:
        leverage = min(leverage, 20)
    else:
        leverage = min(leverage, max_leverage)
    if latest_atr > 300:
        leverage = min(leverage, 10)
    return leverage

def calculate_sl_tp(entry_price, signal, leverage, latest_atr, confidence):
    adjusted_entry_price = entry_price * (1 + slippage_rate if signal == "LONG" else 1 - slippage_rate)
    atr_ratio = latest_atr / entry_price
    atr_factor = max(1.0, 2 * atr_ratio)

    # TP linh động
    tp_base = base_tp_ratio
    tp_confidence_factor = 1 + 0.5 * max(0, confidence - 0.7)
    tp_atr_factor = 1 + 0.1 * max(0, (latest_atr - 200) / 100)
    adjusted_tp_ratio = tp_base * tp_confidence_factor * tp_atr_factor
    tp_price_change = adjusted_tp_ratio / leverage

    # SL linh động
    sl_base = max(0.3, 1.5 * atr_ratio)
    sl_confidence_factor = 1 + 0.2 * max(0, 0.7 - confidence)
    adjusted_sl_ratio = sl_base * sl_confidence_factor
    sl_price_change = adjusted_sl_ratio / leverage

    if signal == "LONG":
        tp_price = adjusted_entry_price * (1 + tp_price_change)
        sl_price = adjusted_entry_price * (1 - sl_price_change)
    else:  # SHORT
        tp_price = adjusted_entry_price * (1 - tp_price_change)
        sl_price = adjusted_entry_price * (1 + sl_price_change)

    logging.info(f"Calculated SL: {sl_price:.2f}, TP: {tp_price:.2f} for signal: {signal}, Entry: {entry_price:.2f}, Leverage: {leverage}x, TP Change: {tp_price_change*100:.2f}%, SL Change: {sl_price_change*100:.2f}%")
    return sl_price, tp_price

def calculate_trailing_stop(entry_price, current_price, latest_atr, signal, sl_price, profit_percent, tp_price):
    atr_ratio = latest_atr / entry_price
    trailing_stop_factor = max(0.005, atr_ratio * (1 + 0.5 * max(0, profit_percent / 2)))
    if signal == "LONG":
        initial_trailing_stop = entry_price * (1 - trailing_stop_factor)
        trailing_stop_price = max(initial_trailing_stop, current_price * (1 - trailing_stop_factor))
        trailing_stop_price = max(trailing_stop_price, sl_price)
    else:
        initial_trailing_stop = entry_price * (1 + trailing_stop_factor)
        trailing_stop_price = min(initial_trailing_stop, current_price * (1 + trailing_stop_factor))
        trailing_stop_price = min(trailing_stop_price, sl_price)
    logging.info(f"Trailing Stop calculated: {trailing_stop_price:.2f}, Factor: {trailing_stop_factor:.4f}, Current Price: {current_price:.2f}")
    return trailing_stop_price, trailing_stop_factor

def adjust_sl_to_breakeven(entry_price, current_price, signal, leverage, sl_price, profit_percent):
    min_profit_threshold = 1.0
    adjusted_profit_percent = profit_percent * leverage
    if adjusted_profit_percent >= min_profit_threshold:
        if signal == "LONG":
            new_sl = entry_price
            sl_price = max(sl_price, new_sl)
        else:
            new_sl = entry_price
            sl_price = min(sl_price, new_sl)
        logging.info(f"SL adjusted to breakeven: {sl_price:.2f}")
    return sl_price

def calculate_pnl(entry_price, current_price, signal, leverage, fee_rate, slippage_rate, amount_usd=100):
    adjusted_entry_price = entry_price * (1 + slippage_rate if signal == "LONG" else 1 - slippage_rate)
    notional_value = amount_usd * leverage
    if signal == "LONG":
        price_diff = (current_price - adjusted_entry_price) / adjusted_entry_price
    else:  # SHORT
        price_diff = (adjusted_entry_price - current_price) / adjusted_entry_price
    profit_percent = price_diff * 100
    profit_before_fee = (profit_percent / 100) * notional_value
    fee = (notional_value * fee_rate) * 2
    profit_usdt = profit_before_fee - fee
    adjusted_exit_price = current_price
    position_value = notional_value
    logging.debug(f"Debug PnL: entry={adjusted_entry_price:.2f}, exit={current_price:.2f}, diff={price_diff*100:.4f}%, leverage={leverage}, value={position_value:.2f}, fee={fee:.2f}, profit={profit_usdt:.2f}")
    return profit_percent, profit_usdt, adjusted_entry_price, adjusted_exit_price, position_value

def generate_signal(df):
    features = ["price_change", "rsi", "atr", "adx", "adx_pos", "adx_neg", "stoch", "momentum", "awesome", "hour", "dayofweek"]
    X = prepare_prediction_data(df, features)
    if X is None or len(X) == 0:
        logging.warning("No valid data for prediction.")
        return "NO_SIGNAL", 0.0, 0
    logging.info(f"Features values: {df[features].iloc[-1].to_dict()}")
    latest_adx = df["adx"].iloc[-1]
    latest_atr = df["atr"].iloc[-1]
    latest_price = df["close"].iloc[-1]
    atr_ratio = latest_atr / latest_price
    latest_price_change = df["price_change"].iloc[-1]
    avg_price_change = df["price_change"].iloc[-base_timesteps:].mean()
    short_momentum = df["momentum"].iloc[-5:].mean()
    ma_50 = df["close"].rolling(window=50).mean().iloc[-1]
    current_trend = "UP" if latest_price > ma_50 else "DOWN"
    adjusted_threshold = base_confidence_threshold * (1 - atr_ratio)
    adjusted_threshold = max(0.1, min(adjusted_threshold, 0.25))
    predictions = model.predict(X, verbose=0)
    predicted_class = np.argmax(predictions[-1]) - 1
    confidence = np.max(predictions[-1])
    logging.info(f"Prediction: Class={predicted_class}, Confidence={confidence}, Adjusted Threshold={adjusted_threshold}")
    min_price_change_threshold = 0.00005
    reason = "PASS"

    # Điều chỉnh confidence_threshold
    confidence_threshold = 0.65  # Tăng từ 0.6 lên 0.65

    # Điều kiện phụ: nếu confidence trong khoảng [0.6, 0.65) và tín hiệu phù hợp xu hướng
    if 0.6 <= confidence < confidence_threshold:
        if (predicted_class == 1 and current_trend == "UP") or (predicted_class == -1 and current_trend == "DOWN"):
            confidence_threshold = 0.6  # Giảm ngưỡng để chấp nhận tín hiệu nếu phù hợp xu hướng
            logging.info(f"Lowering confidence_threshold to 0.6 due to trend alignment: {current_trend}")

    if confidence < confidence_threshold:
        reason = f"Confidence {confidence} < {confidence_threshold}"
        logging.info(f"No signal: {reason}")
        return "NO_SIGNAL", confidence, predicted_class
    if (predicted_class == 1 and (df["rsi"].iloc[-1] > rsi_overbought or df["adx"].iloc[-1] < adx_threshold)):
        reason = "LONG rejected due to RSI or ADX"
        logging.info(f"No signal: {reason}")
        return "NO_SIGNAL", confidence, predicted_class
    elif (predicted_class == -1 and (df["rsi"].iloc[-1] < rsi_oversold or df["adx"].iloc[-1] < adx_threshold)):
        reason = "SHORT rejected due to RSI or ADX"
        logging.info(f"No signal: {reason}")
        return "NO_SIGNAL", confidence, predicted_class
    if (predicted_class == 1 and current_trend == "DOWN") or (predicted_class == -1 and current_trend == "UP"):
        reason = f"Predicted class {predicted_class} against trend {current_trend}"
        logging.info(f"No signal: {reason}")
        return "NO_SIGNAL", confidence, predicted_class
    if (confidence > adjusted_threshold and predicted_class != 0 and latest_adx >= 5 and
        abs(latest_price_change) > min_price_change_threshold and
        ((predicted_class == 1 and avg_price_change >= -0.0001 and short_momentum >= -0.0001) or
         (predicted_class == -1 and avg_price_change <= 0.0001 and short_momentum <= 0.0001))):
        signal = "LONG" if predicted_class == 1 else "SHORT"
        logging.info(f"Signal generated: {signal}, Confidence: {confidence}")
    else:
        reason = "Failed final conditions"
        logging.info(f"No signal: {reason}")
        return "NO_SIGNAL", confidence, predicted_class
    signals_log.append((datetime.now(), signal, confidence, latest_price, reason, predicted_class))
    with open('signals_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["Timestamp", "Signal", "Confidence", "Price", "Reason", "Predicted_Class"])
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), signal, confidence, latest_price, reason, predicted_class])
    return signal, confidence, predicted_class

def handle_partial_profit(current_position, entry_price, current_price, signal, leverage, fee_rate, slippage_rate, amount_usd, capital, total_loss, daily_loss, tp_price, trade_history, hedge_position):
    profit_percent, profit_usdt, _, _, _ = calculate_pnl(entry_price, current_price, signal, leverage, fee_rate, slippage_rate, amount_usd)
    half_tp = tp_price * 0.5 if signal == "LONG" else tp_price * 1.5
    partial_profit_taken = False
    if signal == "LONG" and current_price >= half_tp and amount_usd > 0:
        partial_amount_usd = amount_usd * 0.5
        _, partial_profit, _, _, _ = calculate_pnl(entry_price, current_price, signal, leverage, fee_rate, slippage_rate, partial_amount_usd)
        if partial_profit > 1.0:
            capital += partial_profit
            total_loss += partial_profit
            daily_loss += partial_profit
            amount_usd *= 0.5
            trade_history.append((datetime.now(), entry_price, f"{signal}_PARTIAL_50%", current_price, partial_profit, capital, leverage))
            send_telegram_message(f"Partial profit taken (50%) for {signal} at {current_price:.2f}, PnL: {partial_profit:.2f} USD")
            partial_profit_taken = True
            if hedge_position:
                hedge_signal, hedge_entry_price, hedge_amount, hedge_leverage, _, _, _, _, hedge_amount_usd = hedge_position
                _, hedge_profit_usdt, _, hedge_exit_price, _ = calculate_pnl(hedge_entry_price, current_price, hedge_signal, hedge_leverage, fee_rate, slippage_rate, hedge_amount_usd)
                if hedge_profit_usdt > 0:
                    capital += hedge_profit_usdt
                    total_loss += hedge_profit_usdt
                    daily_loss += hedge_profit_usdt
                    trade_history.append((datetime.now(), hedge_entry_price, f"{hedge_signal}_HEDGE_CLOSE", hedge_exit_price, hedge_profit_usdt, capital, hedge_leverage))
                    send_telegram_message(f"Hedge closed at 50% TP of main: {hedge_signal}, Entry: {hedge_entry_price:.2f}, Exit: {hedge_exit_price:.2f}, PnL: {hedge_profit_usdt:.2f} USD")
                    hedge_position = None
    elif signal == "SHORT" and current_price <= half_tp and amount_usd > 0:
        partial_amount_usd = amount_usd * 0.5
        _, partial_profit, _, _, _ = calculate_pnl(entry_price, current_price, signal, leverage, fee_rate, slippage_rate, partial_amount_usd)
        if partial_profit > 1.0:
            capital += partial_profit
            total_loss += partial_profit
            daily_loss += partial_profit
            amount_usd *= 0.5
            trade_history.append((datetime.now(), entry_price, f"{signal}_PARTIAL_50%", current_price, partial_profit, capital, leverage))
            send_telegram_message(f"Partial profit taken (50%) for {signal} at {current_price:.2f}, PnL: {partial_profit:.2f} USD")
            partial_profit_taken = True
            if hedge_position:
                hedge_signal, hedge_entry_price, hedge_amount, hedge_leverage, _, _, _, _, hedge_amount_usd = hedge_position
                _, hedge_profit_usdt, _, hedge_exit_price, _ = calculate_pnl(hedge_entry_price, current_price, hedge_signal, hedge_leverage, fee_rate, slippage_rate, hedge_amount_usd)
                if hedge_profit_usdt > 0:
                    capital += hedge_profit_usdt
                    total_loss += hedge_profit_usdt
                    daily_loss += hedge_profit_usdt
                    trade_history.append((datetime.now(), hedge_entry_price, f"{hedge_signal}_HEDGE_CLOSE", hedge_exit_price, hedge_profit_usdt, capital, hedge_leverage))
                    send_telegram_message(f"Hedge closed at 50% TP of main: {hedge_signal}, Entry: {hedge_entry_price:.2f}, Exit: {hedge_exit_price:.2f}, PnL: {hedge_profit_usdt:.2f} USD")
                    hedge_position = None
    return partial_profit_taken, amount_usd, capital, total_loss, daily_loss, hedge_position

def plot_trade_distribution():
    short_counts = Counter(short_signals_by_date)
    long_counts = Counter(long_signals_by_date)
    all_counts = Counter(all_signals_by_date)
    start_date = min(short_signals_by_date + long_signals_by_date, default=datetime.now().date())
    end_date = max(short_signals_by_date + long_signals_by_date, default=datetime.now().date())
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    short_data = [short_counts.get(date, 0) for date in dates]
    long_data = [long_counts.get(date, 0) for date in dates]
    total_data = [all_counts.get(date, 0) for date in dates]
    plt.figure(figsize=(15, 7))
    plt.plot(dates, short_data, label="SHORT Trades", color="red", linestyle="-")
    plt.plot(dates, long_data, label="LONG Trades", color="green", linestyle="-")
    plt.plot(dates, total_data, label="Total Trades", color="blue", linestyle="--")
    plt.title("Daily Trade Distribution")
    plt.xlabel("Date")
    plt.ylabel("Number of Trades")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("trade_distribution_realtime.png")
    plt.close()
    logging.info("Trade distribution chart saved")

def save_telegram_log():
    with open('telegram_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["Timestamp", "Message", "Status"])
        for log in telegram_log:
            writer.writerow([log[0].strftime('%Y-%m-%d %H:%M:%S'), log[1], log[2]])
    telegram_log.clear()

TELEGRAM_BOT_TOKEN = "7617216154:AAF-5RxHYmn63pC2BgGJTAMRm2ehO4HcZvA"
TELEGRAM_CHAT_ID = "2028475238"

def send_telegram_message(message):
    global telegram_log
    timestamp = datetime.now()
    formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    message_with_time = f"[{formatted_time}] {message}"
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message_with_time, "parse_mode": "Markdown"}
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            logging.info(f"Sent Telegram: {message}")
            telegram_log.append((timestamp, message, "Success"))
        else:
            logging.error(f"Telegram error: {response.text}")
            telegram_log.append((timestamp, message, f"Failed: {response.text}"))
    except Exception as e:
        logging.error(f"Telegram send error: {e}")
        telegram_log.append((timestamp, message, f"Failed: {str(e)}"))

def save_trade_history():
    with open('trade_history.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["Timestamp", "Entry_Price", "Signal", "Exit_Price", "Profit_USDT", "Capital_After", "Leverage", "Total_Loss"])
        for trade in trade_history:
            writer.writerow([trade[0].strftime('%Y-%m-%d %H:%M:%S')] + list(trade[1:]) + [total_loss])

def save_capital_history():
    with open('capital_history.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["Timestamp", "Capital", "Total_Loss"])
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), capital, total_loss])

def save_market_data():
    with open('market_data.json', 'a') as f:
        for data in market_data_history[-100:]:
            f.write(json.dumps(data) + '\n')

def calculate_drawdown():
    if not trade_history:
        return 0
    capitals = [trade[5] for trade in trade_history]
    max_capital = max(capitals + [initial_capital])
    min_capital = min(capitals + [initial_capital])
    return (max_capital - min_capital) / max_capital * 100 if max_capital > 0 else 0

def calculate_streaks():
    if not trade_history:
        return 0, 0
    profits = [trade[4] for trade in trade_history]
    longest_win_streak = longest_loss_streak = current_win_streak = current_loss_streak = 0
    for profit in profits:
        if profit > 0:
            current_win_streak += 1
            current_loss_streak = 0
        else:
            current_loss_streak += 1
            current_win_streak = 0
        longest_win_streak = max(longest_win_streak, current_win_streak)
        longest_loss_streak = max(longest_loss_streak, current_loss_streak)
    return longest_win_streak, longest_loss_streak

def send_status_update():
    global daily_trades, daily_profit, last_day, daily_loss  # Khai báo global cho daily_loss
    winrate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    drawdown = calculate_drawdown()
    win_streak, loss_streak = calculate_streaks()
    current_day = datetime.now().date()
    if last_day and current_day != last_day:
        message = f"Daily Report {last_day}: Trades: {daily_trades}, Profit: {daily_profit:.2f} USD, Total Loss: {total_loss:.2f} USD, Daily Loss: {daily_loss:.2f} USD"
        send_telegram_message(message)
        daily_trades = 0
        daily_profit = 0
        daily_loss = 0  # Reset daily_loss khi ngày mới bắt đầu
        last_day = current_day
    current_hour = datetime.now().hour
    hour_slot = f"{(current_hour // 6 * 6):02d}h-{((current_hour // 6 * 6 + 6) % 24):02d}h"
    short_counts = Counter(short_signals_by_date)
    long_counts = Counter(long_signals_by_date)
    all_counts = Counter(all_signals_by_date)
    short_info = f"Most SHORT: {short_counts.most_common(1)[0][0]}, Count: {short_counts.most_common(1)[0][1]}" if short_counts else "No SHORT"
    long_info = f"Most LONG: {long_counts.most_common(1)[0][0]}, Count: {long_counts.most_common(1)[0][1]}" if long_counts else "No LONG"
    trades_info = f"Least Trades: {all_counts.most_common()[-1][0]}, Count: {all_counts.most_common()[-1][1]}" if all_counts else "No trades"
    message = (f"Status: Capital: {capital:.2f} USD, Trades: {total_trades}, Wins: {winning_trades}, Winrate: {winrate:.2f}%, "
               f"Drawdown: {drawdown:.2f}%, Win Streak: {win_streak}, Loss Streak: {loss_streak}, "
               f"Hourly ({hour_slot}): Trades: {hourly_stats[hour_slot]['trades']}, Profit: {hourly_stats[hour_slot]['profit']:.2f} USD, "
               f"Total Loss: {total_loss:.2f} USD, Daily Loss: {daily_loss:.2f} USD, {short_info}, {long_info}, {trades_info}")
    send_telegram_message(message)

def check_api_status(exchange, symbol):
    try:
        exchange.fetch_ticker(symbol)
        logging.info("API connection successful.")
        return True
    except Exception as e:
        logging.error(f"API connection failed: {e}")
        return False

def main():
    global current_position, total_trades, winning_trades, capital, daily_trades, daily_profit, is_trading_paused, last_drawdown_check, total_loss, daily_loss, last_candle_time, data_scaled_buffer, last_loss_reset_time, last_reentry_time, hedge_position
    logging.info(f"Starting with capital: {initial_capital} USD")
    send_telegram_message(f"Starting with capital: {initial_capital} USD")

    if not check_api_status(exchange, symbol):
        logging.error("API failure. Exiting.")
        return

    historical_df = get_historical_data(symbol, interval, lookback_hours=24)
    if historical_df is None or len(historical_df) < base_timesteps + 34:
        logging.error("Insufficient data. Accumulating...")
        current_price = get_current_price(symbol)
        if current_price is None:
            logging.error("Cannot fetch price. Exiting.")
            return
        historical_df = pd.DataFrame({"timestamp": [pd.Timestamp.now()], "open": [current_price], "high": [current_price], "low": [current_price], "close": [current_price], "volume": [0]})
        historical_df.set_index("timestamp", inplace=True)
    while len(historical_df) < base_timesteps + 34:
        new_df = get_realtime_data(symbol, interval, historical_df)
        if new_df is not None and len(new_df) > len(historical_df):
            historical_df = new_df
            logging.info(f"Accumulated {len(historical_df)} candles")
        time.sleep(1)

    historical_df = add_features_incremental(historical_df)
    while historical_df is None:
        logging.warning("Waiting for data...")
        time.sleep(60)
        historical_df = add_features_incremental(historical_df)
        if historical_df is not None:
            break

    last_candle_time = historical_df.index[-1]

    while True:
        current_time = time.time()
        if current_time - last_loss_reset_time >= 86400:
            logging.info(f"Resetting losses: Total {total_loss:.2f}, Daily {daily_loss:.2f}")
            total_loss = 0
            daily_loss = 0
            last_loss_reset_time = current_time

        if time.time() - last_drawdown_check >= 3600:
            drawdown = calculate_drawdown()
            if drawdown > max_drawdown_limit or capital < min_capital_threshold:
                is_trading_paused = True
                send_telegram_message(f"Paused: Drawdown {drawdown:.2f}% or Capital {capital:.2f} < {min_capital_threshold:.2f}")
                time.sleep(3600)
                is_trading_paused = False
                last_drawdown_check = time.time()

        if abs(total_loss) >= max_loss_pause_threshold:
            is_trading_paused = True
            send_telegram_message(f"Paused: Loss {total_loss:.2f} > {max_loss_pause_threshold:.2f}")
            logging.info(f"Paused due to loss: {total_loss:.2f}")
            time.sleep(3600)
            is_trading_paused = False
            continue

        if is_trading_paused:
            time.sleep(300)
            continue

        df = get_realtime_data(symbol, interval, historical_df)
        logging.info(f"Data shape: {df.shape}")
        if df is None:
            time.sleep(300)
            continue
        df = add_features_incremental(df, historical_df)
        logging.info(f"Features shape: {df.shape}")
        if df is None:
            time.sleep(60)
            continue

        save_market_data()

        signal = current_position[0] if current_position else None
        current_price = get_current_price(symbol, signal=signal)
        if current_price is None:
            logging.warning("Using last candle price.")
            current_price = df["close"].iloc[-1]

        current_hour = datetime.now().hour
        hour_slot = f"{(current_hour // 6 * 6):02d}h-{((current_hour // 6 * 6 + 6) % 24):02d}h"
        if hourly_stats[hour_slot]["trades"] > 5 and hourly_stats[hour_slot]["profit"] < 0:
            logging.info(f"Pausing in {hour_slot}: Profit {hourly_stats[hour_slot]['profit']:.2f}")
            time.sleep(300)
            continue

        if current_position:
            signal, entry_price, amount, leverage, trailing_stop_price, sl_price, tp_price, entry_time, amount_usd = current_position
            latest_atr = df["atr"].iloc[-1]
            profit_percent, profit_usdt, _, _, _ = calculate_pnl(entry_price, current_price, signal, leverage, fee_rate, slippage_rate, amount_usd)
            logging.info(f"Position: {signal}, Entry: {entry_price:.2f}, Current: {current_price:.2f}, PnL: {profit_percent:.2f}% ({profit_usdt:.2f} USD)")

            new_trailing_stop, _ = calculate_trailing_stop(entry_price, current_price, latest_atr, signal, sl_price, profit_percent, tp_price)
            trailing_stop_price = new_trailing_stop

            sl_price = adjust_sl_to_breakeven(entry_price, current_price, signal, leverage, sl_price, profit_percent)

            partial_profit_taken, amount_usd, capital, total_loss, daily_loss, hedge_position = handle_partial_profit(
                current_position, entry_price, current_price, signal, leverage, fee_rate, slippage_rate, amount_usd, capital, total_loss, daily_loss, tp_price, trade_history, hedge_position
            )

            new_signal, new_confidence, _ = generate_signal(df)
            signal_opposite = False
            if new_confidence > 0.7 and new_signal and ((signal == "LONG" and new_signal == "SHORT") or (signal == "SHORT" and new_signal == "LONG")):
                signal_opposite = True

            time_held = (datetime.now() - entry_time).total_seconds()
            rsi_exit = time_exit = trailing_stop_exit = False
            if time_held >= 3600:
                if (signal == "LONG" and df["rsi"].iloc[-1] < 30) or (signal == "SHORT" and df["rsi"].iloc[-1] > 70):
                    rsi_exit = True
                if time_held >= 7200 and (df["adx"].iloc[-1] < 25 or (signal == "SHORT" and df["rsi"].iloc[-1] > 70)):
                    time_exit = True
            if (signal == "LONG" and current_price <= trailing_stop_price) or (signal == "SHORT" and current_price >= trailing_stop_price):
                trailing_stop_exit = True

            should_exit = False
            exit_reason = ""
            if signal == "LONG":
                if current_price >= tp_price:
                    should_exit = True; exit_reason = "TP"
                elif current_price <= sl_price:
                    should_exit = True; exit_reason = "SL"
                elif trailing_stop_exit:
                    should_exit = True; exit_reason = "Trailing"
                elif signal_opposite:
                    should_exit = True; exit_reason = "Opposite"
                elif rsi_exit or time_exit:
                    should_exit = True; exit_reason = "Exit Condition"
            else:  # SHORT
                if current_price <= tp_price:
                    should_exit = True; exit_reason = "TP"
                elif current_price >= sl_price:
                    should_exit = True; exit_reason = "SL"
                elif trailing_stop_exit:
                    should_exit = True; exit_reason = "Trailing"
                elif signal_opposite:
                    should_exit = True; exit_reason = "Opposite"
                elif rsi_exit or time_exit:
                    should_exit = True; exit_reason = "Exit Condition"

            if hedge_position and should_exit and exit_reason == "TP":
                hedge_signal, hedge_entry_price, hedge_amount, hedge_leverage, _, hedge_sl_price, hedge_tp_price, _, hedge_amount_usd = hedge_position
                _, hedge_profit_usdt, _, hedge_exit_price, _ = calculate_pnl(hedge_entry_price, current_price, hedge_signal, hedge_leverage, fee_rate, slippage_rate, hedge_amount_usd)
                capital += hedge_profit_usdt
                total_loss += hedge_profit_usdt
                daily_loss += hedge_profit_usdt
                trade_history.append((datetime.now(), hedge_entry_price, f"{hedge_signal}_HEDGE", hedge_exit_price, hedge_profit_usdt, capital, hedge_leverage))
                send_telegram_message(f"Hedge closed at TP: {hedge_signal}, Entry: {hedge_entry_price:.2f}, Exit: {hedge_exit_price:.2f}, PnL: {hedge_profit_usdt:.2f}")
                hedge_position = None

            if should_exit or partial_profit_taken:
                if should_exit:
                    total_trades += 1
                    daily_trades += 1
                    current_hour = datetime.now().hour
                    hour_slot = f"{(current_hour // 6 * 6):02d}h-{((current_hour // 6 * 6 + 6) % 24):02d}h"
                    hourly_stats[hour_slot]["trades"] += 1
                    hourly_stats[hour_slot]["profit"] += profit_usdt
                    if ((signal == "LONG" and current_price >= tp_price) or (signal == "SHORT" and current_price <= tp_price)):
                        winning_trades += 1
                    winrate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                    capital += profit_usdt
                    daily_profit += profit_usdt
                    total_loss += profit_usdt
                    daily_loss += profit_usdt
                    trade_history.append((datetime.now(), entry_price, signal, current_price, profit_usdt, capital, leverage))
                    message = f"Exit {signal}: {exit_reason} at {current_price:.2f}, PnL: {profit_percent:.2f}% ({profit_usdt:.2f} USD), Capital: {capital:.2f}, Winrate: {winrate:.2f}%, Loss: {total_loss:.2f}"
                    send_telegram_message(message)
                    logging.info(message)
                    save_trade_history()
                    save_capital_history()
                    current_position = None
                    last_reentry_time = datetime.now()
                else:
                    current_position = (signal, entry_price, amount, leverage, trailing_stop_price, sl_price, tp_price, entry_time, amount_usd)

            if calculate_streaks()[1] >= 5:
                send_telegram_message(f"Loss streak: {calculate_streaks()[1]} trades")

        if not current_position:
            current_candle_time = df.index[-1]
            if last_candle_time != current_candle_time:
                last_candle_time = current_candle_time
                signal, confidence, predicted_class = generate_signal(df)
                timestamp = df.index[-1]
                date = timestamp.date()
                reason = "PASS"
                if signal == "NO_SIGNAL":
                    if df["adx"].iloc[-1] < 5:
                        reason = "Low ADX"
                    elif confidence <= base_confidence_threshold:
                        reason = "Low Confidence"
                signals_log.append((timestamp, signal, confidence, current_price, reason, predicted_class))
                if last_reentry_time and (datetime.now() - last_reentry_time).total_seconds() >= 3600:
                    reentry_signal, reentry_confidence, reentry_class = generate_signal(df)
                    if reentry_signal != "NO_SIGNAL" and reentry_confidence > 0.8 and reentry_class == predicted_class:
                        signal, confidence, predicted_class = reentry_signal, reentry_confidence, reentry_class
                        logging.info(f"Re-entry: {signal}, Confidence: {confidence}")
                if signal != "NO_SIGNAL":
                    leverage = determine_leverage(confidence, df["atr"].iloc[-1], current_price, df["adx"].iloc[-1])
                    adjusted_trade_percentage = trade_percentage * (confidence / 0.7)
                    adjusted_trade_percentage = max(0.05, min(adjusted_trade_percentage, 0.15))
                    amount_usd = capital * adjusted_trade_percentage
                    if amount_usd > capital or amount_usd <= 0:
                        signals_log[-1] = (timestamp, "NO_SIGNAL", confidence, current_price, "Capital Issue", predicted_class)
                        continue
                    if amount_usd > capital * 0.5:
                        logging.warning(f"Amount {amount_usd:.2f} > 50% capital {capital:.2f}. Skipping.")
                        continue
                    notional_value = amount_usd * leverage
                    amount = notional_value / current_price
                    adjusted_entry_price = current_price * (1 + slippage_rate if signal == "LONG" else 1 - slippage_rate)
                    latest_atr = df["atr"].iloc[-1]
                    sl_price, tp_price = calculate_sl_tp(adjusted_entry_price, signal, leverage, latest_atr, confidence)
                    initial_trailing_stop, _ = calculate_trailing_stop(adjusted_entry_price, current_price, latest_atr, signal, sl_price, 0, tp_price)
                    current_position = (signal, adjusted_entry_price, amount, leverage, initial_trailing_stop, sl_price, tp_price, datetime.now(), amount_usd)
                    message = f"Enter {signal} at {adjusted_entry_price:.2f}, Value: {notional_value:.2f} USD, Leverage: {leverage}x, Amount: {amount:.4f}, Confidence: {confidence:.2f}, SL: {sl_price:.2f}, TP: {tp_price:.2f}"
                    if signal == "SHORT" and confidence < 0.8 and not hedge_position:  # Tăng từ 0.75 lên 0.8
                        hedge_amount_usd = amount_usd * 0.75
                        hedge_leverage = leverage
                        hedge_sl_price = current_price - latest_atr
                        hedge_tp_price = current_price + 0.5 * latest_atr
                        hedge_position = ("LONG", current_price, hedge_amount_usd / current_price, hedge_leverage, None, hedge_sl_price, hedge_tp_price, datetime.now(), hedge_amount_usd)
                        message += f"\nHedge LONG at {current_price:.2f}, Amount: {hedge_amount_usd:.2f} USD, SL: {hedge_sl_price:.2f}, TP: {hedge_tp_price:.2f}"
                    send_telegram_message(message)
                    logging.info(message)
                    if signal == "SHORT":
                        short_signals_by_date.append(date)
                    else:
                        long_signals_by_date.append(date)
                    all_signals_by_date.append(date)
                    time.sleep(5)
                else:
                    logging.info(f"No signal, Price: {current_price:.4f}, Confidence: {confidence:.2f}")

        historical_df = df
        if time.time() % 14400 < 10:
            save_trade_history()
            save_capital_history()
            save_market_data()
            save_telegram_log()
            send_status_update()
            plot_trade_distribution()

        time.sleep(0.5)

if __name__ == "__main__":
    main()