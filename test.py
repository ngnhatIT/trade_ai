import requests
import pandas as pd
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, awesome_oscillator
from ta.volatility import AverageTrueRange
import logging
import ccxt
import csv
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from collections import Counter

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tham số giao dịch
exchange = ccxt.binance({'enableRateLimit': True})
symbol = "BTC/USDT"
interval = "5m"
base_timesteps = 20
lookback_hours = 24
initial_capital = 1000
capital = initial_capital
trade_percentage = 0.10  # Luôn sử dụng 10% vốn
fee_rate = 0.0002
slippage_rate = 0.0001
risk_per_trade = 0.02
confidence_threshold = 0.15
adx_threshold = 5
rsi_overbought = 80
rsi_oversold = 20
tp_ratio = 0.4  # Lãi 40% (sẽ chia cho leverage)
sl_ratio = 0.2  # Lỗ 20% (sẽ chia cho leverage)
base_confidence_threshold = 0.20

trade_history = []
capital_history = []
market_data_history = []
current_position = None
total_trades = 0
winning_trades = 0
daily_trades = 0
daily_profit = 0
last_day = None
is_trading_paused = False
last_drawdown_check = time.time()
hourly_stats = {f"{i:02d}h-{((i+6)%24):02d}h": {"trades": 0, "profit": 0} for i in range(0, 24, 6)}
max_drawdown_limit = 20
min_capital_threshold = initial_capital * 0.5
total_loss = 0  # Tổng lỗ tích lũy trong ngày
short_signals_by_date = []
long_signals_by_date = []
all_signals_by_date = []
signals_log = []
last_candle_time = None
last_loss_reset_time = time.time()

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

scaler = pd.read_pickle("scaler.pkl")
model = load_model("optimized_model.keras", custom_objects={"focal_loss_fixed": focal_loss(gamma=1.0, alpha=0.5)})
data_scaled_buffer = None

# Hàm lấy dữ liệu lịch sử từ /klines
def get_historical_data(symbol, interval, lookback_hours=24, max_retries=5):
    url = "https://api.binance.com/api/v3/klines"
    data = []
    end_time = int(time.time() * 1000)
    start_time = int((datetime.fromtimestamp(end_time / 1000) - timedelta(hours=lookback_hours)).timestamp() * 1000)
    limit = 1000

    retries = 0
    while start_time < end_time and retries < max_retries:
        try:
            params = {
                "symbol": symbol.replace("/", ""),
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
                "limit": limit
            }
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

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors='coerce')
    df = df.ffill().bfill()
    logging.info(f"Fetched {len(df)} historical data points")
    return df

# Hàm lấy giá thời gian thực
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

# Hàm lấy dữ liệu realtime
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

def determine_leverage(confidence):
    if confidence < 0.7:
        return 25  # Đòn bẩy nhỏ nhất
    elif confidence < 0.8:
        return 40
    elif confidence < 0.9:
        return 60
    else:
        return 75  # Đòn bẩy lớn nhất

def generate_signal(df):
    features = ["price_change", "rsi", "atr", "adx", "adx_pos", "adx_neg", "stoch", "momentum", "awesome", "hour", "dayofweek"]
    X = prepare_prediction_data(df, features)
    if X is None or len(X) == 0:
        logging.warning("No valid data for prediction.")
        return "NO_SIGNAL", 0.0, 0
    
    logging.info(f"Features values: {df[features].iloc[-1].to_dict()}")
    latest_adx = df["adx"].iloc[-1]
    latest_atr = df["atr"].iloc[-1]
    latest_price_change = df["price_change"].iloc[-1]
    avg_price_change = df["price_change"].iloc[-base_timesteps:].mean()
    short_momentum = df["momentum"].iloc[-5:].mean()
    
    adjusted_threshold = base_confidence_threshold - 0.1 if latest_adx < 20 else base_confidence_threshold
    
    predictions = model.predict(X, verbose=0)
    predicted_class = np.argmax(predictions[-1]) - 1
    confidence = np.max(predictions[-1])
    logging.info(f"Prediction: Class={predicted_class}, Confidence={confidence}, Adjusted Threshold={adjusted_threshold}")
    
    min_price_change_threshold = 0.00005
    
    if confidence < confidence_threshold:
        logging.info(f"No signal: Confidence {confidence} < {confidence_threshold}")
        return "NO_SIGNAL", confidence, predicted_class
    
    if (predicted_class == 1 and (df["rsi"].iloc[-1] > rsi_overbought or df["adx"].iloc[-1] < adx_threshold)):
        logging.info("No signal: LONG rejected due to RSI or ADX")
        return "NO_SIGNAL", confidence, predicted_class
    elif (predicted_class == -1 and (df["rsi"].iloc[-1] < rsi_oversold or df["adx"].iloc[-1] < adx_threshold)):
        logging.info("No signal: SHORT rejected due to RSI or ADX")
        return "NO_SIGNAL", confidence, predicted_class
    
    if (confidence > adjusted_threshold and 
        predicted_class != 0 and 
        latest_adx >= 5 and
        abs(latest_price_change) > min_price_change_threshold and
        ((predicted_class == 1 and avg_price_change >= -0.0001 and short_momentum >= -0.0001) or
         (predicted_class == -1 and avg_price_change <= 0.0001 and short_momentum <= 0.0001))):
        signal = "LONG" if predicted_class == 1 else "SHORT"
        logging.info(f"Signal generated: {signal}, Confidence: {confidence}")
        return signal, confidence, predicted_class
    logging.info(f"No signal: Failed final conditions, Min Price Change Threshold={min_price_change_threshold}, Latest Price Change={latest_price_change}")
    return "NO_SIGNAL", confidence, predicted_class

def calculate_pnl(entry_price, current_price, signal, leverage, fee_rate, slippage_rate, amount_usd=100):
    adjusted_entry_price = entry_price * (1 + slippage_rate if signal == "LONG" else 1 - slippage_rate)
    notional_value = amount_usd * leverage
    if signal == "LONG":
        price_diff = (current_price - adjusted_entry_price) / adjusted_entry_price
    else:  # SHORT
        price_diff = (adjusted_entry_price - current_price) / adjusted_entry_price
    profit_percent = price_diff * 100
    profit_before_fee = (profit_percent / 100) * notional_value  # Tính trên notional_value
    fee = (notional_value * fee_rate) * 2
    profit_usdt = profit_before_fee - fee
    adjusted_exit_price = current_price
    position_value = notional_value
    logging.debug(f"Debug PnL: entry={adjusted_entry_price:.2f}, exit={current_price:.2f}, diff={price_diff*100:.4f}%, leverage={leverage}, value={position_value:.2f}, fee={fee:.2f}, profit={profit_usdt:.2f}")
    return profit_percent, profit_usdt, adjusted_entry_price, adjusted_exit_price, position_value

def calculate_sl_tp(entry_price, signal, leverage, sl_ratio, tp_ratio, slippage_rate=0.0001):
    adjusted_entry_price = entry_price * (1 + slippage_rate if signal == "LONG" else 1 - slippage_rate)
    
    # Tính tỷ lệ thay đổi giá dựa trên đòn bẩy
    tp_price_change = tp_ratio / leverage
    sl_price_change = sl_ratio / leverage
    
    if signal == "LONG":
        tp_price = adjusted_entry_price * (1 + tp_price_change)
        sl_price = adjusted_entry_price * (1 - sl_price_change)
    else:  # SHORT
        tp_price = adjusted_entry_price * (1 - tp_price_change)
        sl_price = adjusted_entry_price * (1 + sl_price_change)
    
    logging.info(f"Calculated SL: {sl_price:.2f}, TP: {tp_price:.2f} for signal: {signal}, Entry: {entry_price:.2f}, Leverage: {leverage}x, TP Change: {tp_price_change*100:.2f}%, SL Change: {sl_price_change*100:.2f}%")
    return sl_price, tp_price

def calculate_trailing_stop(entry_price, current_price, latest_atr, signal, sl_price):
    atr_ratio = latest_atr / entry_price
    trailing_stop_factor = max(0.005, 2 * atr_ratio)
    
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
    plt.plot(dates, total_data, label="Total Trades (SHORT + LONG)", color="blue", linestyle="--")
    plt.title("Daily Trade Distribution (SHORT, LONG, and Total)")
    plt.xlabel("Date")
    plt.ylabel("Number of Trades")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("trade_distribution_realtime.png")
    plt.close()
    logging.info("Trade distribution chart saved as 'trade_distribution_realtime.png'")

TELEGRAM_BOT_TOKEN = "7617216154:AAF-5RxHYmn63pC2BgGJTAMRm2ehO4HcZvA"
TELEGRAM_CHAT_ID = "2028475238"

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            logging.info(f"Đã gửi thông báo Telegram: {message}")
        else:
            logging.error(f"Lỗi gửi Telegram: {response.text}")
    except Exception as e:
        logging.error(f"Lỗi khi gửi Telegram: {e}")
        logging.info(f"Telegram message sent: {message}")

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
    drawdown = (max_capital - min_capital) / max_capital * 100 if max_capital > 0 else 0
    return drawdown

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
    global daily_trades, daily_profit, last_day
    winrate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    drawdown = calculate_drawdown()
    win_streak, loss_streak = calculate_streaks()
    
    current_day = datetime.now().date()
    if last_day is None:
        last_day = current_day
    if current_day != last_day:
        message = f"Daily Report {last_day}:\nTotal Trades: {daily_trades}\nProfit: {daily_profit:.2f} USD\nTotal Loss: {total_loss:.2f} USD"
        send_telegram_message(message)
        daily_trades = 0
        daily_profit = 0
        last_day = current_day
    
    current_hour = datetime.now().hour
    hour_slot = f"{(current_hour // 6 * 6):02d}h-{((current_hour // 6 * 6 + 6) % 24):02d}h"
    
    short_counts = Counter(short_signals_by_date)
    long_counts = Counter(long_signals_by_date)
    all_counts = Counter(all_signals_by_date)
    if short_counts:
        max_short_date, max_short_count = short_counts.most_common(1)[0]
        short_info = f"Day with most SHORT signals: {max_short_date}, Number: {max_short_count}"
    else:
        short_info = "No SHORT signals generated."
    if long_counts:
        max_long_date, max_long_count = long_counts.most_common(1)[0]
        long_info = f"Day with most LONG signals: {max_long_date}, Number: {max_long_count}"
    else:
        long_info = "No LONG signals generated."
    if all_counts:
        min_trades_date, min_trades_count = all_counts.most_common()[-1]
        trades_info = f"Day with least trades: {min_trades_date}, Number: {min_trades_count}"
    else:
        trades_info = "No trades generated."
    
    message = (f"Status Update:\nCapital: {capital:.2f} USD\nTotal Trades: {total_trades}\nWinning Trades: {winning_trades}\n"
               f"Winrate: {winrate:.2f}%\nDrawdown: {drawdown:.2f}%\nLongest Win Streak: {win_streak}\nLongest Loss Streak: {loss_streak}\n"
               f"Hourly Stats ({hour_slot}): Trades: {hourly_stats[hour_slot]['trades']}, Profit: {hourly_stats[hour_slot]['profit']:.2f} USD\n"
               f"Total Loss: {total_loss:.2f} USD\n{short_info}\n{long_info}\n{trades_info}")
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
    global current_position, total_trades, winning_trades, capital, daily_trades, daily_profit, is_trading_paused, last_drawdown_check, total_loss, last_candle_time, data_scaled_buffer, last_loss_reset_time
    logging.info(f"Starting realtime trading system with initial capital: {initial_capital} USD")
    send_telegram_message(f"Starting realtime trading system with initial capital: {initial_capital} USD, make money!")

    if not check_api_status(exchange, symbol):
        logging.error("Cannot proceed due to API failure. Please check your API key or network connection.")
        return

    historical_df = get_historical_data(symbol, interval, lookback_hours=24)
    if historical_df is None or len(historical_df) < base_timesteps + 34:
        logging.error("Insufficient historical data. Starting with current price accumulation...")
        current_price = get_current_price(symbol)
        if current_price is None:
            logging.error("Cannot fetch current price. Exiting...")
            return
        historical_df = pd.DataFrame({
            "timestamp": [pd.Timestamp.now()],
            "open": [current_price],
            "high": [current_price],
            "low": [current_price],
            "close": [current_price],
            "volume": [0]
        })
        historical_df.set_index("timestamp", inplace=True)

    while len(historical_df) < base_timesteps + 34:
        new_df = get_realtime_data(symbol, interval, historical_df)
        if new_df is not None and len(new_df) > len(historical_df):
            historical_df = new_df
            logging.info(f"Accumulated {len(historical_df)} candles so far...")
        else:
            logging.warning("No new data fetched. Retrying...")
        time.sleep(1)

    historical_df = add_features_incremental(historical_df)
    while historical_df is None:
        logging.warning("Waiting for sufficient accumulated data...")
        time.sleep(60)
        historical_df = add_features_incremental(historical_df)
        if historical_df is not None:
            break

    last_candle_time = historical_df.index[-1]

    while True:
        current_time = time.time()
        if current_time - last_loss_reset_time >= 86400:
            logging.info(f"Resetting total_loss after 24 hours. Previous total_loss: {total_loss:.2f} USD")
            total_loss = 0
            last_loss_reset_time = current_time

        if time.time() - last_drawdown_check >= 3600:
            drawdown = calculate_drawdown()
            if drawdown > max_drawdown_limit or capital < min_capital_threshold:
                is_trading_paused = True
                send_telegram_message(f"Trading paused due to high drawdown: {drawdown:.2f}% or capital {capital:.2f} < {min_capital_threshold:.2f} USD")
                time.sleep(3600)
                is_trading_paused = False
                last_drawdown_check = time.time()

        max_loss_threshold = capital * 0.02
        current_trade_loss = 0
        if current_position is not None:
            signal, entry_price, amount, current_leverage, trailing_stop_price, sl_price, tp_price = current_position
            current_price = get_current_price(symbol, signal=signal)
            if current_price is None:
                current_price = historical_df["close"].iloc[-1]
            _, profit_usdt, _, _, _ = calculate_pnl(entry_price, current_price, signal, current_leverage, fee_rate, slippage_rate)
            current_trade_loss = -profit_usdt if profit_usdt < 0 else 0

        if total_loss < 0 and (abs(total_loss) + current_trade_loss) >= max_loss_threshold:
            logging.info(f"Total loss: {total_loss:.2f} USD, Current trade loss: {current_trade_loss:.2f} USD, Max loss threshold: {max_loss_threshold:.2f} USD")
            if current_position is not None:
                signal, entry_price, amount, current_leverage, trailing_stop_price, sl_price, tp_price = current_position
                current_price = get_current_price(symbol, signal=signal)
                if current_price is None:
                    current_price = historical_df["close"].iloc[-1]
                _, profit_usdt, _, adjusted_exit_price, _ = calculate_pnl(entry_price, current_price, signal, current_leverage, fee_rate, slippage_rate)
                capital += profit_usdt
                trade_history.append((datetime.now(), entry_price, signal, adjusted_exit_price, profit_usdt, capital, current_leverage))
                message = f"Force exit {signal} due to total loss exceeding 2% (Total: {total_loss:.2f} USD, Current: {current_trade_loss:.2f} USD), Capital: {capital:.2f} USD"
                send_telegram_message(message)
                save_trade_history()
                save_capital_history()
                current_position = None
            is_trading_paused = True
            send_telegram_message(f"Trading paused due to total loss exceeding 2% (Total: {total_loss:.2f} USD). Capital: {capital:.2f} USD")
            time.sleep(300)
            is_trading_paused = False
            continue

        if is_trading_paused:
            time.sleep(300)
            continue

        df = get_realtime_data(symbol, interval, historical_df)
        logging.info(f"Realtime data shape: {df.shape}")
        if df is None:
            time.sleep(300)
            continue
        df = add_features_incremental(df, historical_df)
        logging.info(f"Data with features shape: {df.shape}")
        if df is None:
            time.sleep(60)
            continue

        save_market_data()

        signal = current_position[0] if current_position else None
        current_price = get_current_price(symbol, signal=signal)
        if current_price is None:
            logging.warning("Using fallback price from last candle due to fetch failure.")
            current_price = df["close"].iloc[-1]

        if current_position is not None:
            signal, entry_price, amount, current_leverage, trailing_stop_price, sl_price, tp_price = current_position
            latest_atr = df["atr"].iloc[-1]
            latest_rsi = df["rsi"].iloc[-1]
            latest_adx = df["adx"].iloc[-1]
            profit_percent, profit_usdt, adjusted_entry_price, adjusted_exit_price, position_value = calculate_pnl(entry_price, current_price, signal, current_leverage, fee_rate, slippage_rate)
            logging.info(f"Current position: {signal}, Entry: {entry_price:.2f}, Current: {current_price:.2f}, PnL: {profit_percent:.2f}% ({profit_usdt:.2f} USD), Leverage: {current_leverage}x, Total Loss: {total_loss:.2f} USD, Position Value: {position_value:.2f} USD")

            new_trailing_stop, trailing_stop_factor = calculate_trailing_stop(entry_price, current_price, latest_atr, signal, sl_price)
            trailing_stop_price = new_trailing_stop
            logging.info(f"Trailing Stop Factor: {trailing_stop_factor:.4f}, Current Trailing Stop: {trailing_stop_price:.2f}")

            trend_favorable = False
            if signal == "LONG":
                trend_favorable = (latest_rsi > 50 or latest_adx > 20)
            else:
                trend_favorable = (latest_rsi < 50 or latest_adx > 20)

            should_exit = False
            if signal == "LONG":
                if current_price >= tp_price:
                    should_exit = True
                    logging.info("Exit LONG: Hit Take Profit")
                elif current_price <= sl_price:
                    should_exit = True
                    logging.info("Exit LONG: Hit Stop Loss")
                elif current_price <= trailing_stop_price and not trend_favorable:
                    should_exit = True
                    logging.info("Exit LONG: Hit Trailing Stop and trend not favorable")
            else:
                if current_price <= tp_price:
                    should_exit = True
                    logging.info("Exit SHORT: Hit Take Profit")
                elif current_price >= sl_price:
                    should_exit = True
                    logging.info("Exit SHORT: Hit Stop Loss")
                elif current_price >= trailing_stop_price and not trend_favorable:
                    should_exit = True
                    logging.info("Exit SHORT: Hit Trailing Stop and trend not favorable")

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
                trade_history.append((datetime.now(), entry_price, signal, adjusted_exit_price, profit_usdt, capital, current_leverage))
                message = f"Exit {signal} at {adjusted_exit_price:.2f}, PnL: {profit_percent:.2f}% ({profit_usdt:.2f} USD), Capital: {capital:.2f} USD, Winrate: {winrate:.2f}%, Leverage: {current_leverage}x, Total Loss: {total_loss:.2f} USD"
                send_telegram_message(message)
                save_trade_history()
                save_capital_history()
                current_position = None

            _, loss_streak = calculate_streaks()
            if loss_streak >= 5:
                send_telegram_message(f"Warning: Loss streak reached {loss_streak} trades!")

        if current_position is None:
            current_candle_time = df.index[-1]
            if last_candle_time != current_candle_time:
                last_candle_time = current_candle_time
                signal, confidence, predicted_class = generate_signal(df)
                timestamp = df.index[-1]
                date = timestamp.date()
                reason = "PASS"
                if signal == "NO_SIGNAL":
                    if df["adx"].iloc[-1] < 5:
                        reason = "ADX too low"
                    elif confidence <= base_confidence_threshold:
                        reason = "Confidence too low"
                signals_log.append((timestamp, signal, confidence, current_price, reason, predicted_class))
                if signal != "NO_SIGNAL":
                    if predicted_class == 1:
                        current_leverage = determine_leverage(confidence)
                    elif predicted_class == -1:
                        current_leverage = determine_leverage(confidence)
                    else:
                        current_leverage = determine_leverage(confidence)
                    amount_usd = capital * trade_percentage
                    if amount_usd > capital:
                        amount_usd = capital
                    if amount_usd <= 0:
                        signals_log[-1] = (timestamp, "NO_SIGNAL", confidence, current_price, "Insufficient capital", predicted_class)
                        continue
                    if amount_usd > capital * 0.5:
                        logging.warning(f"Amount to trade {amount_usd:.2f} USD exceeds 50% of capital {capital:.2f} USD. Skipping trade.")
                        continue
                    notional_value = amount_usd * current_leverage
                    amount = notional_value / current_price
                    adjusted_entry_price = current_price * (1 + slippage_rate if signal == "LONG" else 1 - slippage_rate)
                    position_value = notional_value
                    latest_atr = df["atr"].iloc[-1]
                    sl_price, tp_price = calculate_sl_tp(adjusted_entry_price, signal, current_leverage, sl_ratio, tp_ratio)
                    initial_trailing_stop, trailing_stop_factor = calculate_trailing_stop(adjusted_entry_price, current_price, latest_atr, signal, sl_price)
                    current_position = (signal, adjusted_entry_price, amount, current_leverage, initial_trailing_stop, sl_price, tp_price)
                    message = (f"Enter {signal} at {adjusted_entry_price:.2f}, Position Value: {position_value:.2f} USD, "
                               f"Leverage: {current_leverage}x, Amount: {amount:.4f} BTC, Confidence: {confidence:.2f}, "
                               f"Predicted Class: {predicted_class}, Capital: {capital:.2f} USD, Total Loss: {total_loss:.2f} USD, "
                               f"SL: {sl_price:.2f}, TP: {tp_price:.2f}")
                    send_telegram_message(message)
                    logging.info(message)
                    if signal == "SHORT":
                        short_signals_by_date.append(date)
                    else:
                        long_signals_by_date.append(date)
                    all_signals_by_date.append(date)
                    time.sleep(5)
                else:
                    logging.info(f"No signal generated, Price: {current_price:.4f}, Confidence: {confidence:.2f}, Predicted Class: {predicted_class}")

        historical_df = df
        if time.time() % 86400 < 10:
            save_trade_history()
            save_capital_history()
            save_market_data()
            send_status_update()
            plot_trade_distribution()

        time.sleep(0.5)

if __name__ == "__main__":
    main()