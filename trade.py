import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import ta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
import logging
import os
from pathlib import Path
import requests
import signal
import sys
from sklearn.model_selection import ParameterGrid
import gc
import psutil

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading.log"),
        logging.StreamHandler()
    ]
)

# Thông tin API
TELEGRAM_BOT_TOKEN = "7617216154:AAF-5RxHYmn63pC2BgGJTAMRm2ehO4HcZvA"
TELEGRAM_CHAT_ID = "2028475238"

# Xử lý tín hiệu dừng
def signal_handler(sig, frame):
    logging.info("Shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Hàm gửi thông báo Telegram
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

# Hàm kiểm tra tài nguyên hệ thống
def check_system_resources():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage_mb = mem_info.rss / 1024 / 1024
    total_mem_mb = psutil.virtual_memory().total / 1024 / 1024
    mem_percent = (mem_usage_mb / total_mem_mb) * 100
    cpu_percent = psutil.cpu_percent(interval=1)
    logging.info(f"System Resources - Memory: {mem_usage_mb:.2f}/{total_mem_mb:.2f} MB ({mem_percent:.2f}%), CPU: {cpu_percent:.2f}%")
    if mem_percent > 90:
        logging.warning("High memory usage detected! Consider reducing batch size or using a smaller model.")
    if cpu_percent > 90:
        logging.warning("High CPU usage detected! Performance may be impacted.")

# Class quản lý dữ liệu
class DataManager:
    def __init__(self):
        self.df_5m = pd.DataFrame()
        self.df_1h = pd.DataFrame()
        self.df_4h = pd.DataFrame()
        self.last_ema_5 = None
        self.last_ema_10 = None
        self.last_data_update = None
        self.current_price = None
        self.price_buffer = []
        self.scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lookback = 60
        self.X_buffer = []
        self.y_buffer = []
        self.predictions_log = []
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        self.feature_weights = None
        self.signal_thresholds = {
            'prob_threshold': 0.4,
            'rsi_lower': 30,
            'rsi_upper': 70,
            'adx_threshold': 25
        }

    def update_df(self, timeframe, new_data):
        if timeframe == '5m':
            self.df_5m = preprocess_data(new_data, self)
            if len(self.df_5m) >= self.lookback + 1:
                self.update_buffers()
        elif timeframe == '1h':
            self.df_1h = preprocess_data(new_data, self)
        elif timeframe == '4h':
            self.df_4h = preprocess_data(new_data, self)
        self.last_data_update = datetime.now()
        logging.info(f"Updated DataFrame for {timeframe}: {len(new_data)} candles")

    def update_current_price(self, price):
        self.price_buffer.append(price)
        if len(self.price_buffer) > 5:
            self.price_buffer.pop(0)
        self.current_price = np.mean(self.price_buffer)
        self.last_data_update = datetime.now()
        logging.info(f"Updated current price (smoothed): {self.current_price}")

    def update_buffers(self):
        features = ['close', 'ema_5', 'ema_10', 'rsi', 'macd_diff', 'atr', 'bb_width', 'stoch_k', 'stoch_d', 'obv', 'adx', 'rvi', 'roc']
        missing_cols = [col for col in features if col not in self.df_5m.columns]
        if missing_cols:
            logging.error(f"Missing columns in df_5m for update_buffers: {missing_cols}")
            return
        data = self.df_5m[features].values
        data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i])
            y.append(data[i, 0])

        self.X_buffer = X[-1000:]
        self.y_buffer = y[-1000:]

    def log_prediction(self, prediction, actual_price):
        self.predictions_log.append({
            'timestamp': prediction['timestamp'],
            'predicted_signal': prediction['signal'],
            'actual_price': actual_price
        })
        if len(self.predictions_log) > 1000:
            self.predictions_log = self.predictions_log[-1000:]

    def evaluate_predictions(self):
        if len(self.predictions_log) < 10:
            return None
        correct_predictions = 0
        for log in self.predictions_log:
            if log['actual_price'] is None:
                continue
            if log['predicted_signal'] == 'LONG' and log['actual_price'] > self.current_price:
                correct_predictions += 1
            elif log['predicted_signal'] == 'SHORT' and log['actual_price'] < self.current_price:
                correct_predictions += 1
        accuracy = correct_predictions / len(self.predictions_log)
        logging.info(f"Prediction Accuracy (Long/Short): {accuracy:.4f}")
        return accuracy

# Hàm tính metrics đánh giá
def calculate_metrics(df):
    returns = df['close'].pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 12)
    signals = df[df['signal'].notnull()]
    if len(signals) > 0:
        win_rate = len(signals[(signals['signal'] == 'LONG') & (signals['close'].shift(-1) > signals['close'])]) / len(signals[signals['signal'] == 'LONG'])
    else:
        win_rate = 0
    drawdown = (df['close'].cummax() - df['close']) / df['close'].cummax()
    max_drawdown = drawdown.max()
    logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}, Win Rate: {win_rate:.2f}, Max Drawdown: {max_drawdown:.2f}")
    return sharpe_ratio, win_rate, max_drawdown

# Hàm backtest chiến lược
def backtest_strategy(df, initial_balance=10000):
    balance = initial_balance
    position = None
    entry_price = 0
    for idx, row in df.iterrows():
        if row['signal'] == 'LONG' and position is None:
            position = 'LONG'
            entry_price = row['close']
        elif row['signal'] == 'SHORT' and position is None:
            position = 'SHORT'
            entry_price = row['close']
        elif position == 'LONG' and (row['close'] >= row['tp'] or row['close'] <= row['sl']):
            balance += (row['close'] - entry_price) * (balance / entry_price)
            position = None
        elif position == 'SHORT' and (row['close'] <= row['tp'] or row['close'] >= row['sl']):
            balance += (entry_price - row['close']) * (balance / entry_price)
            position = None
    return balance

# Hàm tối ưu ngưỡng tín hiệu
def optimize_signal_thresholds(df, param_grid):
    best_accuracy = 0
    best_params = None
    for params in ParameterGrid(param_grid):
        prob_threshold = params['prob_threshold']
        rsi_lower = params['rsi_lower']
        rsi_upper = params['rsi_upper']
        adx_threshold = params['adx_threshold']
        
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = None
        signals.loc[(df['prob_positive'].fillna(0) > prob_threshold) & 
                    (df['ema_5'] > df['ema_10']) & 
                    (df['macd_diff'] > 0) & 
                    (df['rsi'] > rsi_lower) & (df['rsi'] < rsi_upper) & 
                    (df['stoch_k'] > df['stoch_d']) & 
                    (df['adx'] > adx_threshold) & 
                    (df['roc'] > 0), 'signal'] = 'LONG'
        signals.loc[(df['prob_negative'].fillna(0) > prob_threshold) & 
                    (df['ema_5'] < df['ema_10']) & 
                    (df['macd_diff'] < 0) & 
                    (df['rsi'] > rsi_lower) & (df['rsi'] < rsi_upper) & 
                    (df['stoch_k'] < df['stoch_d']) & 
                    (df['adx'] > adx_threshold) & 
                    (df['roc'] < 0), 'signal'] = 'SHORT'
        
        df_with_signals = df.copy()
        df_with_signals['signal'] = signals['signal']
        final_balance = backtest_strategy(df_with_signals)
        accuracy = final_balance / 10000
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
    return best_params

# Hàm lấy dữ liệu từ Binance API
def fetch_ohlcv_data(exchange, symbol, timeframe, hours_back, data_manager, limit=288):
    cache_file = f"{symbol.replace('/', '_')}_{timeframe}_cache.csv"
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_csv(cache_file)
            cached_df['timestamp'] = pd.to_datetime(cached_df['timestamp'])
            if (datetime.now() - cached_df['timestamp'].iloc[-1]).total_seconds() < 300:
                logging.info(f"Loaded {timeframe} data from cache: {cache_file}")
                data_manager.last_data_update = datetime.now()
                return cached_df
        except Exception as e:
            logging.warning(f"Error loading cache for {timeframe}: {e}, fetching new data")

    try:
        since = int((datetime.now() - pd.Timedelta(hours=hours_back)).timestamp() * 1000)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            logging.warning(f"No {timeframe} data received from Binance API")
            return pd.DataFrame()
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            logging.error(f"Missing required columns in {timeframe} data")
            return pd.DataFrame()
        if df[required_columns].isnull().any().any():
            logging.warning(f"{timeframe} data contains NaN values, dropping invalid rows")
            df = df.dropna(subset=required_columns)
        
        df.to_csv(cache_file, index=False)
        logging.info(f"Saved {timeframe} data to cache: {cache_file}")
        
        data_manager.last_data_update = datetime.now()
        logging.info(f"{timeframe} data loaded: {len(df)} candles")
        return df
    except Exception as e:
        logging.error(f"Error fetching {timeframe} data: {e}")
        return pd.DataFrame()

# Tự triển khai RVI
def calculate_rvi(df, window=14):
    try:
        numerator = (df['close'] - df['open']) + 2 * (df['close'].shift(1) - df['open'].shift(1)) + \
                    2 * (df['close'].shift(2) - df['open'].shift(2)) + (df['close'].shift(3) - df['open'].shift(3))
        denominator = (df['high'] - df['low']) + 2 * (df['high'].shift(1) - df['low'].shift(1)) + \
                      2 * (df['high'].shift(2) - df['low'].shift(2)) + (df['high'].shift(3) - df['low'].shift(3))
        
        denominator = denominator.replace(0, np.nan)
        rvi = numerator / denominator
        rvi = rvi.rolling(window=4).mean()
        return rvi.fillna(0)
    except Exception as e:
        logging.error(f"Error calculating RVI: {e}")
        return pd.Series(0, index=df.index)

# Thêm các chỉ báo tối ưu
def preprocess_data(df, data_manager, current_price=None):
    df = df.copy()
    
    if not df.empty:
        if current_price is not None:
            latest_row = df.iloc[-1].copy()
            latest_row['close'] = current_price
            latest_row['timestamp'] = datetime.now()
            df = pd.concat([df, pd.DataFrame([latest_row])], ignore_index=True)
        
        if data_manager.last_ema_5 is not None and data_manager.last_ema_10 is not None:
            last_row = pd.DataFrame({
                'close': [df['close'].iloc[-2] if len(df) > 1 else df['close'].iloc[-1]],
                'ema_5': [data_manager.last_ema_5],
                'ema_10': [data_manager.last_ema_10]
            })
            new_data = df.tail(1)
            df_to_compute = pd.concat([last_row, new_data], ignore_index=True)
        else:
            df_to_compute = df

        try:
            df['ema_5'] = ta.trend.EMAIndicator(df_to_compute['close'], window=5).ema_indicator()
            df['ema_10'] = ta.trend.EMAIndicator(df_to_compute['close'], window=10).ema_indicator()
            data_manager.last_ema_5 = df['ema_5'].iloc[-1]
            data_manager.last_ema_10 = df['ema_10'].iloc[-1]

            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['macd_diff'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9).macd_diff()
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
            df['rvi'] = calculate_rvi(df, window=14)
            df['roc'] = ta.momentum.ROCIndicator(df['close'], window=12).roc()
            df['price_change'] = df['close'].pct_change() * 100
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            for col in ['ema_5', 'ema_10', 'rsi', 'macd_diff', 'atr', 'bb_width', 
                        'stoch_k', 'stoch_d', 'obv', 'adx', 'rvi', 'roc', 'price_change']:
                if col not in df:
                    df[col] = 0
    
    df['text'] = (
        np.where(df['price_change'] > 0, "Price increased by ", "Price decreased by ") + 
        np.abs(df['price_change']).astype(str) + "%, " +
        "RSI at " + df['rsi'].astype(str) + ", " +
        "MACD diff " + df['macd_diff'].astype(str) + ", " +
        np.where(df['ema_5'] > df['ema_10'], "EMA_5 above EMA_10", "EMA_5 below EMA_10") + ", " +
        "BB width " + df['bb_width'].astype(str) + ", " +
        "Stoch K " + df['stoch_k'].astype(str) + ", " +
        "OBV " + df['obv'].astype(str) + ", " +
        "ADX " + df['adx'].astype(str) + ", " +
        "RVI " + df['rvi'].astype(str) + ", " +
        "ROC " + df['roc'].astype(str)
    )
    
    # Đảm bảo các cột xác suất tồn tại ngay từ đầu
    for col in ['prob_positive', 'prob_negative', 'prob_neutral']:
        if col not in df.columns:
            df[col] = 0.0
    
    return df.ffill().fillna(0)

# Dự đoán với FinBERT và tính TP/SL
def predict_with_finbert(df, data_manager, model, tokenizer, device, batch_size=2, max_rows_to_predict=50):
    if df.empty:
        logging.warning("DataFrame is empty in predict_with_finbert, returning unchanged")
        return df

    # Đảm bảo các cột xác suất tồn tại và khởi tạo với giá trị 0
    for col in ['prob_positive', 'prob_negative', 'prob_neutral']:
        if col not in df.columns:
            df[col] = 0.0
            logging.info(f"Initialized {col} column with 0")
        else:
            df[col] = df[col].fillna(0)

    # Kiểm tra trước khi dự đoán
    if df['text'].isnull().any():
        logging.warning("Some text entries are null, filling with empty string")
        df['text'] = df['text'].fillna("")

    # Chỉ dự đoán cho các hàng chưa có xác suất (prob_positive == 0)
    to_predict = df[df['prob_positive'] == 0]
    if len(to_predict) > max_rows_to_predict:
        logging.info(f"Too many rows to predict ({len(to_predict)}), limiting to {max_rows_to_predict}")
        to_predict = to_predict.tail(max_rows_to_predict)
    logging.info(f"Rows to predict: {len(to_predict)} out of {len(df)}")

    if to_predict.empty:
        if not hasattr(predict_with_finbert, 'no_data_count'):
            predict_with_finbert.no_data_count = 0
        predict_with_finbert.no_data_count += 1
        if predict_with_finbert.no_data_count % 5 == 0:
            logging.info(f"No new data to predict with FinBERT, Total no-data counts: {predict_with_finbert.no_data_count}")
    else:
        try:
            check_system_resources()  # Kiểm tra tài nguyên trước khi dự đoán
            texts = to_predict['text'].tolist()
            if not texts:
                logging.warning("No text data available for FinBERT prediction")
                df['prob_positive'] = df['prob_positive'].fillna(0)
                df['prob_negative'] = df['prob_negative'].fillna(0)
                df['prob_neutral'] = df['prob_neutral'].fillna(0)
            else:
                all_probs = []
                total_batches = (len(texts) + batch_size - 1) // batch_size
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
                    try:
                        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            probs = torch.softmax(outputs.logits, dim=-1)
                            all_probs.append(probs.cpu().numpy())
                        del inputs, outputs, probs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    except Exception as e:
                        logging.error(f"Error in FinBERT prediction batch {batch_num}: {e}")
                        # Fallback: Điền giá trị mặc định cho batch này
                        batch_indices = to_predict.index[i:i + batch_size]
                        df.loc[batch_indices, 'prob_positive'] = 0
                        df.loc[batch_indices, 'prob_negative'] = 0
                        df.loc[batch_indices, 'prob_neutral'] = 0
                        continue
                
                if all_probs:
                    probs = np.concatenate(all_probs, axis=0)
                    predicted_indices = to_predict.index[:len(probs)]
                    df.loc[predicted_indices, 'prob_negative'] = probs[:, 0]
                    df.loc[predicted_indices, 'prob_neutral'] = probs[:, 1]
                    df.loc[predicted_indices, 'prob_positive'] = probs[:, 2]
                    logging.info(f"Predicted probabilities for {len(probs)} new candles")
                else:
                    logging.warning("No probabilities predicted by FinBERT, filling with 0")
                    df['prob_positive'] = df['prob_positive'].fillna(0)
                    df['prob_negative'] = df['prob_negative'].fillna(0)
                    df['prob_neutral'] = df['prob_neutral'].fillna(0)
        except Exception as e:
            logging.error(f"Error predicting with FinBERT: {e}")
            df['prob_positive'] = df['prob_positive'].fillna(0)
            df['prob_negative'] = df['prob_negative'].fillna(0)
            df['prob_neutral'] = df['prob_neutral'].fillna(0)

    # Kiểm tra lại cột prob_positive
    if df['prob_positive'].isnull().any():
        logging.warning("Found NaN values in prob_positive after prediction, filling with 0")
        df['prob_positive'] = df['prob_positive'].fillna(0)
    if df['prob_negative'].isnull().any():
        df['prob_negative'] = df['prob_negative'].fillna(0)
    if df['prob_neutral'].isnull().any():
        df['prob_neutral'] = df['prob_neutral'].fillna(0)
    logging.info(f"prob_positive stats: min={df['prob_positive'].min()}, max={df['prob_positive'].max()}, mean={df['prob_positive'].mean()}")

    # Khởi tạo cột tín hiệu
    df['signal'] = None
    try:
        prob_threshold = data_manager.signal_thresholds['prob_threshold']
        rsi_lower = data_manager.signal_thresholds['rsi_lower']
        rsi_upper = data_manager.signal_thresholds['rsi_upper']
        adx_threshold = data_manager.signal_thresholds['adx_threshold']
        
        trend_1h = data_manager.df_1h['ema_5'].iloc[-1] > data_manager.df_1h['ema_10'].iloc[-1] if not data_manager.df_1h.empty else True
        trend_4h = data_manager.df_4h['ema_5'].iloc[-1] > data_manager.df_4h['ema_10'].iloc[-1] if not data_manager.df_4h.empty else True
        
        df.loc[(df['prob_positive'] > prob_threshold) & 
               (df['ema_5'] > df['ema_10']) & 
               (df['macd_diff'] > 0) & 
               (df['rsi'] > rsi_lower) & (df['rsi'] < rsi_upper) & 
               (df['stoch_k'] > df['stoch_d']) & 
               (df['adx'] > adx_threshold) & 
               (df['roc'] > 0) & 
               (trend_1h) & 
               (trend_4h), 'signal'] = 'LONG'
        
        df.loc[(df['prob_negative'] > prob_threshold) & 
               (df['ema_5'] < df['ema_10']) & 
               (df['macd_diff'] < 0) & 
               (df['rsi'] > rsi_lower) & (df['rsi'] < rsi_upper) & 
               (df['stoch_k'] < df['stoch_d']) & 
               (df['adx'] > adx_threshold) & 
               (df['roc'] < 0) & 
               (~trend_1h) & 
               (~trend_4h), 'signal'] = 'SHORT'
    except Exception as e:
        logging.error(f"Error generating signals: {e}")
        df['signal'] = None

    # Tính TP/SL dựa trên ATR
    df['tp'] = np.nan
    df['sl'] = np.nan
    try:
        latest_atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0
        current_price = data_manager.current_price
        if current_price is None:
            current_price = df['close'].iloc[-1] if not df.empty and 'close' in df.columns else 0
            logging.warning(f"Current price is None, using last close price: {current_price}")

        for idx in df.index:
            signal = df.at[idx, 'signal']
            if signal == 'LONG':
                df.at[idx, 'tp'] = current_price + 2 * latest_atr
                df.at[idx, 'sl'] = current_price - 1 * latest_atr
            elif signal == 'SHORT':
                df.at[idx, 'tp'] = current_price - 2 * latest_atr
                df.at[idx, 'sl'] = current_price + 1 * latest_atr
    except Exception as e:
        logging.error(f"Error calculating TP/SL: {e}")
    
    return df

# Hàm tối ưu trọng số đặc trưng
def optimize_feature_weights(data_manager):
    try:
        if len(data_manager.X_buffer) < 100:
            logging.warning("Not enough data to optimize feature weights")
            return None
        
        X = np.array(data_manager.X_buffer)
        y = np.array(data_manager.y_buffer)
        
        if X.shape[0] != y.shape[0]:
            logging.error(f"Mismatch in X and y shapes: {X.shape[0]} vs {y.shape[0]}")
            return None
        
        X_2d = X.reshape(X.shape[0], -1)
        
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_2d, y)
        
        weights = model.feature_importances_
        weights = weights.reshape(X.shape[1], X.shape[2])
        weights = np.mean(weights, axis=0)
        
        feature_names = ['close', 'ema_5', 'ema_10', 'rsi', 'macd_diff', 'atr', 'bb_width', 
                         'stoch_k', 'stoch_d', 'obv', 'adx', 'rvi', 'roc']
        feature_weights = dict(zip(feature_names, weights))
        
        logging.info(f"Updated feature weights: {feature_weights}")
        return feature_weights
    except Exception as e:
        logging.error(f"Error optimizing feature weights: {e}")
        return None

# Hàm chính để xử lý và dự đoán xu hướng
def process_and_predict(exchange, symbol, data_manager, model, tokenizer, device, script_start_time):
    update_interval = 30
    ohlcv_update_interval = 300
    retrain_interval = 86400
    feature_weight_update_interval = 3600
    prediction_interval = 300
    threshold_update_interval = 86400
    
    last_ohlcv_update = 0
    last_retrain = 0
    last_feature_weight_update = 0
    last_prediction_time = 0
    last_threshold_update = 0
    previous_signal = None

    while True:
        start_time = time.time()
        
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            data_manager.update_current_price(current_price)
        except Exception as e:
            logging.error(f"Error fetching ticker: {e}")
            if not data_manager.df_5m.empty and 'close' in data_manager.df_5m.columns:
                data_manager.update_current_price(data_manager.df_5m['close'].iloc[-1])
                logging.warning(f"Using last close price as fallback: {data_manager.current_price}")
            else:
                time.sleep(update_interval)
                continue
        
        data_updated = False
        if time.time() - last_ohlcv_update >= ohlcv_update_interval:
            try:
                ohlcv_5m = fetch_ohlcv_data(exchange, symbol, '5m', hours_back=24, data_manager=data_manager, limit=288)
                if not ohlcv_5m.empty:
                    data_manager.update_df('5m', ohlcv_5m)
                    data_updated = True
                
                ohlcv_1h = fetch_ohlcv_data(exchange, symbol, '1h', hours_back=24*7, data_manager=data_manager, limit=168)
                if not ohlcv_1h.empty:
                    data_manager.update_df('1h', ohlcv_1h)
                
                ohlcv_4h = fetch_ohlcv_data(exchange, symbol, '4h', hours_back=24*30, data_manager=data_manager, limit=180)
                if not ohlcv_4h.empty:
                    data_manager.update_df('4h', ohlcv_4h)
                
                last_ohlcv_update = time.time()
            except Exception as e:
                logging.error(f"Error updating OHLCV data: {e}")

        if time.time() - last_feature_weight_update >= feature_weight_update_interval:
            data_manager.feature_weights = optimize_feature_weights(data_manager)
            last_feature_weight_update = time.time()

        if time.time() - last_threshold_update >= threshold_update_interval:
            param_grid = {
                'prob_threshold': [0.3, 0.4, 0.5],
                'rsi_lower': [20, 30, 40],
                'rsi_upper': [60, 70, 80],
                'adx_threshold': [20, 25, 30]
            }
            best_params = optimize_signal_thresholds(data_manager.df_5m, param_grid)
            if best_params:
                data_manager.signal_thresholds = best_params
                logging.info(f"Updated signal thresholds: {best_params}")
            last_threshold_update = time.time()

        if time.time() - last_retrain >= retrain_interval:
            last_retrain = time.time()

        if data_manager.df_5m.empty:
            logging.warning("DataFrame 5m is empty, cannot process data")
            time.sleep(update_interval)
            continue
        
        if data_updated or (time.time() - last_prediction_time >= prediction_interval):
            try:
                df_5m_processed = preprocess_data(data_manager.df_5m, data_manager, current_price=data_manager.current_price)
                df_1h_processed = preprocess_data(data_manager.df_1h, data_manager) if not data_manager.df_1h.empty else pd.DataFrame()
                df_4h_processed = preprocess_data(data_manager.df_4h, data_manager) if not data_manager.df_4h.empty else pd.DataFrame()
                
                df_5m_processed = predict_with_finbert(df_5m_processed, data_manager, model, tokenizer, device, batch_size=2)
                
                if df_5m_processed.empty or 'timestamp' not in df_5m_processed.columns:
                    logging.warning("Processed DataFrame 5m is empty or missing 'timestamp' column")
                    time.sleep(update_interval)
                    continue
                
                data_manager.df_5m = df_5m_processed  # Cập nhật DataFrame chính
                last_prediction_time = time.time()
            except Exception as e:
                logging.error(f"Error preprocessing data: {e}")
                time.sleep(update_interval)
                continue
        
        latest_signal = df_5m_processed['signal'].iloc[-1] if not df_5m_processed.empty and 'signal' in df_5m_processed.columns else "N/A"
        latest_tp = df_5m_processed['tp'].iloc[-1] if not df_5m_processed.empty and 'tp' in df_5m_processed.columns else None
        latest_sl = df_5m_processed['sl'].iloc[-1] if not df_5m_processed.empty and 'sl' in df_5m_processed.columns else None
        
        if latest_signal in ['LONG', 'SHORT'] and latest_signal != previous_signal:
            log_message = f"New Trading Signal: {latest_signal}, TP: {latest_tp:.2f}, SL: {latest_sl:.2f}"
            display_message = (
                f"New Trading Signal:\n"
                f"-------------------------------------------------------\n"
                f"Signal: {latest_signal}\n"
                f"Current Price: {data_manager.current_price:.2f}\n"
                f"Take Profit (TP): {latest_tp:.2f}\n"
                f"Stop Loss (SL): {latest_sl:.2f}\n"
                f"-------------------------------------------------------"
            )
            telegram_message = (
                f"📈 *New Trading Signal*\n"
                f"Signal: {latest_signal}\n"
                f"Current Price: {data_manager.current_price:.2f}\n"
                f"Take Profit (TP): {latest_tp:.2f}\n"
                f"Stop Loss (SL): {latest_sl:.2f}"
            )
            
            logging.info(log_message)
            print("\n" + display_message)
            send_telegram_message(telegram_message)
            
            prediction = {
                'timestamp': datetime.now(),
                'signal': latest_signal
            }
            data_manager.log_prediction(prediction, None)
        
        previous_signal = latest_signal
        
        if len(data_manager.predictions_log) > 0:
            data_manager.evaluate_predictions()

        if time.time() - last_retrain >= retrain_interval:
            final_balance = backtest_strategy(data_manager.df_5m)
            logging.info(f"Backtest result: Final balance = {final_balance:.2f}")
            calculate_metrics(data_manager.df_5m)

        elapsed_time = time.time() - start_time
        sleep_time = max(0, update_interval - elapsed_time)
        time.sleep(sleep_time)

if __name__ == "__main__":
    try:
        symbol = 'BTC/USDT'
        script_start_time = pd.to_datetime(datetime.now())
        
        data_manager = DataManager()
        
        exchange = ccxt.binance({'enableRateLimit': True})
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        
        model_dir = Path("finbert_model")
        model_dir.mkdir(exist_ok=True)
        if (model_dir / "pytorch_model.bin").exists():
            logging.info("Loading FinBERT model from local directory")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        else:
            logging.info("Downloading FinBERT model from Hugging Face")
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)
            logging.info(f"FinBERT model saved to {model_dir}")
        model.to(device)
        model.eval()
        
        # Giảm số lượng dữ liệu ban đầu để kiểm tra
        data_manager.df_5m = fetch_ohlcv_data(exchange, symbol, '5m', hours_back=24, data_manager=data_manager, limit=50)  # Giảm từ 288 xuống 50
        data_manager.df_1h = fetch_ohlcv_data(exchange, symbol, '1h', hours_back=24*7, data_manager=data_manager, limit=168)
        data_manager.df_4h = fetch_ohlcv_data(exchange, symbol, '4h', hours_back=24*30, data_manager=data_manager, limit=180)
        
        if data_manager.df_5m.empty:
            logging.error("Failed to load initial 5m data. Exiting.")
            exit(1)
        
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        data_manager.update_current_price(current_price)
        
        data_manager.df_5m = preprocess_data(data_manager.df_5m, data_manager)
        data_manager.df_1h = preprocess_data(data_manager.df_1h, data_manager) if not data_manager.df_1h.empty else pd.DataFrame()
        data_manager.df_4h = preprocess_data(data_manager.df_4h, data_manager) if not data_manager.df_4h.empty else pd.DataFrame()
        
        data_manager.df_5m = predict_with_finbert(data_manager.df_5m, data_manager, model, tokenizer, device, batch_size=2)
        
        initial_long_signals = data_manager.df_5m[data_manager.df_5m['signal'] == 'LONG'].copy()
        initial_short_signals = data_manager.df_5m[data_manager.df_5m['signal'] == 'SHORT'].copy()
        for _, signal in initial_long_signals.iterrows():
            logging.info(f"Initial past LONG signal: Time: {signal['timestamp']}, Price: {signal['close']:.2f}")
        for _, signal in initial_short_signals.iterrows():
            logging.info(f"Initial past SHORT signal: Time: {signal['timestamp']}, Price: {signal['close']:.2f}")
        
        process_and_predict(exchange, symbol, data_manager, model, tokenizer, device, script_start_time)
    except Exception as e:
        logging.error(f"Unexpected error in main execution: {e}")
        exit(1)