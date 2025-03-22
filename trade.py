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
import tweepy

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cấu hình Telegram
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

# Đặt tùy chọn để tránh cảnh báo downcasting
pd.set_option('future.no_silent_downcasting', True)

# Định nghĩa mô hình LSTM (giữ lại nhưng không dùng để dự đoán giá)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Class để quản lý trạng thái dữ liệu
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
        self.model = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lookback = 60
        self.X_buffer = []
        self.y_buffer = []
        self.predictions_log = []  # Lưu dự đoán để đánh giá
        self.learning_rate = 0.001  # Learning rate động
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        self.feature_weights = None  # Trọng số của các chỉ số
        self.sentiment_score = 0.0  # Điểm tâm lý thị trường

    def update_df(self, timeframe, new_data):
        if timeframe == '5m':
            self.df_5m = preprocess_data(new_data, self)  # Gọi preprocess_data ngay
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

        self.X_buffer = X
        self.y_buffer = y

    def log_prediction(self, prediction, actual_price):
        self.predictions_log.append({
            'timestamp': prediction['timestamp'],
            'predicted_signal': prediction['signal'],
            'actual_price': actual_price
        })
        if len(self.predictions_log) > 1000:
            self.predictions_log.pop(0)

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

# Hàm lấy dữ liệu từ Binance API (REST) bằng ccxt
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

# Lấy dữ liệu tâm lý thị trường từ X (tích hợp dữ liệu thực với kiểm soát rate limit)
BEARER_TOKEN = "your_bearer_token_here"
RATE_LIMIT_THRESHOLD = 10  # Ngưỡng an toàn: chỉ gửi request nếu còn ít nhất 10 request
MONTHLY_TWEET_LIMIT = 10000  # Giới hạn tweet đọc/tháng (gói Basic)
TWEET_READ_COUNT = 0  # Biến toàn cục để theo dõi số tweet đã đọc trong tháng
TWEET_READ_LOG_FILE = "tweet_read_log.csv"  # File để lưu số tweet đã đọc

# Khởi tạo client X API
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Biến để lưu trữ thông tin rate limit
rate_limit_info = {
    "limit": 900,  # Giá trị mặc định
    "remaining": 900,
    "reset": int(time.time()) + 900  # 15 phút từ hiện tại
}

def fetch_sentiment_data(data_manager, model, tokenizer, device):
    global TWEET_READ_COUNT, rate_limit_info
    
    try:
        # Kiểm tra số tweet đã đọc trong tháng
        if TWEET_READ_COUNT >= MONTHLY_TWEET_LIMIT:
            logging.warning("Reached monthly tweet read limit (10,000). Pausing sentiment data fetching.")
            data_manager.sentiment_score = 0.0
            return

        # Kiểm tra rate limit trước khi gửi request
        if rate_limit_info["remaining"] <= RATE_LIMIT_THRESHOLD:
            wait_time = max(0, rate_limit_info["reset"] - int(time.time()))
            if wait_time > 0:
                logging.info(f"Rate limit nearly reached. Waiting for {wait_time} seconds until reset.")
                time.sleep(wait_time)

        # Tìm kiếm tweet liên quan đến Bitcoin
        query = "Bitcoin OR BTC -is:retweet lang:en"
        max_retries = 3
        retry_delay = 1  # Bắt đầu với 1 giây

        for attempt in range(max_retries):
            try:
                response = client.search_recent_tweets(
                    query=query,
                    max_results=100,  # Tối đa 100 tweet mỗi request
                    tweet_fields=["created_at", "lang"]
                )

                # Cập nhật thông tin rate limit từ header
                rate_limit_info["limit"] = int(response.headers.get("x-rate-limit-limit", 900))
                rate_limit_info["remaining"] = int(response.headers.get("x-rate-limit-remaining", 0))
                rate_limit_info["reset"] = int(response.headers.get("x-rate-limit-reset", int(time.time()) + 900))

                tweets = [tweet.text for tweet in response.data] if response.data else []
                num_tweets = len(tweets)
                TWEET_READ_COUNT += num_tweets

                # Ghi log số tweet đã đọc
                with open(TWEET_READ_LOG_FILE, "a") as f:
                    f.write(f"{datetime.now()},{num_tweets}\n")

                if not tweets:
                    logging.warning("No tweets found for sentiment analysis.")
                    data_manager.sentiment_score = 0.0
                    return

                # Phân tích tâm lý với FinBERT
                inputs = tokenizer(tweets, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                sentiment_score = np.mean(probs[:, 2] - probs[:, 0])  # Positive - Negative
                data_manager.sentiment_score = sentiment_score
                logging.info(f"Updated sentiment score: {sentiment_score:.4f} (based on {num_tweets} tweets)")
                break  # Thoát khỏi vòng lặp retry nếu thành công

            except tweepy.errors.TooManyRequests as e:
                # Xử lý lỗi 429 (Too Many Requests) với exponential backoff
                wait_time = max(1, rate_limit_info["reset"] - int(time.time()))
                logging.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds (attempt {attempt+1}/{max_retries}).")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                if attempt == max_retries - 1:
                    logging.error("Max retries reached. Setting sentiment score to 0.")
                    data_manager.sentiment_score = 0.0

            except Exception as e:
                logging.error(f"Error fetching sentiment data: {e}")
                data_manager.sentiment_score = 0.0
                break

    except Exception as e:
        logging.error(f"Error in fetch_sentiment_data: {e}")
        data_manager.sentiment_score = 0.0

# Tối ưu hóa trọng số của các chỉ số bằng XGBoost
def optimize_feature_weights(data_manager):
    df = data_manager.df_5m
    if len(df) < 100:
        return None

    features = ['ema_5', 'ema_10', 'rsi', 'macd_diff', 'atr', 'bb_width', 'stoch_k', 'stoch_d', 'obv', 'adx', 'rvi', 'roc']
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing columns in df_5m for optimize_feature_weights: {missing_cols}")
        return None
    try:
        X = df[features].values
        y = df['close'].pct_change().shift(-1).fillna(0).values

        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        weights = model.feature_importances_
        feature_weights = dict(zip(features, weights))
        logging.info(f"Feature weights: {feature_weights}")
        return feature_weights
    except Exception as e:
        logging.error(f"Error optimizing feature weights: {e}")
        return None

# Tự triển khai RVI
def calculate_rvi(df, window=14):
    try:
        # Tính giá trị numerator và denominator
        numerator = (df['close'] - df['open']) + 2 * (df['close'].shift(1) - df['open'].shift(1)) + \
                    2 * (df['close'].shift(2) - df['open'].shift(2)) + (df['close'].shift(3) - df['open'].shift(3))
        denominator = (df['high'] - df['low']) + 2 * (df['high'].shift(1) - df['low'].shift(1)) + \
                      2 * (df['high'].shift(2) - df['low'].shift(2)) + (df['high'].shift(3) - df['low'].shift(3))
        
        # Tránh chia cho 0
        denominator = denominator.replace(0, np.nan)
        rvi = numerator / denominator
        
        # Tính trung bình động 4 kỳ
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
            
            # Sử dụng hàm tự triển khai để tính RVI
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                logging.error("Missing required columns for RVI calculation")
                df['rvi'] = 0
            else:
                df['rvi'] = calculate_rvi(df, window=14)

            df['roc'] = ta.momentum.ROCIndicator(df['close'], window=12).roc()
            df['price_change'] = df['close'].pct_change() * 100
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            # Đặt giá trị mặc định nếu có lỗi
            for col in ['ema_5', 'ema_10', 'rsi', 'macd_diff', 'atr', 'bb_upper', 'bb_lower', 'bb_width', 
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
    
    return df.ffill().fillna(0)

# Dự đoán với FinBERT và tính TP/SL
def predict_with_finbert(df, data_manager, model, tokenizer, device, prob_threshold=0.4, batch_size=16):
    if 'prob_positive' not in df.columns:
        df['prob_positive'] = np.nan
        df['prob_negative'] = np.nan
        df['prob_neutral'] = np.nan
    
    to_predict = df[df['prob_positive'].isna()]
    if to_predict.empty:
        if not hasattr(predict_with_finbert, 'no_data_count'):
            predict_with_finbert.no_data_count = 0
        predict_with_finbert.no_data_count += 1
        if predict_with_finbert.no_data_count % 5 == 0:
            logging.info(f"No new data to predict with FinBERT, Total no-data counts: {predict_with_finbert.no_data_count}")
    else:
        try:
            texts = to_predict['text'].tolist()
            all_probs = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    all_probs.append(probs.cpu().numpy())
            
            if all_probs:
                probs = np.concatenate(all_probs, axis=0)
                df.loc[df['prob_positive'].isna(), 'prob_negative'] = probs[:, 0]
                df.loc[df['prob_positive'].isna(), 'prob_neutral'] = probs[:, 1]
                df.loc[df['prob_positive'].isna(), 'prob_positive'] = probs[:, 2]
                logging.info(f"Predicted probabilities for {len(probs)} new candles")
        except Exception as e:
            logging.error(f"Error predicting with FinBERT: {e}")
    
    df['signal'] = None
    try:
        df.loc[(df['prob_positive'] > prob_threshold) & 
               (df['ema_5'] > df['ema_10']) & 
               (df['macd_diff'] > 0) & 
               (df['rsi'] > 30) & (df['rsi'] < 70) & 
               (df['stoch_k'] > df['stoch_d']) & 
               (df['adx'] > 25) & 
               (df['roc'] > 0), 'signal'] = 'LONG'
        df.loc[(df['prob_negative'] > prob_threshold) & 
               (df['ema_5'] < df['ema_10']) & 
               (df['macd_diff'] < 0) & 
               (df['rsi'] > 30) & (df['rsi'] < 70) & 
               (df['stoch_k'] < df['stoch_d']) & 
               (df['adx'] > 25) & 
               (df['roc'] < 0), 'signal'] = 'SHORT'
    except Exception as e:
        logging.error(f"Error generating signals: {e}")

    # Tính TP/SL dựa trên ATR
    df['tp'] = np.nan
    df['sl'] = np.nan
    try:
        latest_atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0
        current_price = data_manager.current_price
        if current_price is None:
            logging.warning("Current price is None, cannot calculate TP/SL")
            return df

        # Thiết lập TP/SL dựa trên tín hiệu
        for idx in df.index:
            signal = df.at[idx, 'signal']
            if signal == 'LONG':
                df.at[idx, 'tp'] = current_price + 2 * latest_atr  # TP = giá hiện tại + 2 * ATR
                df.at[idx, 'sl'] = current_price - 1 * latest_atr  # SL = giá hiện tại - 1 * ATR
            elif signal == 'SHORT':
                df.at[idx, 'tp'] = current_price - 2 * latest_atr  # TP = giá hiện tại - 2 * ATR
                df.at[idx, 'sl'] = current_price + 1 * latest_atr  # SL = giá hiện tại + 1 * ATR
    except Exception as e:
        logging.error(f"Error calculating TP/SL: {e}")
    
    return df

# Hàm kiểm tra xem thời gian hiện tại có nằm trong khung giờ nhạy cảm không
def is_sensitive_time():
    current_hour = datetime.utcnow().hour  # Lấy giờ hiện tại theo UTC
    sensitive_hours = [
        (0, 2),   # 00:00–02:00 UTC
        (8, 10),  # 08:00–10:00 UTC
        (14, 16), # 14:00–16:00 UTC
        (18, 20)  # 18:00–20:00 UTC
    ]
    for start_hour, end_hour in sensitive_hours:
        if start_hour <= current_hour < end_hour:
            return True
    return False

# Hàm kiểm tra xem đã lấy dữ liệu trong khung giờ hiện tại chưa
last_fetched_hour = None  # Biến toàn cục để theo dõi khung giờ đã lấy dữ liệu

def should_fetch_sentiment_data():
    global last_fetched_hour
    current_hour = datetime.utcnow().hour
    if not is_sensitive_time():
        return False
    if last_fetched_hour == current_hour:
        return False  # Đã lấy dữ liệu trong khung giờ này
    return True

# Hàm chính để xử lý và dự đoán xu hướng
def process_and_predict(exchange, symbol, data_manager, model, tokenizer, device, script_start_time):
    global last_fetched_hour
    
    update_interval = 30  # Cập nhật mỗi 30 giây
    ohlcv_update_interval = 300  # Cập nhật OHLCV mỗi 5 phút
    retrain_interval = 86400  # Retrain mỗi ngày
    feature_weight_update_interval = 3600  # Cập nhật trọng số mỗi giờ
    
    last_ohlcv_update = 0
    last_retrain = 0
    last_feature_weight_update = 0
    previous_signal = None  # Biến để theo dõi tín hiệu trước đó

    while True:
        start_time = time.time()
        
        # Cập nhật dữ liệu OHLCV
        if time.time() - last_ohlcv_update >= ohlcv_update_interval:
            try:
                ohlcv_5m = fetch_ohlcv_data(exchange, symbol, '5m', hours_back=24, data_manager=data_manager, limit=288)
                if not ohlcv_5m.empty:
                    data_manager.update_df('5m', ohlcv_5m)
                
                ohlcv_1h = fetch_ohlcv_data(exchange, symbol, '1h', hours_back=24*7, data_manager=data_manager, limit=168)
                if not ohlcv_1h.empty:
                    data_manager.update_df('1h', ohlcv_1h)
                
                ohlcv_4h = fetch_ohlcv_data(exchange, symbol, '4h', hours_back=24*30, data_manager=data_manager, limit=180)
                if not ohlcv_4h.empty:
                    data_manager.update_df('4h', ohlcv_4h)
                
                last_ohlcv_update = time.time()
            except Exception as e:
                logging.error(f"Error updating OHLCV data: {e}")

        # Cập nhật tâm lý thị trường trong các khung giờ nhạy cảm
        if should_fetch_sentiment_data():
            fetch_sentiment_data(data_manager, model, tokenizer, device)
            last_fetched_hour = datetime.utcnow().hour
            logging.info(f"Fetched sentiment data at hour {last_fetched_hour} UTC")

        # Cập nhật trọng số của các chỉ số
        if time.time() - last_feature_weight_update >= feature_weight_update_interval:
            data_manager.feature_weights = optimize_feature_weights(data_manager)
            last_feature_weight_update = time.time()

        # Retrain định kỳ (nếu cần)
        if time.time() - last_retrain >= retrain_interval:
            last_retrain = time.time()

        if data_manager.df_5m.empty:
            logging.warning("DataFrame 5m is empty, cannot process data")
            time.sleep(update_interval)
            continue
        
        # Lấy giá real-time
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            data_manager.update_current_price(current_price)
        except Exception as e:
            logging.error(f"Error fetching ticker: {e}")
            time.sleep(update_interval)
            continue
        
        # Tiền xử lý dữ liệu và dự đoán tín hiệu
        try:
            df_5m_processed = preprocess_data(data_manager.df_5m, data_manager, current_price=data_manager.current_price)
            df_1h_processed = preprocess_data(data_manager.df_1h, data_manager) if not data_manager.df_1h.empty else pd.DataFrame()
            df_4h_processed = preprocess_data(data_manager.df_4h, data_manager) if not data_manager.df_4h.empty else pd.DataFrame()
            
            df_5m_processed = predict_with_finbert(df_5m_processed, data_manager, model, tokenizer, device, prob_threshold=0.4, batch_size=16)
            
            if df_5m_processed.empty or 'timestamp' not in df_5m_processed.columns:
                logging.warning("Processed DataFrame 5m is empty or missing 'timestamp' column")
                time.sleep(update_interval)
                continue
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            time.sleep(update_interval)
            continue
        
        # Hiển thị tín hiệu và TP/SL khi tín hiệu thay đổi
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
            
            # Ghi log
            logging.info(log_message)
            
            # Hiển thị trên màn hình
            print("\n" + display_message)
            
            # Gửi thông báo qua Telegram
            send_telegram_message(telegram_message)
            
            # Ghi log dự đoán để đánh giá
            prediction = {
                'timestamp': datetime.now(),
                'signal': latest_signal
            }
            data_manager.log_prediction(prediction, None)
        
        # Cập nhật tín hiệu trước đó
        previous_signal = latest_signal
        
        # Đánh giá tín hiệu
        if len(data_manager.predictions_log) > 0:
            data_manager.evaluate_predictions()

        # Điều chỉnh thời gian chờ
        elapsed_time = time.time() - start_time
        sleep_time = max(0, update_interval - elapsed_time)
        time.sleep(sleep_time)

# Main execution
if __name__ == "__main__":
    # Cài đặt
    symbol = 'BTC/USDT'
    script_start_time = pd.to_datetime(datetime.now())
    
    # Khởi tạo DataManager
    data_manager = DataManager()
    
    # Khởi tạo ccxt exchange
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
    except Exception as e:
        logging.error(f"Error initializing Binance exchange: {e}")
        exit(1)
    
    # Phát hiện thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Tải FinBERT
    model_dir = Path("finbert_model")
    model_dir.mkdir(exist_ok=True)
    try:
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
    except Exception as e:
        logging.error(f"Error loading FinBERT model: {e}")
        exit(1)
    
    # Lấy dữ liệu ban đầu
    try:
        data_manager.df_5m = fetch_ohlcv_data(exchange, symbol, '5m', hours_back=24, data_manager=data_manager, limit=288)
        data_manager.df_1h = fetch_ohlcv_data(exchange, symbol, '1h', hours_back=24*7, data_manager=data_manager, limit=168)
        data_manager.df_4h = fetch_ohlcv_data(exchange, symbol, '4h', hours_back=24*30, data_manager=data_manager, limit=180)
        
        if data_manager.df_5m.empty:
            logging.error("Failed to load initial 5m data. Exiting.")
            exit(1)
    except Exception as e:
        logging.error(f"Error fetching initial data: {e}")
        exit(1)
    
    # Tiền xử lý dữ liệu ban đầu
    try:
        data_manager.df_5m = preprocess_data(data_manager.df_5m, data_manager)
        data_manager.df_1h = preprocess_data(data_manager.df_1h, data_manager) if not data_manager.df_1h.empty else pd.DataFrame()
        data_manager.df_4h = preprocess_data(data_manager.df_4h, data_manager) if not data_manager.df_4h.empty else pd.DataFrame()
        
        data_manager.df_5m = predict_with_finbert(data_manager.df_5m, data_manager, model, tokenizer, device, prob_threshold=0.4, batch_size=16)
    except Exception as e:
        logging.error(f"Error preprocessing initial data: {e}")
        exit(1)
    
    # Ghi log tín hiệu cũ ban đầu
    try:
        initial_long_signals = data_manager.df_5m[data_manager.df_5m['signal'] == 'LONG'].copy()
        initial_short_signals = data_manager.df_5m[data_manager.df_5m['signal'] == 'SHORT'].copy()
        for _, signal in initial_long_signals.iterrows():
            logging.info(f"Initial past LONG signal: Time: {signal['timestamp']}, Price: {signal['close']:.2f}")
        for _, signal in initial_short_signals.iterrows():
            logging.info(f"Initial past SHORT signal: Time: {signal['timestamp']}, Price: {signal['close']:.2f}")
    except Exception as e:
        logging.error(f"Error logging initial signals: {e}")
    
    # Bắt đầu xử lý và dự đoán xu hướng
    try:
        process_and_predict(exchange, symbol, data_manager, model, tokenizer, device, script_start_time)
    except Exception as e:
        logging.error(f"Error in main process: {e}")
        exit(1)