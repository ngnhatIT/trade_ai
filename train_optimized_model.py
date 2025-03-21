import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, MultiHeadAttention, GlobalAveragePooling1D, Add
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
import logging
import optuna
from optuna.samplers import TPESampler
from ta.trend import ADXIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, awesome_oscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import volume_weighted_average_price, OnBalanceVolumeIndicator, MFIIndicator

# In phiên bản TensorFlow
print(f"TensorFlow Version: {tf.__version__}")

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tf.config.experimental.set_visible_devices([], 'GPU')  # Chạy trên CPU

# Định nghĩa Focal Loss
def focal_loss(gamma=5.0, alpha=[0.7, 0.1, 0.7]):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int64)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true_one_hot = tf.one_hot(y_true, depth=3)
        ce_loss = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        focal_weight = tf.pow(1 - p_t, gamma)
        alpha_weight = tf.reduce_sum(y_true_one_hot * tf.constant(alpha, dtype=tf.float32), axis=-1)
        return tf.reduce_mean(alpha_weight * focal_weight * ce_loss)
    return focal_loss_fn

# 1. Load dữ liệu
def load_data(file_path="binance_BTCUSDT_1h.csv"):
    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        logging.info(f"Kích thước dữ liệu gốc: {df.shape}")
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors='coerce')
        df = df.ffill().bfill()
        logging.info(f"Kích thước sau khi xử lý: {df.shape}")
        if len(df) < 5000:
            logging.warning(f"Dữ liệu chỉ có {len(df)} mẫu, có thể không tối ưu.")
        return df
    except Exception as e:
        logging.error(f"Lỗi khi tải dữ liệu: {str(e)}")
        raise

# 2. Thêm đặc trưng
def add_features(df):
    logging.info("Bắt đầu thêm đặc trưng...")
    df = df.copy()
    
    if len(df) < 200:
        logging.warning("Dữ liệu <200 bản ghi, EMA_200 có thể không chính xác.")
    
    df["price_change"] = df["close"].pct_change().shift(1).fillna(0)
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi().shift(1).fillna(50)
    df["adx"] = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14).adx().shift(1).fillna(25)
    df["adx_pos"] = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14).adx_pos().shift(1).fillna(0)
    df["adx_neg"] = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14).adx_neg().shift(1).fillna(0)
    df["stoch"] = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3).stoch().shift(1).fillna(50)
    df["momentum"] = df["close"].diff(5).shift(1).fillna(0)
    df["awesome"] = awesome_oscillator(high=df["high"], low=df["low"], window1=5, window2=34).shift(1).fillna(0)
    df["macd"] = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9).macd_diff().shift(1).fillna(0)
    df["bb_upper"] = BollingerBands(close=df["close"], window=20, window_dev=2).bollinger_hband().shift(1).fillna(0)
    df["bb_lower"] = BollingerBands(close=df["close"], window=20, window_dev=2).bollinger_lband().shift(1).fillna(0)
    df["vwap"] = volume_weighted_average_price(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14).shift(1).fillna(0)
    df["ema_10"] = EMAIndicator(close=df["close"], window=10).ema_indicator().shift(1).fillna(0)
    df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator().shift(1).fillna(0)
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator().shift(1).fillna(0)
    df["ema_200"] = EMAIndicator(close=df["close"], window=200).ema_indicator().shift(1).fillna(0)
    df["obv"] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume().shift(1).fillna(0)
    df["roc"] = ROCIndicator(close=df["close"], window=14).roc().shift(1).fillna(0)
    df["mfi"] = MFIIndicator(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14).money_flow_index().shift(1).fillna(50)
    df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=5).average_true_range().shift(1).fillna(0)
    
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    df["vol_breakout"] = ((df["high"] - df["low"]) / df["high"].shift(1)).shift(1).fillna(0)
    df["vol_delta"] = df["obv"].diff().shift(1).fillna(0)
    df["rolling_mean_5"] = df["close"].rolling(window=5).mean().shift(1).fillna(0)
    df["rolling_std_5"] = df["close"].rolling(window=5).std().shift(1).fillna(0)
    df["lag_1"] = df["close"].shift(1).fillna(0)
    
    df["rsi_macd_interaction"] = (df["rsi"] * df["macd"]).shift(1).fillna(0)
    
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    logging.info(f"Kích thước sau khi thêm đặc trưng: {df.shape}")
    return df

# 3. Định nghĩa nhãn
def define_target_new(df, horizon=5, atr_multiplier=0.1, long_threshold=0.001, short_threshold=-0.001):
    logging.info(f"Định nghĩa nhãn: horizon={horizon}, atr_multiplier={atr_multiplier}, "
                 f"long_threshold={long_threshold}, short_threshold={short_threshold}")
    df = df.copy()
    
    df["ema_10"] = EMAIndicator(close=df["close"], window=10).ema_indicator().shift(1).fillna(0)
    df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=5).average_true_range().shift(1).fillna(0)
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["threshold"] = df["atr"] * atr_multiplier
    
    df["target"] = 1  # NEUTRAL
    long_condition = (df["close"] > df["ema_10"] + df["threshold"]) & (df["future_return"] >= long_threshold)
    short_condition = (df["close"] < df["ema_10"] - df["threshold"]) & (df["future_return"] <= short_threshold)
    
    df.loc[long_condition, "target"] = 2  # LONG
    df.loc[short_condition, "target"] = 0  # SHORT
    
    df = df.drop(columns=["threshold"], errors='ignore').dropna()
    
    label_ratios = df["target"].value_counts(normalize=True)
    logging.info(f"Phân phối nhãn: {label_ratios}")
    return df

# 4. Chuẩn bị dữ liệu
def prepare_data(df, features, timesteps, horizon):
    logging.info("Chuẩn bị dữ liệu...")
    feature_data = df[features].values
    X = np.lib.stride_tricks.sliding_window_view(feature_data, (timesteps, len(features))).squeeze().copy()
    y = df["target"].values[timesteps-1:].astype(int)
    
    logging.info(f"Kích thước X: {X.shape}, y: {len(y)}")
    
    scaler = RobustScaler()
    X_2d = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_2d).reshape(X.shape)
    
    return X_scaled, y, scaler

# 5. Xây dựng mô hình LSTM
def build_lstm_model(input_shape, lstm_units=32, num_heads=8, dropout_rate=0.5, l2_lambda=0.01):
    inputs = Input(shape=input_shape)
    lstm1 = Bidirectional(LSTM(lstm_units // 2, return_sequences=True, kernel_regularizer=l2(l2_lambda)))(inputs)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(dropout_rate)(lstm1)
    
    lstm1_projected = Dense(input_shape[1], kernel_regularizer=l2(l2_lambda))(lstm1)
    residual1 = Add()([lstm1_projected, inputs])
    
    lstm2 = Bidirectional(LSTM(lstm_units // 4, return_sequences=True, kernel_regularizer=l2(l2_lambda)))(residual1)
    lstm2 = BatchNormalization()(lstm2)
    lstm2 = Dropout(dropout_rate)(lstm2)
    
    lstm2_projected = Dense(input_shape[1], kernel_regularizer=l2(l2_lambda))(lstm2)
    residual2 = Add()([lstm2_projected, residual1])
    
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units // 4)(residual2, residual2)
    attention = BatchNormalization()(attention)
    x = Add()([attention, residual2])
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(16, activation="relu", kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(3, activation="softmax")(x)
    
    return Model(inputs=inputs, outputs=outputs)

# 6. Tối ưu hóa siêu tham số
def objective(trial):
    try:
        logging.info(f"Starting Trial {trial.number}")
        
        timesteps = trial.suggest_int("timesteps", 10, 30)
        horizon = trial.suggest_int("horizon", 2, 5)
        lstm_units = trial.suggest_int("lstm_units", 16, 64)
        num_heads = trial.suggest_int("num_heads", 4, 12)
        dropout_rate = trial.suggest_float("dropout_rate", 0.4, 0.8)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        gamma = trial.suggest_float("gamma", 4.5, 6.0)
        atr_multiplier = trial.suggest_float("atr_multiplier", 0.2, 0.5)
        long_threshold = trial.suggest_float("long_threshold", 0.0005, 0.005)
        short_threshold = -trial.suggest_float("short_threshold", 0.0005, 0.005)
        l2_lambda = trial.suggest_float("l2_lambda", 0.01, 0.1)
        
        alpha = [trial.suggest_float("alpha_0", 0.6, 0.8),
                 trial.suggest_float("alpha_1", 0.1, 0.3),
                 trial.suggest_float("alpha_2", 0.6, 0.8)]
        
        logging.info(f"Trial {trial.number} Params: timesteps={timesteps}, horizon={horizon}, "
                     f"lstm_units={lstm_units}, num_heads={num_heads}, dropout_rate={dropout_rate:.4f}, "
                     f"learning_rate={learning_rate:.6f}, gamma={gamma:.4f}, alpha={alpha}, "
                     f"atr_multiplier={atr_multiplier:.4f}, long_threshold={long_threshold:.4f}, "
                     f"short_threshold={short_threshold:.4f}, l2_lambda={l2_lambda:.4f}")
        
        df_trial = define_target_new(df.copy(), horizon=horizon, atr_multiplier=atr_multiplier,
                                     long_threshold=long_threshold, short_threshold=short_threshold)
        X_scaled, y, scaler = prepare_data(df_trial, features, timesteps, horizon)
        
        val_size = int(len(y) * 0.2)
        X_train, X_val = X_scaled[:-val_size], X_scaled[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]
        
        lstm_model = build_lstm_model((timesteps, len(features)), lstm_units, num_heads, dropout_rate, l2_lambda)
        lstm_model.compile(optimizer=AdamW(learning_rate=learning_rate, weight_decay=1e-4),
                           loss=focal_loss(gamma=gamma, alpha=alpha),
                           metrics=["accuracy"])
        
        callbacks = [EarlyStopping(patience=15, restore_best_weights=True),
                     ReduceLROnPlateau(factor=0.5, patience=5)]
        
        class_weight = {0: 10.0, 1: 1.0, 2: 10.0}
        
        logging.info(f"Trial {trial.number} - Starting model training...")
        lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=64,
                       callbacks=callbacks, class_weight=class_weight, verbose=0)
        
        logging.info(f"Trial {trial.number} - Predicting on validation set...")
        lstm_val_pred = lstm_model.predict(X_val, verbose=0)
        y_pred = np.argmax(lstm_val_pred, axis=1)
        f1 = f1_score(y_val, y_pred, average="weighted")
        
        df_val = df_trial.iloc[-len(y_val):].copy()
        df_val["pred"] = y_pred
        returns = df_val["future_return"]
        profit = np.where(y_pred == 2, returns, np.where(y_pred == 0, -returns, 0)).mean()
        
        logging.info(f"Trial {trial.number} F1-score: {f1:.4f}, Profit: {profit:.4f}")
        objective_value = f1 + profit
        
        if not np.isfinite(objective_value):
            raise ValueError("Objective value is not finite")
        
        return objective_value
    
    except Exception as e:
        logging.error(f"Trial {trial.number} failed with error: {str(e)}")
        return -1

# 7. Huấn luyện và đánh giá
def train_and_evaluate(X, y, best_params, features, timesteps, horizon):
    logging.info("Huấn luyện và đánh giá với Walk-Forward Validation động...")
    
    window_size = int(len(y) * 0.2)
    step_size = int(len(y) * 0.1)
    n_steps = min(5, (len(y) - window_size) // step_size + 1)
    
    f1_scores = []
    for i in range(n_steps):
        start = i * step_size
        end = start + window_size
        train_end = int(start + window_size * 0.7)
        
        X_train, X_test = X[:train_end], X[start:end]
        y_train, y_test = y[:train_end], y[start:end]
        
        logging.info(f"Step {i+1}/{n_steps}: Train {len(X_train)}, Test {len(X_test)}")
        
        lstm_model = build_lstm_model((timesteps, len(features)), 
                                     best_params["lstm_units"], 
                                     best_params["num_heads"], 
                                     best_params["dropout_rate"],
                                     best_params["l2_lambda"])
        lstm_model.compile(optimizer=AdamW(learning_rate=best_params["learning_rate"], weight_decay=1e-4),
                           loss=focal_loss(gamma=best_params["gamma"], 
                                          alpha=[best_params["alpha_0"], best_params["alpha_1"], best_params["alpha_2"]]),
                           metrics=["accuracy"])
        
        callbacks = [EarlyStopping(patience=15, restore_best_weights=True),
                     ReduceLROnPlateau(factor=0.5, patience=5)]
        
        label_counts = pd.Series(y_train).value_counts()
        total_samples = len(y_train)
        class_weight = {0: total_samples / (3 * label_counts.get(0, 1)),
                        1: total_samples / (3 * label_counts.get(1, 1)),
                        2: total_samples / (3 * label_counts.get(2, 1))}
        
        lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64,
                       callbacks=callbacks, class_weight=class_weight, verbose=1)
        
        lstm_test_pred = lstm_model.predict(X_test, verbose=0)
        y_pred = np.argmax(lstm_test_pred, axis=1)
        f1 = f1_score(y_test, y_pred, average="weighted")
        f1_scores.append(f1)
        
        logging.info(f"Step {i+1} F1-score: {f1:.4f}")
    
    avg_f1 = np.mean(f1_scores)
    logging.info(f"Average F1-score: {avg_f1:.3f} ± {np.std(f1_scores):.3f}")
    return lstm_model, scaler, avg_f1

# 8. Pipeline chính
if __name__ == "__main__":
    try:
        df = load_data()
        df = add_features(df)
        
        features = ["price_change", "rsi", "adx", "adx_pos", "adx_neg", "stoch", "momentum", "awesome", 
                    "macd", "bb_upper", "bb_lower", "vwap", "ema_10", "ema_20", "ema_50", "ema_200", 
                    "obv", "roc", "mfi", "vol_breakout", "vol_delta", "rolling_mean_5", "rolling_std_5", 
                    "lag_1", "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "atr", 
                    "rsi_macd_interaction"]
        
        # Tối ưu hóa siêu tham số
        study = optuna.create_study(direction="maximize", sampler=TPESampler(n_startup_trials=20, seed=42))
        study.optimize(objective, n_trials=10, n_jobs=2)  # Sử dụng 2 luồng CPU
        
        best_trial = study.best_trial
        logging.info(f"Best Trial: {best_trial.number}, Value: {best_trial.value:.4f}")
        logging.info(f"Best Params: {best_trial.params}")
        
        trials_df = study.trials_dataframe()
        trials_df.to_csv("optuna_trials_log.csv", index=False)
        
        # Kiểm tra kết quả tối ưu hóa
        if best_trial.value <= 0:
            logging.error(f"Optimization failed: Best value {best_trial.value} is not positive.")
            raise ValueError("Optimization did not yield a valid positive objective value")
        
        # Huấn luyện và đánh giá với tham số tốt nhất
        best_params = study.best_params
        df_final = define_target_new(df, horizon=best_params["horizon"], 
                                    atr_multiplier=best_params["atr_multiplier"],
                                    long_threshold=best_params["long_threshold"],
                                    short_threshold=best_params["short_threshold"])
        X_scaled, y, scaler = prepare_data(df_final, features, best_params["timesteps"], best_params["horizon"])
        
        lstm_model, scaler, avg_f1 = train_and_evaluate(X_scaled, y, best_params, features, 
                                                       best_params["timesteps"], best_params["horizon"])
        
        # Chỉ lưu mô hình nếu tất cả hoàn thành và hiệu suất đạt yêu cầu
        if avg_f1 > 0.5:  # Ngưỡng F1-score, có thể điều chỉnh
            lstm_model.save("lstm_model.h5")
            np.save("scaler.npy", scaler)
            logging.info("Pipeline completed successfully. Model and scaler saved.")
        else:
            logging.warning(f"Pipeline completed but Average F1-score {avg_f1:.3f} is too low. Model not saved.")
    
    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}")
        raise