import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import optuna
import pickle

# Cấu hình logging (ghi vào file trading_log.log)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_log.log'),
        logging.StreamHandler()  # Hiển thị log trên console
    ]
)

# Load dữ liệu
def load_data(file_path="binance_BTCUSDT_5m.csv"):
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors='coerce')
    df = df.ffill().bfill()
    logging.info(f"Kích thước dữ liệu: {df.shape}")
    return df

# Thêm đặc trưng
def add_features(df):
    from ta.trend import MACD
    from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
    from ta.volatility import AverageTrueRange, BollingerBands
    
    df = df.copy()
    df["price_change_lag1"] = df["close"].pct_change().shift(1).fillna(0)
    df["price_change_lag2"] = df["close"].pct_change(periods=2).shift(1).fillna(0)
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["rsi"] = RSIIndicator(close=df["close"], window=7).rsi().shift(1).fillna(50)
    df["stoch"] = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=7, smooth_window=3).stoch().shift(1).fillna(50)
    macd = MACD(close=df["close"], window_slow=13, window_fast=6, window_sign=4)
    df["macd"] = macd.macd().shift(1).fillna(0)
    df["macd_signal"] = macd.macd_signal().shift(1).fillna(0)
    df["roc"] = ROCIndicator(close=df["close"], window=5).roc().shift(1).fillna(0)
    df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=7).average_true_range().shift(1).fillna(0)
    df["atr_sma"] = df["atr"].rolling(window=7).mean().shift(1).fillna(0)
    df["atr_normalized"] = (df["atr"] / df["close"]).shift(1).fillna(0)
    bb = BollingerBands(close=df["close"], window=10, window_dev=2)
    df["bollinger_width"] = (bb.bollinger_hband() - bb.bollinger_lband()).shift(1).fillna(0)
    df["ema_10"] = df["close"].ewm(span=10).mean().shift(1).fillna(df["close"])
    df["ema_20"] = df["close"].ewm(span=20).mean().shift(1).fillna(df["close"])
    df["ema_cross"] = np.where(df["ema_10"] > df["ema_20"], 1, -1)
    df["volume_change"] = df["volume"].pct_change().shift(1).fillna(0)
    df["volume_sma_7"] = df["volume"].rolling(window=7).mean().shift(1).fillna(0)
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df

# Định nghĩa nhãn
def define_target(df, horizon=2, threshold=0.0005):
    df = df.copy()
    future_return = df["close"].pct_change(periods=horizon).shift(-horizon).fillna(0)
    
    conditions = [
        (future_return > threshold) & (df["rsi"] < 70) & (df["macd"] > df["macd_signal"]) & 
        (df["stoch"].between(20, 80)) & (df["roc"] > 0) & (df["atr"] > df["atr_sma"]) & 
        (df["close"] > df["ema_10"]),
        (future_return < -threshold) & (df["rsi"] > 30) & (df["macd"] < df["macd_signal"]) & 
        (df["stoch"].between(20, 80)) & (df["roc"] < 0) & (df["atr"] > df["atr_sma"]) & 
        (df["close"] < df["ema_10"])
    ]
    df["target"] = np.select(conditions, [1, -1], default=0)
    
    label_dist = df["target"].value_counts(normalize=True)
    logging.info(f"Phân phối nhãn:\n{label_dist}")
    return df

# Chuẩn bị dữ liệu cho LSTM
def prepare_lstm_data(df, sequence_length):
    features = [
        "price_change_lag1", "price_change_lag2", "high_low_range",
        "rsi", "stoch", "macd", "macd_signal", "roc",
        "atr_normalized", "bollinger_width",
        "ema_10", "ema_20", "ema_cross",
        "volume_change", "volume_sma_7",
        "hour", "dayofweek"
    ]
    
    X, y = [], []
    scaler = RobustScaler()
    feature_data = scaler.fit_transform(df[features])
    
    for i in range(sequence_length, len(df)):
        X.append(feature_data[i-sequence_length:i])
        y.append(df["target"].iloc[i])
    
    X = np.array(X)
    y = np.array(y)
    y_encoded = np.zeros((len(y), 3))
    for i, label in enumerate(y):
        y_encoded[i, label + 1] = 1
    
    return X, y_encoded, scaler

# Tính class weights
def get_class_weights(y):
    y_classes = np.argmax(y, axis=1) - 1
    class_weights = compute_class_weight("balanced", classes=np.array([-1, 0, 1]), y=y_classes)
    return dict(enumerate(class_weights))

# Hàm mục tiêu cho Optuna
def objective(trial):
    sequence_length = trial.suggest_int("sequence_length", 10, 50)
    lstm_units_1 = trial.suggest_int("lstm_units_1", 32, 128)
    lstm_units_2 = trial.suggest_int("lstm_units_2", 16, 64)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    cnn_filters = trial.suggest_int("cnn_filters", 16, 64)
    
    X, y, _ = prepare_lstm_data(df, sequence_length)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = Sequential([
        Input(shape=(sequence_length, X.shape[2])),
        Conv1D(filters=cnn_filters, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(lstm_units_1, return_sequences=True)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Bidirectional(LSTM(lstm_units_2)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax")
    ])
    
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    class_weights = get_class_weights(y_train)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        class_weight=class_weights,
        verbose=0
    )
    
    val_accuracy = max(history.history["val_accuracy"])
    return val_accuracy

# Huấn luyện mô hình với tham số tốt nhất
def build_and_train_lstm(X, y, best_params):
    sequence_length = best_params["sequence_length"]
    model = Sequential([
        Input(shape=(sequence_length, X.shape[2])),
        Conv1D(filters=best_params["cnn_filters"], kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(best_params["lstm_units_1"], return_sequences=True)),
        BatchNormalization(),
        Dropout(best_params["dropout_rate"]),
        Bidirectional(LSTM(best_params["lstm_units_2"])),
        BatchNormalization(),
        Dropout(best_params["dropout_rate"]),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax")
    ])
    
    optimizer = AdamW(learning_rate=best_params["learning_rate"], weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    class_weights = get_class_weights(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        class_weight=class_weights,
        verbose=1
    )
    
    return model, history, X_test, y_test

# Đánh giá mô hình và log dự đoán trên tập test
def evaluate_model(model, X_test, y_test, df, sequence_length, scaler):
    features = [
        "price_change_lag1", "price_change_lag2", "high_low_range",
        "rsi", "stoch", "macd", "macd_signal", "roc",
        "atr_normalized", "bollinger_width",
        "ema_10", "ema_20", "ema_cross",
        "volume_change", "volume_sma_7",
        "hour", "dayofweek"
    ]
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1) - 1
    y_test_classes = np.argmax(y_test, axis=1) - 1
    
    report = classification_report(y_test_classes, y_pred_classes, target_names=["SHORT", "NEUTRAL", "LONG"])
    logging.info(f"Báo cáo phân loại:\n{report}")
    
    # Log dự đoán trên tập test
    test_start_idx = len(df) - len(X_test) - sequence_length + 1
    for i in range(len(X_test)):
        timestamp = df.index[test_start_idx + i]
        probabilities = y_pred[i].tolist()
        predicted_label = y_pred_classes[i]
        confidence = max(probabilities)
        label_str = "LONG" if predicted_label == 1 else "SHORT" if predicted_label == -1 else "NEUTRAL"
        close_price = df["close"].iloc[test_start_idx + i]
        atr = df["atr"].iloc[test_start_idx + i]
        
        if predicted_label != 0:  # Chỉ log nếu là LONG hoặc SHORT
            if predicted_label == 1:  # LONG
                sl = close_price - 1.5 * atr
                tp = close_price + 2 * atr
            else:  # SHORT
                sl = close_price + 1.5 * atr
                tp = close_price - 2 * atr
            logging.info(f"{timestamp} - Prediction: {label_str}, Probabilities: {probabilities}, Confidence: {confidence:.4f}, Close Price: {close_price:.2f}, SL: {sl:.2f}, TP: {tp:.2f}")
        else:
            logging.info(f"{timestamp} - Prediction: {label_str}, Probabilities: {probabilities}, Confidence: {confidence:.4f}, Close Price: {close_price:.2f}")
    
    return report

# Pipeline chính
if __name__ == "__main__":
    # Load và xử lý dữ liệu
    df = load_data()
    df = add_features(df)
    df = define_target(df)

    # Tối ưu hóa với Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  # Thử 20 cấu hình
    best_params = study.best_params
    logging.info(f"Tham số tốt nhất: {best_params}")

    # Chuẩn bị dữ liệu với timesteps tối ưu
    X, y, scaler = prepare_lstm_data(df, best_params["sequence_length"])

    # Huấn luyện và đánh giá
    model, history, X_test, y_test = build_and_train_lstm(X, y, best_params)
    report = evaluate_model(model, X_test, y_test, df, best_params["sequence_length"], scaler)

    # Lưu mô hình và scaler để dùng realtime
    model.save("lstm_model.h5")
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logging.info("Đã lưu mô hình vào lstm_model.h5 và scaler vào scaler.pkl")

    # Vẽ biểu đồ huấn luyện
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("training_history_optuna_quality.png")
    plt.close()