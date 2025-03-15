import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Input
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, awesome_oscillator
from ta.volatility import AverageTrueRange
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import os
import subprocess

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dữ liệu
def load_data(file_path="binance_BTCUSDT_5m_2021_05_2022_05.csv"):
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors='coerce')
    df = df.ffill().bfill()
    logging.info(f"Kích thước dữ liệu: {df.shape}")
    return df

# Thêm đặc trưng
def add_features(df):
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

# Định nghĩa nhãn
def define_target(df, horizon=5, threshold=0.001):  # Quay lại threshold = 0.1%
    df = df.copy()
    future_return = df["close"].pct_change(periods=horizon).shift(-horizon).fillna(0)
    conditions = [
        (future_return > threshold) & (df["adx_signal"] == 1),
        (future_return < -threshold) & (df["adx_signal"] == -1)
    ]
    df["target"] = np.select(conditions, [1, -1], default=0)
    logging.info(f"Phân phối nhãn:\n{df['target'].value_counts(normalize=True)}")
    return df

# Chuẩn bị dữ liệu
def prepare_data(df, features, timesteps=40):
    X, y = [], []
    data = df[features].values
    targets = df["target"].values
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)
    pd.to_pickle(scaler, "scaler.pkl")
    
    for i in range(timesteps, len(df) - 5):
        X.append(data_scaled[i-timesteps:i])
        y.append(targets[i])
    X = np.array(X)
    y = np.array(y)
    y_cat = tf.keras.utils.to_categorical(y + 1, num_classes=3)
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weight_dict = {0: class_weights[0] * 1.5, 1: class_weights[1] * 1.0, 2: class_weights[2] * 1.5}  # Cân bằng hơn
    logging.info(f"Class weights: {class_weight_dict}")
    
    logging.info(f"X shape: {X.shape}, y shape: {y_cat.shape}")
    return X, y, y_cat, class_weight_dict

# Focal loss
def focal_loss(gamma=1.0, alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(loss, axis=-1)
    return focal_loss_fixed

# Xây dựng mô hình
def build_model(input_shape, model_type="lstm", units_1=64, units_2=32, dropout_rate=0.3):
    model = Sequential()
    model.add(Input(shape=input_shape))
    if model_type == "lstm":
        model.add(LSTM(units_1, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units_2, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(int(units_2/2)))
        model.add(Dropout(dropout_rate))
    elif model_type == "gru":
        model.add(GRU(units_1, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(GRU(units_2, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(GRU(int(units_2/2)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), 
                  loss=focal_loss(gamma=1.0, alpha=0.75), 
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model

# Đánh giá mô hình
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1) - 1
    y_val_classes = np.argmax(y_val, axis=1) - 1
    
    cm = confusion_matrix(y_val_classes, y_pred_classes, labels=[-1, 0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["SHORT", "NEUTRAL", "LONG"], yticklabels=["SHORT", "NEUTRAL", "LONG"])
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    report = classification_report(y_val_classes, y_pred_classes, target_names=["SHORT", "NEUTRAL", "LONG"])
    logging.info(f"Classification Report:\n{report}")

# Huấn luyện với Optuna
def objective(trial):
    timesteps = trial.suggest_int("timesteps", 20, 80, step=10)
    units_1 = trial.suggest_int("units_1", 32, 128, step=16)
    units_2 = trial.suggest_int("units_2", 16, 64, step=16)
    dropout_rate = trial.suggest_float("dropout_rate", 0.4, 0.5, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    model_type = trial.suggest_categorical("model_type", ["lstm", "gru"])
    
    X, y, y_cat, class_weight_dict = prepare_data(df, features, timesteps)
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y_cat[:train_size], y_cat[train_size:]
    
    model = build_model((timesteps, len(features)), model_type, units_1, units_2, dropout_rate)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ],
        verbose=0
    )
    val_auc = max(history.history["val_auc"])
    return val_auc

# Pipeline chính
def main():
    global df, features
    df = load_data()
    df = add_features(df)
    df = define_target(df, threshold=0.001)
    
    features = ["price_change", "rsi", "atr", "adx", "adx_pos", "adx_neg", "stoch", "momentum", "awesome", "hour", "dayofweek"]
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    
    best_params = study.best_params
    logging.info(f"Best params: {best_params}")
    
    timesteps = best_params["timesteps"]
    X, y, y_cat, class_weight_dict = prepare_data(df, features, timesteps)
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y_cat[:train_size], y_cat[train_size:]
    
    model = build_model((timesteps, len(features)), best_params["model_type"], 
                       best_params["units_1"], best_params["units_2"], best_params["dropout_rate"])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=best_params["batch_size"],
        class_weight=class_weight_dict,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ],
        verbose=1
    )
    
    evaluate_model(model, X_val, y_val)
    logging.info(f"Best val_auc: {max(history.history['val_auc'])}")
    logging.info(f"Best val_accuracy: {max(history.history['val_accuracy'])}")
    model.save("optimized_model.keras")
    
    # Kích hoạt file realtime sau khi học xong
    #subprocess.run(["python", "realtime_trading.py"])

if __name__ == "__main__":
    main()