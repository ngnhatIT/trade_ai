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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import logging
import optuna
from optuna.samplers import TPESampler
from ta.trend import ADXIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, awesome_oscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import volume_weighted_average_price, OnBalanceVolumeIndicator, MFIIndicator
from scipy.stats import skew
import shap

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
        focal_loss = alpha_weight * focal_weight * ce_loss
        return tf.reduce_mean(focal_loss)
    return focal_loss_fn

# 1. Load dữ liệu từ CSV
def load_data(file_path="binance_BTCUSDT_1h.csv"):
    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors='coerce')
        df = df.ffill().bfill()
        df = df.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        if df.empty:
            raise ValueError("Dữ liệu sau khi xử lý là rỗng!")
        if len(df) < 5000:
            logging.warning(f"Dữ liệu chỉ có {len(df)} mẫu, nhỏ hơn yêu cầu 5000 mẫu. Chương trình vẫn chạy nhưng có thể không tối ưu.")
        elif len(df) < 50:
            raise ValueError(f"Dữ liệu quá nhỏ ({len(df)} mẫu), không đủ cho timesteps từ 10-50!")
        logging.info(f"Kích thước dữ liệu: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"Không tìm thấy file: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"File {file_path} rỗng hoặc không chứa dữ liệu hợp lệ!")
        raise
    except Exception as e:
        logging.error(f"Lỗi khi tải dữ liệu: {str(e)}")
        raise

# 2. Các hàm tính toán đặc trưng (Đảm bảo không có leakage)
def calc_rsi(df):
    return RSIIndicator(close=df["close"], window=14).rsi().shift(1).fillna(50)

def calc_adx(df):
    return ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14).adx().shift(1).fillna(25)

def calc_adx_pos(df):
    return ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14).adx_pos().shift(1).fillna(0)

def calc_adx_neg(df):
    return ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14).adx_neg().shift(1).fillna(0)

def calc_stoch(df):
    return StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3).stoch().shift(1).fillna(50)

def calc_momentum(df):
    return df["close"].diff(5).shift(1).fillna(0)

def calc_awesome(df):
    return awesome_oscillator(high=df["high"], low=df["low"], window1=5, window2=34).shift(1).fillna(0)

def calc_macd(df):
    return MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9).macd_diff().shift(1).fillna(0)

def calc_bb_upper(df):
    return BollingerBands(close=df["close"], window=20, window_dev=2).bollinger_hband().shift(1).fillna(0)

def calc_bb_lower(df):
    return BollingerBands(close=df["close"], window=20, window_dev=2).bollinger_lband().shift(1).fillna(0)

def calc_vwap(df):
    return volume_weighted_average_price(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14).shift(1).fillna(0)

def calc_ema_20(df):
    return EMAIndicator(close=df["close"], window=20).ema_indicator().shift(1).fillna(0)

def calc_ema_50(df):
    return EMAIndicator(close=df["close"], window=50).ema_indicator().shift(1).fillna(0)

def calc_ema_200(df):
    return EMAIndicator(close=df["close"], window=200).ema_indicator().shift(1).fillna(0)

def calc_obv(df):
    return OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume().shift(1).fillna(0)

def calc_roc(df):
    return ROCIndicator(close=df["close"], window=14).roc().shift(1).fillna(0)

def calc_mfi(df):
    return MFIIndicator(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14).money_flow_index().shift(1).fillna(50)

def calc_atr(df):
    return AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range().shift(1).fillna(0)

def add_features(df):
    logging.info("Bắt đầu thêm đặc trưng...")
    df = df.copy()
    df["price_change"] = df["close"].pct_change().fillna(0)
    
    df["rsi"] = calc_rsi(df)
    df["adx"] = calc_adx(df)
    df["adx_pos"] = calc_adx_pos(df)
    df["adx_neg"] = calc_adx_neg(df)
    df["stoch"] = calc_stoch(df)
    df["momentum"] = calc_momentum(df)
    df["awesome"] = calc_awesome(df)
    df["macd"] = calc_macd(df)
    df["bb_upper"] = calc_bb_upper(df)
    df["bb_lower"] = calc_bb_lower(df)
    df["vwap"] = calc_vwap(df)
    df["ema_20"] = calc_ema_20(df)
    df["ema_50"] = calc_ema_50(df)
    df["ema_200"] = calc_ema_200(df)
    df["obv"] = calc_obv(df)
    df["roc"] = calc_roc(df)
    df["mfi"] = calc_mfi(df)
    df["atr"] = calc_atr(df)
    
    # Chuyển đổi cyclic features thành sine-cosine
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    df["vol_breakout"] = (df["high"] - df["low"]) / df["high"].shift(1).fillna(0)
    df["vol_delta"] = df["obv"].diff().shift(1).fillna(0)
    df["rolling_mean_5"] = df["close"].rolling(window=5).mean().shift(1).fillna(0)
    df["rolling_std_5"] = df["close"].rolling(window=5).std().shift(1).fillna(0)
    df["lag_1"] = df["close"].shift(1).fillna(0)
    
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    logging.info("Hoàn thành thêm đặc trưng.")
    return df

# 3. Định nghĩa nhãn (Điều chỉnh logic gán nhãn để cân bằng phân phối)
def define_target(df, horizon=3, atr_multiplier=2.0):
    logging.info("Bắt đầu định nghĩa nhãn...")
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["atr"] = calc_atr(df)
    df["rsi"] = calc_rsi(df)
    
    # Sử dụng ATR để điều chỉnh ngưỡng theo volatility
    df["threshold"] = df["atr"] * atr_multiplier
    df["breakout_up"] = (df["close"] > df["high"].shift(1).rolling(5).max()) & (df["volume"] > df["volume"].shift(1).rolling(5).mean())
    df["breakout_down"] = (df["close"] < df["low"].shift(1).rolling(5).min()) & (df["volume"] > df["volume"].shift(1).rolling(5).mean())
    
    df["target"] = 1  # Default là NEUTRAL
    for i in range(len(df)):
        curr_threshold = df["threshold"].iloc[i]
        curr_close = df["close"].iloc[i]
        curr_ema_50 = df["ema_50"].iloc[i]
        curr_breakout_up = df["breakout_up"].iloc[i]
        curr_breakout_down = df["breakout_down"].iloc[i]
        curr_rsi = df["rsi"].iloc[i]
        
        # Gán nhãn dựa trên thông tin quá khứ và hiện tại, thêm điều kiện RSI
        if (curr_close > curr_ema_50 + curr_threshold) or curr_breakout_up or (curr_rsi > 70 and curr_close > curr_ema_50):
            df.iloc[i, df.columns.get_loc("target")] = 2  # LONG
        elif (curr_close < curr_ema_50 - curr_threshold) or curr_breakout_down or (curr_rsi < 30 and curr_close < curr_ema_50):
            df.iloc[i, df.columns.get_loc("target")] = 0  # SHORT
    
    df = df.drop(columns=["threshold", "breakout_up", "breakout_down"], errors='ignore').dropna()
    
    # Kiểm tra phân phối nhãn
    label_counts = df["target"].value_counts()
    total_samples = len(df)
    label_ratios = label_counts / total_samples
    logging.info(f"Phân phối nhãn (số lượng): {label_counts}")
    logging.info(f"Phân phối nhãn (tỷ lệ): {label_ratios}")
    
    # Cảnh báo nếu nhãn "LONG" hoặc "SHORT" quá ít
    if label_ratios.get(2, 0) < 0.1 or label_ratios.get(0, 0) < 0.1:
        logging.warning("Phân phối nhãn không cân bằng: Tỷ lệ LONG hoặc SHORT quá thấp (<10%). Có thể cần điều chỉnh logic gán nhãn.")
    
    return df

# 4. Chuẩn bị dữ liệu
def prepare_data(df, features, timesteps, horizon):
    logging.info("Bắt đầu chuẩn bị dữ liệu...")
    feature_data = df[features].values
    
    # Sử dụng sliding_window_view và sao chép để tránh non-contiguous array
    X = np.lib.stride_tricks.sliding_window_view(feature_data, (timesteps, len(features))).squeeze().copy()
    y = df["target"].values[timesteps-1:].astype(int)  # Cast y về int
    
    if len(X) != len(y):
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
    
    # Sử dụng TimeSeriesSplit với gap=horizon để tránh leakage
    tscv = TimeSeriesSplit(n_splits=5, test_size=len(y)//10, gap=horizon)
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        break  # Lấy fold đầu tiên để huấn luyện ban đầu
    
    X_test = X[-len(y) // 10:]
    y_test = y[-len(y) // 10:]
    X_train = X_train[:-len(X_test)]
    y_train = y_train[:-len(X_test)]
    
    # Fit scaler chỉ trên tập train
    scaler = RobustScaler()
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_2d)
    X_train_scaled = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    logging.info(f"Phân phối nhãn ban đầu (train): {pd.Series(y_train).value_counts()}")
    logging.info(f"Phân phối nhãn ban đầu (validation): {pd.Series(y_val).value_counts()}")
    logging.info(f"Phân phối nhãn ban đầu (test): {pd.Series(y_test).value_counts()}")
    logging.info(f"Kích thước dữ liệu: X_train={X_train_scaled.shape}, X_val={X_val_scaled.shape}, X_test={X_test_scaled.shape}")
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler

# 5. Xây dựng mô hình LSTM
def build_lstm_model(input_shape, lstm_units=32, num_heads=8, dropout_rate=0.5):
    inputs = Input(shape=input_shape)
    lstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(0.01)))(inputs)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(dropout_rate)(lstm1)
    
    lstm1_projected = Dense(input_shape[1], kernel_regularizer=l2(0.01))(lstm1)
    residual1 = Add()([lstm1_projected, inputs])
    
    lstm2 = Bidirectional(LSTM(lstm_units // 2, return_sequences=True, kernel_regularizer=l2(0.01)))(residual1)
    lstm2 = BatchNormalization()(lstm2)
    lstm2 = Dropout(dropout_rate)(lstm2)
    
    lstm2_projected = Dense(input_shape[1], kernel_regularizer=l2(0.01))(lstm2)
    residual2 = Add()([lstm2_projected, residual1])
    
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units // 2)(residual2, residual2)
    attention = BatchNormalization()(attention)
    x = Add()([attention, residual2])
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(3, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 6. Tối ưu hóa siêu tham số (Thêm atr_multiplier vào Optuna)
def objective(trial):
    try:
        timesteps = trial.suggest_int("timesteps", 10, 50)
        horizon = trial.suggest_int("horizon", 2, 5)
        lstm_units = trial.suggest_int("lstm_units", 16, 128)
        num_heads = trial.suggest_int("num_heads", 4, 16)
        dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        gamma = trial.suggest_float("gamma", 4.5, 6.0)
        alpha = [trial.suggest_float("alpha_0", 0.6, 0.8),
                 trial.suggest_float("alpha_1", 0.0, 0.2),
                 trial.suggest_float("alpha_2", 0.6, 0.8)]
        atr_multiplier = trial.suggest_float("atr_multiplier", 1.0, 3.0)  # Tối ưu hóa hệ số ATR
        
        logging.info(f"Tối ưu hóa với timesteps={timesteps}, horizon={horizon}, atr_multiplier={atr_multiplier}")
        df_trial = define_target(df.copy(), horizon=horizon, atr_multiplier=atr_multiplier)
        X_train, y_train, X_val, y_val, _, _, _ = prepare_data(df_trial, features, timesteps, horizon)
        if X_train.shape[0] == 0 or X_val.shape[0] == 0:
            raise ValueError("Dữ liệu rỗng sau khi xử lý!")
        
        lstm_model = build_lstm_model((timesteps, len(features)), lstm_units, num_heads, dropout_rate)
        lstm_model.compile(optimizer=AdamW(learning_rate=learning_rate, weight_decay=1e-4), 
                          loss=focal_loss(gamma=gamma, alpha=alpha), metrics=["accuracy"])
        
        callbacks = [EarlyStopping(patience=15, restore_best_weights=True), ReduceLROnPlateau(factor=0.5, patience=5)]
        class_weight = {0: 10.0, 1: 1.0, 2: 10.0}
        lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, callbacks=callbacks, class_weight=class_weight, verbose=0)
        
        lstm_val_pred = lstm_model.predict(X_val, verbose=0)
        y_pred = np.argmax(lstm_val_pred, axis=1)
        f1 = f1_score(y_val, y_pred, average="weighted")
        logging.info(f"Trial hoàn thành với F1-score: {f1}")
        return f1
    except Exception as e:
        logging.error(f"Error in trial: {str(e)}")
        return -1

# 7. Huấn luyện và đánh giá
def train_and_evaluate(X, y, best_params, features, timesteps, horizon):
    logging.info("Bắt đầu huấn luyện và đánh giá mô hình cuối cùng...")
    
    tscv = TimeSeriesSplit(n_splits=5, test_size=len(y)//10, gap=horizon)
    f1_scores = []
    roc_aucs = []
    
    # Fit scaler trên toàn bộ train set sau tuning
    scaler = RobustScaler()
    X_2d = X.reshape(-1, X.shape[-1])
    scaler.fit(X_2d)
    X_scaled = scaler.transform(X_2d).reshape(X.shape)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        logging.info(f"Fold {fold + 1}/5")
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        lstm_model = build_lstm_model((timesteps, len(features)), best_params["lstm_units"], best_params["num_heads"], best_params["dropout_rate"])
        lstm_model.compile(optimizer=AdamW(learning_rate=best_params["learning_rate"], weight_decay=1e-4), 
                           loss=focal_loss(gamma=best_params["gamma"], alpha=[best_params["alpha_0"], best_params["alpha_1"], best_params["alpha_2"]]), 
                           metrics=["accuracy"])
        
        callbacks = [EarlyStopping(patience=15, restore_best_weights=True), ReduceLROnPlateau(factor=0.5, patience=5)]
        
        # Điều chỉnh class_weight dựa trên phân phối nhãn
        label_counts = pd.Series(y_train).value_counts()
        total_samples = len(y_train)
        class_weight = {0: total_samples / (3 * label_counts.get(0, 1)),  # SHORT
                        1: total_samples / (3 * label_counts.get(1, 1)),  # NEUTRAL
                        2: total_samples / (3 * label_counts.get(2, 1))}  # LONG
        logging.info(f"Class Weights: Using Focal Loss with gamma={best_params['gamma']}, alpha={[best_params['alpha_0'], best_params['alpha_1'], best_params['alpha_2']]}, class_weight={class_weight}")
        
        history = lstm_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=64,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        # Monte Carlo Dropout cho meta-labeling
        num_mc_samples = 50
        predictions = np.zeros((len(y_test), 3))
        for _ in range(num_mc_samples):
            preds = lstm_model.predict(X_test, verbose=0)
            predictions += preds / num_mc_samples
        y_pred_mc = np.argmax(predictions, axis=1)
        
        # Thêm SHAP values để giải thích dự đoán
        explainer = shap.DeepExplainer(lstm_model, X_train[:100])
        shap_values = explainer.shap_values(X_test[:50])
        shap.summary_plot(shap_values, X_test[:50], feature_names=features, plot_type="bar", show=False)
        plt.savefig(f"shap_summary_fold_{fold + 1}.png")
        plt.close()
        
        # Báo cáo phân loại
        report = classification_report(y_test, y_pred_mc, target_names=["SHORT", "NEUTRAL", "LONG"], zero_division=1)
        logging.info(f"Fold {fold + 1} Classification Report (Monte Carlo):\n{report}")
        
        # Ma trận nhầm lẫn
        cm = confusion_matrix(y_test, y_pred_mc)
        plt.imshow(cm, cmap="Blues")
        plt.title(f"Confusion Matrix - Fold {fold + 1}")
        plt.colorbar()
        plt.xticks([0, 1, 2], ["SHORT", "NEUTRAL", "LONG"])
        plt.yticks([0, 1, 2], ["SHORT", "NEUTRAL", "LONG"])
        for i in range(3):
            for j in range(3):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.savefig(f"confusion_matrix_fold_{fold + 1}.png")
        plt.close()
        
        # Precision-Recall Curve
        precision = dict()
        recall = dict()
        y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=3)
        for i in range(3):
            precision[i], recall[i], _ = precision_recall_curve(y_test_one_hot[:, i], predictions[:, i])
            plt.plot(recall[i], precision[i], label=f'Class {i} (SHORT=0, NEUTRAL=1, LONG=2)')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - Fold {fold + 1}')
        plt.legend()
        plt.savefig(f"precision_recall_curve_fold_{fold + 1}.png")
        plt.close()
        
        # ROC-AUC
        roc_auc = dict()
        fpr = dict()
        tpr = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], predictions[:, i])
            roc_auc[i] = roc_auc_score(y_test_one_hot[:, i], predictions[:, i])
            logging.info(f"Fold {fold + 1} ROC-AUC for class {i}: {roc_auc[i]:.3f}")
        
        plt.figure()
        for i in range(3):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (ROC-AUC = {roc_auc[i]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Fold {fold + 1}')
        plt.legend()
        plt.savefig(f"roc_curve_fold_{fold + 1}.png")
        plt.close()
        
        # Lưu F1-score và ROC-AUC
        f1 = f1_score(y_test, y_pred_mc, average="weighted")
        f1_scores.append(f1)
        roc_aucs.append(np.mean(list(roc_auc.values())))
        
        # Lưu lịch sử huấn luyện
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title(f"Model Loss - Fold {fold + 1}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title(f"Model Accuracy - Fold {fold + 1}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"training_history_fold_{fold + 1}.png")
        plt.close()
    
    logging.info(f"Average F1-score across folds: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    logging.info(f"Average ROC-AUC across folds: {np.mean(roc_aucs):.3f} ± {np.std(roc_aucs):.3f}")
    return lstm_model, scaler

# 8. Pipeline chính
if __name__ == "__main__":
    try:
        df = load_data()
        df = add_features(df)
        
        features = ["price_change", "rsi", "adx", "adx_pos", "adx_neg", "stoch", "momentum", "awesome", 
                    "macd", "bb_upper", "bb_lower", "vwap", "ema_20", "ema_50", "ema_200", "obv", 
                    "roc", "mfi", "vol_breakout", "vol_delta", "rolling_mean_5", "rolling_std_5", 
                    "lag_1", "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "atr"]
        
        study = optuna.create_study(direction="maximize", sampler=TPESampler(n_startup_trials=20, multivariate=True, seed=42))
        study.optimize(objective, n_trials=100)
        best_params = study.best_params
        logging.info(f"Best Params: {best_params}")
        
        # Chuẩn bị dữ liệu toàn bộ để backtesting
        timesteps = best_params["timesteps"]
        horizon = best_params["horizon"]
        atr_multiplier = best_params["atr_multiplier"]
        df = define_target(df, horizon=horizon, atr_multiplier=atr_multiplier)
        
        feature_data = df[features].values
        X = np.lib.stride_tricks.sliding_window_view(feature_data, (timesteps, len(features))).squeeze().copy()
        y = df["target"].values[timesteps-1:].astype(int)  # Cast y về int
        
        if len(X) != len(y):
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
        
        lstm_model, scaler = train_and_evaluate(X, y, best_params, features, timesteps, horizon)
        
        lstm_model.save("lstm_model.keras")
        np.save("scaler.npy", scaler)
        logging.info("Model and scaler saved successfully.")
    except Exception as e:
        logging.error(f"Error in main pipeline: {str(e)}")
        raise