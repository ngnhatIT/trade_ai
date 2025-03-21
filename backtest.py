import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats import skewnorm, pareto
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ta.trend import ADXIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, awesome_oscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import volume_weighted_average_price, OnBalanceVolumeIndicator, MFIIndicator

# Thiết lập kiểu biểu đồ
plt.style.use('default')
plt.rcParams['figure.figsize'] = [15, 10]

# Định nghĩa focal loss
def focal_loss_fn(y_true, y_pred, gamma=5.5888, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
    focal_weight = tf.pow(1.0 - p_t, gamma)
    return tf.reduce_mean(focal_weight * alpha * ce_loss)

class EnhancedBacktestConfig:
    def __init__(self):
        self.initial_equity = 10000.0
        self.risk_per_trade = 0.02
        self.min_leverage = 30
        self.max_leverage = 75
        self.leverage_smoothing_window = 6
        self.taker_fee = 0.0004
        self.maker_fee = 0.0002
        self.slippage_shape_param = 3.5
        self.maintenance_margin = 0.005
        self.stop_loss = 0.03
        self.take_profit = 0.06
        self.max_daily_loss = 0.15
        self.liquidity_impact_factor = 0.001
        self.model_path = "lstm_model.h5"
        self.scaler_path = "scaler.npy"
        self.timesteps = 13
        self.horizon = 6
        self.features = [
            "price_change", "rsi", "adx", "adx_pos", "adx_neg", "stoch",
            "momentum", "awesome", "macd", "bb_upper", "bb_lower", "vwap",
            "ema_10", "ema_20", "ema_50", "ema_200", "obv", "roc", "mfi",
            "vol_breakout", "vol_delta", "rolling_mean_5", "rolling_std_5",
            "lag_1", "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "atr",
            "rsi_macd_interaction"
        ]

class SmartFuturesTradingEngine:
    def __init__(self, config):
        self.config = config
        with tf.keras.utils.custom_object_scope({'focal_loss_fn': focal_loss_fn}):
            self.model = tf.keras.models.load_model(config.model_path)
        self.scaler = np.load(config.scaler_path, allow_pickle=True).item()
        self.reset()

    def reset(self):
        self.equity = self.config.initial_equity
        self.balance = self.config.initial_equity
        self.margin = 0.0
        self.leverage = 30
        self.positions = []
        self.portfolio_values = [self.equity]
        self.max_drawdown = 0
        self.peak = self.equity
        self.avg_liquidity = None
        self.current_position = None

    def calculate_dynamic_leverage(self, volatility):
        smoothed_volatility = pd.Series([volatility]).ewm(span=self.config.leverage_smoothing_window).mean().iloc[-1]
        leverage_range = self.config.max_leverage - self.config.min_leverage
        leverage = self.config.min_leverage + leverage_range / (1 + np.exp(10 * (smoothed_volatility - 0.015)))
        return np.clip(leverage, self.config.min_leverage, self.config.max_leverage)

    def calculate_position_size(self, price, liquidity):
        risk_amount = self.equity * self.config.risk_per_trade
        base_size = (risk_amount * self.leverage) / (price * self.config.stop_loss)
        liquidity_ratio = liquidity / self.avg_liquidity if self.avg_liquidity else 1
        liquidity_adjustment = 1 - np.exp(-self.config.liquidity_impact_factor * liquidity_ratio)
        return base_size * liquidity_adjustment

    def apply_slippage(self, price, is_entry=True):
        base_spread = price * 0.0005
        if self.current_position:
            if is_entry:
                slippage = pareto.rvs(self.config.slippage_shape_param) * base_spread
                return price + slippage if self.current_position['type'] == 'LONG' else price - slippage
            else:
                slippage = skewnorm.rvs(5, loc=base_spread, scale=base_spread/2)
                return price - slippage if self.current_position['type'] == 'LONG' else price + slippage
        return price

    def update_pnl(self, current_price):
        if self.current_position and not self.current_position['closed']:
            if self.current_position['type'] == 'LONG':
                unrealized_pnl = (current_price - self.current_position['entry_price']) * self.current_position['size']
            else:
                unrealized_pnl = (self.current_position['entry_price'] - current_price) * self.current_position['size']
            self.equity = self.balance + self.margin + unrealized_pnl
            self.portfolio_values.append(self.equity)
            self.peak = max(self.peak, self.equity)
            self.max_drawdown = max(self.max_drawdown, (self.peak - self.equity) / self.peak)

    def check_liquidation(self, current_price):
        if self.current_position and not self.current_position['closed']:
            if self.current_position['type'] == 'LONG':
                loss = (self.current_position['entry_price'] - current_price) / self.current_position['entry_price']
            else:
                loss = (current_price - self.current_position['entry_price']) / self.current_position['entry_price']
            return loss >= (1 / self.current_position['leverage'] - self.config.maintenance_margin)
        return False

    def close_position(self, current_data):
        if self.current_position and not self.current_position['closed']:
            exit_price = self.apply_slippage(current_data['close'], is_entry=False)
            fee = self.current_position['size'] * exit_price * self.config.taker_fee
            if self.current_position['type'] == 'LONG':
                pnl = (exit_price - self.current_position['entry_price']) * self.current_position['size'] - fee
            else:
                pnl = (self.current_position['entry_price'] - exit_price) * self.current_position['size'] - fee
            
            self.current_position['exit_price'] = exit_price
            self.current_position['exit_time'] = current_data.name
            self.current_position['pnl'] = pnl
            self.current_position['closed'] = True
            self.balance += pnl
            self.margin = 0
            self.positions.append(self.current_position)
            self.current_position = None

    def execute_trade(self, signal, current_data):
        if not self.current_position and signal != 1:  # 1 là NEUTRAL
            volatility = current_data['atr']
            self.leverage = self.calculate_dynamic_leverage(volatility)
            position_size = self.calculate_position_size(current_data['close'], current_data['volume'])
            entry_price = self.apply_slippage(current_data['close'], is_entry=True)
            fee = position_size * entry_price * self.config.taker_fee
            
            position = {
                'type': 'LONG' if signal == 2 else 'SHORT',
                'entry_price': entry_price,
                'ideal_price': current_data['close'],
                'size': position_size,
                'entry_time': current_data.name,
                'leverage': self.leverage,
                'liquidity_ratio': current_data['volume'] / self.avg_liquidity if self.avg_liquidity else 1,
                'closed': False,
                'pnl': 0
            }
            self.current_position = position
            self.margin = (position_size * entry_price) / self.leverage
            self.balance -= fee

        elif self.current_position and not self.current_position['closed']:
            current_price = current_data['close']
            entry_price = self.current_position['entry_price']
            if self.current_position['type'] == 'LONG':
                if current_price >= entry_price * (1 + self.config.take_profit) or current_price <= entry_price * (1 - self.config.stop_loss):
                    self.close_position(current_data)
            else:
                if current_price <= entry_price * (1 - self.config.take_profit) or current_price >= entry_price * (1 + self.config.stop_loss):
                    self.close_position(current_data)

    def run_backtest(self, data):
        self.avg_liquidity = data['volume'].rolling(24).mean().median()

        for idx in range(len(data)):
            current_data = data.iloc[idx]
            self.update_pnl(current_data['close'])

            if self.current_position and not self.current_position['closed']:
                if self.check_liquidation(current_data['close']):
                    self.close_position(current_data)

            if idx >= self.config.timesteps + self.config.horizon:
                input_data = data[self.config.features].iloc[idx - self.config.timesteps:idx]
                # Chuyển DataFrame thành mảng numpy để loại bỏ tên cột
                scaled_input = self.scaler.transform(input_data.to_numpy())
                signal = self.model.predict(np.expand_dims(scaled_input, axis=0), verbose=0)
                signal = np.argmax(signal, axis=1)[0]
                self.execute_trade(signal, current_data)

        return self.generate_report()

    def generate_report(self):
        total_trades = len(self.positions)
        if total_trades == 0:
            return {
                'final_equity': self.equity,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_leverage_used': 0,
                'avg_slippage': 0,
                'liquidity_impact': 0
            }

        report = {
            'final_equity': self.equity,
            'total_return': (self.equity / self.config.initial_equity - 1) * 100,
            'sharpe_ratio': (np.sqrt(365 * 24) * np.mean(np.diff(self.portfolio_values)) / 
                            np.std(np.diff(self.portfolio_values))) if np.std(np.diff(self.portfolio_values)) != 0 else 0,
            'max_drawdown': self.max_drawdown * 100,
            'win_rate': len([p for p in self.positions if p['pnl'] > 0]) / total_trades,
            'profit_factor': (sum(p['pnl'] for p in self.positions if p['pnl'] > 0) / 
                             abs(sum(p['pnl'] for p in self.positions if p['pnl'] < 0))) if any(p['pnl'] < 0 for p in self.positions) else float('inf'),
            'max_leverage_used': max([p['leverage'] for p in self.positions]) if self.positions else 0,
            'avg_slippage': np.mean([abs(p['entry_price'] / p['ideal_price'] - 1) for p in self.positions]) * 100 if self.positions else 0,
            'liquidity_impact': np.mean([p['liquidity_ratio'] for p in self.positions]) if self.positions else 0
        }
        return report

    def plot_results(self, df):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

        # Biểu đồ giá
        ax1.plot(df.index, df['close'], label='Price', color='blue')
        ax1.set_title('Price')
        ax1.legend()

        # Biểu đồ equity
        ax2.plot(range(len(self.portfolio_values)), self.portfolio_values, label='Equity', color='green')
        ax2.set_title('Equity Curve')
        ax2.legend()

        # Biểu đồ PnL
        pnl_timeline = [{'time': p['exit_time'], 'pnl': p['pnl']} for p in self.positions if p['closed']]
        if pnl_timeline:
            pnl_df = pd.DataFrame(pnl_timeline).set_index('time')
            ax3.bar(pnl_df.index, pnl_df['pnl'], width=0.01, color='blue', label='PnL')
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.set_title('Profit and Loss')
        ax3.legend()

        plt.tight_layout()
        plt.show()

def add_features(df):
    df = df.copy()
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
    return df

if __name__ == "__main__":
    try:
        # Load dữ liệu
        raw_data = pd.read_csv("binance_BTCUSDT_1h.csv", parse_dates=["timestamp"], index_col="timestamp")
        raw_data = raw_data[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors='coerce')
        data = add_features(raw_data)

        # Khởi tạo và chạy backtest
        config = EnhancedBacktestConfig()
        engine = SmartFuturesTradingEngine(config)
        report = engine.run_backtest(data)

        # In báo cáo
        print("\n=== Backtest Report ===")
        print(f"Final Equity: ${report['final_equity']:,.2f}")
        print(f"Total Return: {report['total_return']:.2f}%")
        print(f"Sharpe Ratio: {report['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {report['max_drawdown']:.2f}%")
        print(f"Win Rate: {report['win_rate']:.2f}")
        print(f"Profit Factor: {report['profit_factor']:.2f}")
        print(f"Max Leverage Used: {report['max_leverage_used']:.1f}x")
        print(f"Average Slippage: {report['avg_slippage']:.4f}%")
        print(f"Liquidity Impact: {report['liquidity_impact']:.2f}")

        # Vẽ biểu đồ
        engine.plot_results(data)

    except Exception as e:
        print(f"Error occurred: {str(e)}")