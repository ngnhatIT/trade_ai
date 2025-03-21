import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats import skewnorm, pareto
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ta import add_all_ta_features

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [15, 10]

class EnhancedBacktestConfig:
    def __init__(self):
        # Thiết lập tài khoản
        self.initial_equity = 10000.0       # Vốn ban đầu (USD)
        self.risk_per_trade = 0.02          # % vốn mạo hiểm mỗi lệnh
        self.min_leverage = 30              # Đòn bẩy tối thiểu
        self.max_leverage = 75              # Đòn bẩy tối đa
        self.leverage_smoothing_window = 6  # Cửa sổ làm mịn đòn bẩy (giờ)
        
        # Chi phí giao dịch
        self.taker_fee = 0.0004             # Phí maker/taker
        self.maker_fee = 0.0002
        self.slippage_shape_param = 3.5     # Tham số hình dạng phân phối trượt giá
        
        # Quản lý rủi ro
        self.maintenance_margin = 0.005     # 0.5% margin duy trì
        self.stop_loss = 0.03               # 3% stop loss
        self.take_profit = 0.06             # 6% take profit
        self.max_daily_loss = 0.15          # Tối đa thua lỗ 15%/ngày
        self.liquidity_impact_factor = 0.001 # Ảnh hưởng thanh khoản
        
        # Thông số mô hình
        self.model_path = "lstm_model.keras"
        self.scaler_path = "scaler.npy"
        self.timesteps = 24
        self.horizon = 6
        self.features = [
            "price_change", "rsi", "adx", "adx_pos", "adx_neg", "stoch", 
            "momentum", "awesome", "macd", "bb_upper", "bb_lower", "vwap",
            "ema_10", "ema_20", "ema_50", "ema_200", "obv", "roc", "mfi",
            "vol_breakout", "vol_delta", "rolling_mean_5", "rolling_std_5",
            "lag_1", "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "atr"
        ]

class SmartFuturesTradingEngine:
    def __init__(self, config):
        self.config = config
        self.model = tf.keras.models.load_model(config.model_path)
        self.scaler = np.load(config.scaler_path, allow_pickle=True).item()
        self.reset()
    
    def reset(self):
        # Trạng thái tài khoản
        self.equity = self.config.initial_equity
        self.balance = self.config.initial_equity
        self.margin = 0.0
        self.leverage = 30
        self.daily_pnl = []
        
        # Lịch sử giao dịch
        self.positions = []
        self.trade_history = []
        self.current_position = None
        self.historical_data = []
        
        # Theo dõi hiệu suất
        self.portfolio_values = [self.equity]
        self.max_drawdown = 0
        self.peak = self.equity
        self.avg_liquidity = None
        self.liquidity_zones = None

    def calculate_dynamic_leverage(self, volatility_series):
        """Đòn bẩy động với làm mịn EMA"""
        smoothed_volatility = volatility_series.dropna().ewm(span=self.config.leverage_smoothing_window).mean().iloc[-1]
        
        # Hàm chuyển đổi phi tuyến tính
        leverage_range = self.config.max_leverage - self.config.min_leverage
        leverage = self.config.min_leverage + leverage_range / (1 + np.exp(10*(smoothed_volatility - 0.015)))
        
        return np.clip(leverage, self.config.min_leverage, self.config.max_leverage)

    def calculate_position_size(self, price, liquidity):
        """Tính toán vị thế với độ sâu thanh khoản"""
        risk_amount = self.equity * self.config.risk_per_trade
        base_size = (risk_amount * self.leverage) / (price * self.config.stop_loss)
        
        # Điều chỉnh theo thanh khoản
        liquidity_ratio = liquidity / self.avg_liquidity if self.avg_liquidity != 0 else 0
        liquidity_adjustment = 1 - np.exp(-self.config.liquidity_impact_factor * liquidity_ratio)
        
        return base_size * liquidity_adjustment

    def apply_slippage(self, price, is_entry=True):
        """Phân phối trượt giá lệch với đuôi nặng"""
        base_spread = price * 0.0005
        if self.current_position:
            if is_entry:
                # Phân phối Pareto cho entry 
                slippage = pareto.rvs(self.config.slippage_shape_param) * base_spread
                return price + slippage if self.current_position['type'] == 'LONG' else price - slippage
            else:
                # Phân phối lệch chuẩn cho exit
                slippage = skewnorm.rvs(5, loc=base_spread, scale=base_spread/2)
                return price - slippage if self.current_position['type'] == 'LONG' else price + slippage
        else:
            return price

    def update_real_time_pnl(self, current_price):
        """Cập nhật PnL thời gian thực"""
        if self.current_position and not self.current_position['closed']:
            if self.current_position['type'] == 'LONG':
                unrealized_pnl = (current_price - self.current_position['entry_price']) * self.current_position['size']
            else:
                unrealized_pnl = (self.current_position['entry_price'] - current_price) * self.current_position['size']
            
            # Cập nhật equity và margin
            self.equity = self.balance + self.margin + unrealized_pnl
            self.portfolio_values.append(self.equity)
            
            # Cập nhật drawdown
            self.peak = max(self.peak, self.equity)
            self.max_drawdown = max(self.max_drawdown, (self.peak - self.equity)/self.peak)

    def calculate_liquidity_zones(self, df):
        """Xác định vùng thanh khoản dùng Volume Profile"""
        vp = df.groupby(pd.cut(df.index.hour, bins=24))['volume'].mean()
        self.liquidity_zones = {
            'support': df['low'].rolling(24).mean(),
            'resistance': df['high'].rolling(24).mean(),
            'volume_profile': vp.reindex(df.index, method='ffill')
        }

    def plot_enhanced_results(self, df):
        fig, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # Biểu đồ giá với vùng thanh khoản
        ax[0].plot(df['close'], label='Price')
        ax[0].plot(self.liquidity_zones['support'], '--', color='green', alpha=0.5, label='Support')
        ax[0].plot(self.liquidity_zones['resistance'], '--', color='red', alpha=0.5, label='Resistance')
        ax[0].fill_between(df.index, 
                         self.liquidity_zones['volume_profile'].quantile(0.25), 
                         self.liquidity_zones['volume_profile'].quantile(0.75), 
                         color='gray', alpha=0.2, label='Liquidity Zone')
        
        # Biểu đồ equity và drawdown
        ax[1].plot(self.portfolio_values, label='Equity Curve')
        ax[1].fill_between(range(len(self.portfolio_values)), 
                         self.portfolio_values, 
                         self.peak * np.ones(len(self.portfolio_values)), 
                         color='red', alpha=0.3, label='Drawdown')
        ax[1].legend()
        
        # Biểu đồ đòn bẩy
        leverage_timeline = [{'time': p['entry_time'], 'leverage': p['leverage']} for p in self.positions]
        leverage_df = pd.DataFrame(leverage_timeline).set_index('time')
        ax[2].bar(leverage_df.index, leverage_df['leverage'], width=0.01, color='purple', label='Leverage')
        ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax[2].legend()
        
        # Biểu đồ PnL theo thời gian
        pnl_timeline = [{'time': p['exit_time'], 'pnl': p['pnl']} for p in self.positions]
        pnl_df = pd.DataFrame(pnl_timeline).set_index('time')
        ax[3].bar(pnl_df.index, pnl_df['pnl'], width=0.01, color='blue', label='PnL')
        ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax[3].legend()
        
        plt.tight_layout()
        plt.show()

    def run_backtest(self, data):
        # Thêm tính toán thanh khoản
        self.avg_liquidity = data['volume'].rolling(24).mean().median()
        self.calculate_liquidity_zones(data)
        
        # Vòng lặp chính với cập nhật thời gian thực
        for idx in range(len(data)):
            current_data = data.iloc[idx]
            self.update_real_time_pnl(current_data['close'])
            
            # Kiểm tra liquidation
            if self.current_position and not self.current_position['closed']:
                if self.check_liquidation(current_data['close']):
                    self.close_position(current_data)
                    
            # Thực thi lệnh mới
            if idx in range(self.config.timesteps + self.config.horizon, len(data)):
                signal = self.model.predict(np.expand_dims(self.scaler.transform(data[self.config.features].iloc[idx-self.config.timesteps:idx]), axis=0))
                signal = np.argmax(signal, axis=1)[0]
                self.execute_trade(signal, current_data)
                
            # Cập nhật PnL hàng giờ
            if self.current_position and not self.current_position['closed']:
                self.update_unrealized_pnl(current_data['close'])
        
        return self.generate_enhanced_report()

    def generate_enhanced_report(self):
        report = {
            'final_equity': self.equity,
            'total_return': (self.equity / self.config.initial_equity - 1) * 100,
            'sharpe_ratio': np.sqrt(365*24) * np.mean(np.diff(self.portfolio_values)) / np.std(np.diff(self.portfolio_values)),
            'max_drawdown': self.max_drawdown * 100,
            'win_rate': len([p for p in self.positions if p['pnl'] > 0]) / len(self.positions),
            'profit_factor': sum(p['pnl'] for p in self.positions if p['pnl'] > 0) / 
                            abs(sum(p['pnl'] for p in self.positions if p['pnl'] < 0)),
            'max_leverage_used': max([p['leverage'] for p in self.positions]),
            'avg_slippage': np.mean([abs(p['entry_price']/p['ideal_price']-1) for p in self.positions]),
            'liquidity_impact': np.mean([p['liquidity_ratio'] for p in self.positions]),
            'realized_vs_unrealized': sum(p['pnl'] for p in self.positions)/self.portfolio_values[-1]
        }
        return report

# Sử dụng
if __name__ == "__main__":
    config = EnhancedBacktestConfig()
    engine = SmartFuturesTradingEngine(config)
    
    # Load dữ liệu
    raw_data = pd.read_csv("binance_BTCUSDT_1h.csv", parse_dates=["timestamp"], index_col="timestamp")
    report = engine.run_backtest(raw_data)
    
    print("\n=== Enhanced Backtest Report ===")
    print(f"Final Equity: ${report['final_equity']:,.2f}")
    print(f"Total Return: {report['total_return']:.2f}%")
    print(f"Max Leverage Used: {report['max_leverage_used']:.1f}x")
    print(f"Average Slippage: {report['avg_slippage']:.4f}%")
    print(f"Liquidity Impact: {report['liquidity_impact']:.2f}")
    print(f"Realized/Unrealized PnL Ratio: {report['realized_vs_unrealized']:.2f}")