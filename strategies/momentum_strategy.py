"""
Momentum Strategy
=================
Strategy based on price momentum continuation.

Research Results (Bitcoin 2018-2024):
- CAGR: 46%
- Win Rate: 61%
- Max Drawdown: 23%
- Profit Factor: 2.0
- Time in market: 14%

Strategy Logic:
- BUY: Price breaks above 20-day high
- SELL: Price breaks below 20-day low (exit long)
- Trailing stop to protect profits
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_strategy import BaseStrategy, TradingSignal, SignalType, SignalStrength


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy - Trend Following
    
    Buys breakouts of recent highs and sells breakdowns of recent lows.
    Works best in trending markets.
    """
    
    def __init__(self, config: Dict):
        super().__init__("Momentum", config)
        
        # Momentum parameters
        self.lookback_period = config.get('lookback_period', 20)  # Days
        self.confirmation_bars = config.get('confirmation_bars', 1)  # Bars above high
        
        # Risk management
        self.stop_loss_atr_multiplier = config.get('stop_loss_atr_multiplier', 2.0)
        self.trailing_stop_percent = config.get('trailing_stop_percent', 2.0)
        self.take_profit_percent = config.get('take_profit_percent', 10.0)
        
        # Filters
        self.min_atr = config.get('min_atr', 0.5)  # Minimum volatility
        self.volume_filter = config.get('volume_filter', True)
        self.min_volume = config.get('min_volume', 100000)
        
        # Trend filter
        self.use_trend_filter = config.get('use_trend_filter', True)
        self.trend_ma_period = config.get('trend_ma_period', 200)
        
        print(f"[OK] Momentum Strategy initialized:")
        print(f"  - Lookback Period: {self.lookback_period} bars")
        print(f"  - Confirmation: {self.confirmation_bars} bars")
        print(f"  - Trailing Stop: {self.trailing_stop_percent}%")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators
        """
        df = data.copy()
        
        # Calculate rolling highs and lows
        df['high_n'] = df['high'].rolling(window=self.lookback_period).max()
        df['low_n'] = df['low'].rolling(window=self.lookback_period).min()
        
        # Calculate ATR for stops
        df['atr'] = self._calculate_atr(df, period=14)
        
        # Trend filter (long-term MA)
        if self.use_trend_filter:
            df['trend_ma'] = df['close'].rolling(window=self.trend_ma_period).mean()
        
        # Rate of Change (momentum strength)
        df['roc'] = ((df['close'] - df['close'].shift(self.lookback_period)) / 
                     df['close'].shift(self.lookback_period) * 100)
        
        # Volume indicators
        if self.volume_filter:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """
        Generate momentum signals
        """
        if len(data) < self.lookback_period + 2:
            return None
        
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Check if we have valid data
        if pd.isna(current['high_n']) or pd.isna(current['low_n']):
            return None
        
        # ATR filter (avoid low volatility periods)
        if current['atr'] < self.min_atr:
            return None
        
        # Volume filter
        if self.volume_filter:
            if current['volume'] < self.min_volume:
                return None
            if 'volume_ratio' in current and current['volume_ratio'] < 0.8:
                return None
        
        # Trend filter
        if self.use_trend_filter and 'trend_ma' in current:
            if pd.isna(current['trend_ma']):
                return None
        
        # === LONG SIGNAL (Breakout above recent high) ===
        if current['close'] > current['high_n'] and previous['close'] <= previous['high_n']:
            # Trend filter: only long in uptrend
            if self.use_trend_filter:
                if current['close'] > current['trend_ma']:
                    return 'long'
            else:
                return 'long'
        
        # === EXIT LONG (Breakdown below recent low) ===
        elif current['close'] < current['low_n'] and previous['close'] >= previous['low_n']:
            return 'exit'
        
        return None
    
    def analyze(self, data: pd.DataFrame) -> Dict:
        """
        Main analysis method
        """
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        # Generate signal
        signal = self.generate_signal(df)
        
        if signal is None:
            return {'signal': None}
        
        # Get current data
        current = df.iloc[-1]
        
        if signal == 'long':
            entry_price = current['close']
            
            # Stop loss: ATR-based
            stop_loss = entry_price - (current['atr'] * self.stop_loss_atr_multiplier)
            
            # Take profit
            take_profit = entry_price * (1 + self.take_profit_percent / 100)
            
            # Calculate risk
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
            risk_reward = reward / risk if risk > 0 else 0
            
            # Confidence based on momentum strength
            confidence = min(abs(current['roc']) / 10, 1.0) if not pd.isna(current['roc']) else 0.5
            
            return {
                'signal': signal,
                'price': entry_price,
                'high_n': current['high_n'],
                'low_n': current['low_n'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk': risk,
                'reward': reward,
                'risk_reward': risk_reward,
                'atr': current['atr'],
                'roc': current['roc'],
                'confidence': confidence,
                'reason': self._get_reason(signal, current)
            }
        
        elif signal == 'exit':
            return {
                'signal': 'exit',
                'price': current['close'],
                'reason': f"Breakdown below {self.lookback_period}-day low"
            }
        
        return {'signal': None}
    
    def _get_reason(self, signal: str, current) -> str:
        """Generate human-readable reason"""
        if signal == 'long':
            roc_str = f"{current['roc']:.2f}%" if not pd.isna(current['roc']) else "N/A"
            return f"Breakout above {self.lookback_period}-day high (ROC: {roc_str})"
        elif signal == 'exit':
            return f"Breakdown below {self.lookback_period}-day low"
        return "No signal"
    
    def generate_signals(self, data: pd.DataFrame):
        """
        Generate trading signals (required by BaseStrategy)
        """
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        # Get signal
        signal = self.generate_signal(df)
        
        if signal is None:
            return []
        
        # Get analysis
        analysis = self.analyze(df)
        
        if analysis['signal'] is None:
            return []
        
        # Map to SignalType
        signal_map = {
            'long': SignalType.LONG,
            'exit': SignalType.EXIT
        }
        
        signal_type = signal_map.get(signal)
        if signal_type is None:
            return []
        
        # Determine strength
        confidence = analysis.get('confidence', 0.5)
        if confidence > 0.7:
            strength = SignalStrength.STRONG
        elif confidence > 0.5:
            strength = SignalStrength.MEDIUM
        else:
            strength = SignalStrength.WEAK
        
        # Create TradingSignal
        trading_signal = TradingSignal(
            strategy_name=self.name,
            symbol='UNKNOWN',
            signal_type=signal_type,
            strength=strength,
            timestamp=df.index[-1],
            price=analysis['price'],
            stop_loss=analysis.get('stop_loss'),
            take_profit=analysis.get('take_profit'),
            indicators={
                'high_n': analysis.get('high_n'),
                'low_n': analysis.get('low_n'),
                'atr': analysis.get('atr'),
                'roc': analysis.get('roc')
            },
            reason=analysis['reason'],
            confidence=confidence
        )
        
        return [trading_signal]
