"""
EMA Cross Strategy
=================
אסטרטגיית מעברי EMA (Exponential Moving Average)

אסטרטגיה קלאסית המבוססת על מעבר ממוצעים נעים:
- קנה: EMA מהיר חוצה מעל EMA איטי (Golden Cross)
- מכור: EMA מהיר חוצה מתחת EMA איטי (Death Cross)

משתמש במספר אינדיקטורים לאישור:
- EMA(12), EMA(26), EMA(50) - זיהוי מגמות
- RSI - מניעת קניה ב-overbought
- Volume - אישור כוח התנועה
- ATR - חישוב Stop Loss דינמי
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from .base_strategy import (
    BaseStrategy, TradingSignal, SignalType, SignalStrength
)
from indicators.custom_indicators import TechnicalIndicators


class EMACrossStrategy(BaseStrategy):
    """אסטרטגיית מעברי EMA"""
    
    def __init__(self, config: Dict):
        """
        אתחול אסטרטגיה
        
        Args:
            config: קונפיגורציה מקובץ trading_config.yaml
        """
        super().__init__(name="EMA_Cross", config=config)
        
        # Strategy parameters from config
        self.fast_ema = config.get('fast_ema', 12)
        self.slow_ema = config.get('slow_ema', 26)
        self.signal_line = config.get('signal_line', 9)
        self.min_volume = config.get('min_volume', 100000)
        
        # Additional filters
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.volume_threshold = config.get('volume_threshold', 1.2)  # 1.2x average
        
        # Initialize technical indicators calculator
        self.indicators = TechnicalIndicators()
        
        # Track previous values for cross detection
        self.prev_fast_ema = None
        self.prev_slow_ema = None
        
    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        חישוב כל האינדיקטורים הנדרשים
        
        Args:
            data: DataFrame עם OHLCV
            
        Returns:
            DataFrame עם אינדיקטורים מחושבים
        """
        if data is None or len(data) < self.slow_ema + 10:
            raise ValueError(f"Need at least {self.slow_ema + 10} bars for analysis")
        
        df = data.copy()
        
        # Calculate EMAs
        df[f'ema_{self.fast_ema}'] = TechnicalIndicators.ema(
            df['close'], period=self.fast_ema
        )
        df[f'ema_{self.slow_ema}'] = TechnicalIndicators.ema(
            df['close'], period=self.slow_ema
        )
        df['ema_50'] = TechnicalIndicators.ema(df['close'], period=50)
        
        # Calculate trend indicator (EMA difference)
        df['ema_diff'] = df[f'ema_{self.fast_ema}'] - df[f'ema_{self.slow_ema}']
        df['ema_diff_pct'] = (df['ema_diff'] / df['close']) * 100
        
        # Calculate RSI for overbought/oversold filter
        df['rsi'] = TechnicalIndicators.rsi(df['close'], period=14)
        
        # Calculate ATR for stop loss
        df['atr'] = TechnicalIndicators.atr(df, period=14)
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['relative_volume'] = df['volume'] / df['volume_sma']
        
        # MACD for additional confirmation
        macd_line, macd_signal, macd_histogram = TechnicalIndicators.macd(
            df['close'],
            fast=self.fast_ema,
            slow=self.slow_ema,
            signal=self.signal_line
        )
        df['macd'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_histogram
        
        # Identify trend direction
        df['trend'] = 'neutral'
        df.loc[
            (df[f'ema_{self.fast_ema}'] > df[f'ema_{self.slow_ema}']) &
            (df['close'] > df['ema_50']),
            'trend'
        ] = 'bullish'
        df.loc[
            (df[f'ema_{self.fast_ema}'] < df[f'ema_{self.slow_ema}']) &
            (df['close'] < df['ema_50']),
            'trend'
        ] = 'bearish'
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        זיהוי מעברי EMA ויצירת סיגנלים
        
        Args:
            data: DataFrame עם אינדיקטורים מחושבים
            
        Returns:
            רשימת סיגנלי מסחר
        """
        signals = []
        
        if len(data) < 2:
            return signals
        
        # Get current and previous bars
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Extract values
        fast_col = f'ema_{self.fast_ema}'
        slow_col = f'ema_{self.slow_ema}'
        
        current_fast = current[fast_col]
        current_slow = current[slow_col]
        prev_fast = previous[fast_col]
        prev_slow = previous[slow_col]
        
        current_price = current['close']
        current_rsi = current['rsi']
        current_volume = current.get('relative_volume', 1.0)
        current_atr = current['atr']
        
        # --- BULLISH CROSS DETECTION ---
        # Fast EMA crossed above Slow EMA
        if prev_fast <= prev_slow and current_fast > current_slow:
            
            # Apply filters
            passed_filters = True
            filter_reasons = []
            
            # Filter 1: Not overbought
            if current_rsi > self.rsi_overbought:
                passed_filters = False
                filter_reasons.append(f"RSI overbought ({current_rsi:.1f})")
            
            # Filter 2: Sufficient volume
            if current_volume < self.volume_threshold:
                passed_filters = False
                filter_reasons.append(f"Low volume ({current_volume:.2f}x)")
            
            # Filter 3: Price above EMA50 (uptrend confirmation)
            if current_price < current['ema_50']:
                passed_filters = False
                filter_reasons.append("Price below EMA50")
            
            # Filter 4: MACD confirmation
            if current['macd'] < current['macd_signal']:
                passed_filters = False
                filter_reasons.append("MACD bearish")
            
            if passed_filters:
                # Calculate stop loss and take profit
                stop_loss = self.calculate_stop_loss(
                    current_price, SignalType.BUY, atr=current_atr
                )
                take_profit = self.calculate_take_profit(
                    current_price, stop_loss, SignalType.BUY, risk_reward_ratio=2.5
                )
                
                # Calculate confidence based on indicators alignment
                confidence = self._calculate_confidence(current, signal_type='buy')
                strength = self.get_signal_strength(confidence)
                
                signal = TradingSignal(
                    timestamp=current.name if hasattr(current.name, 'to_pydatetime') else datetime.now(),
                    symbol="",  # Will be set by caller
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    strategy_name=self.name,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    indicators={
                        'ema_fast': current_fast,
                        'ema_slow': current_slow,
                        'ema_50': current['ema_50'],
                        'rsi': current_rsi,
                        'relative_volume': current_volume,
                        'atr': current_atr,
                        'macd': current['macd'],
                        'trend': current['trend']
                    },
                    reason=f"Bullish EMA cross: EMA{self.fast_ema} crossed above EMA{self.slow_ema}",
                    confidence=confidence
                )
                signals.append(signal)
                self.add_signal_to_history(signal)
        
        # --- BEARISH CROSS DETECTION ---
        # Fast EMA crossed below Slow EMA
        elif prev_fast >= prev_slow and current_fast < current_slow:
            
            # Apply filters
            passed_filters = True
            filter_reasons = []
            
            # Filter 1: Not oversold
            if current_rsi < self.rsi_oversold:
                passed_filters = False
                filter_reasons.append(f"RSI oversold ({current_rsi:.1f})")
            
            # Filter 2: Sufficient volume
            if current_volume < self.volume_threshold:
                passed_filters = False
                filter_reasons.append(f"Low volume ({current_volume:.2f}x)")
            
            # Filter 3: Price below EMA50 (downtrend confirmation)
            if current_price > current['ema_50']:
                passed_filters = False
                filter_reasons.append("Price above EMA50")
            
            # Filter 4: MACD confirmation
            if current['macd'] > current['macd_signal']:
                passed_filters = False
                filter_reasons.append("MACD bullish")
            
            if passed_filters:
                # Calculate stop loss and take profit
                stop_loss = self.calculate_stop_loss(
                    current_price, SignalType.SELL, atr=current_atr
                )
                take_profit = self.calculate_take_profit(
                    current_price, stop_loss, SignalType.SELL, risk_reward_ratio=2.5
                )
                
                confidence = self._calculate_confidence(current, signal_type='sell')
                strength = self.get_signal_strength(confidence)
                
                signal = TradingSignal(
                    timestamp=current.name if hasattr(current.name, 'to_pydatetime') else datetime.now(),
                    symbol="",
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=current_price,
                    strategy_name=self.name,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    indicators={
                        'ema_fast': current_fast,
                        'ema_slow': current_slow,
                        'ema_50': current['ema_50'],
                        'rsi': current_rsi,
                        'relative_volume': current_volume,
                        'atr': current_atr,
                        'macd': current['macd'],
                        'trend': current['trend']
                    },
                    reason=f"Bearish EMA cross: EMA{self.fast_ema} crossed below EMA{self.slow_ema}",
                    confidence=confidence
                )
                signals.append(signal)
                self.add_signal_to_history(signal)
        
        return signals
    
    def _calculate_confidence(self, bar: pd.Series, signal_type: str) -> float:
        """
        חישוב רמת ביטחון בסיגנל על בסיס התאמת אינדיקטורים
        
        Args:
            bar: השורה הנוכחית
            signal_type: 'buy' או 'sell'
            
        Returns:
            רמת ביטחון (0-1)
        """
        confidence = 0.5  # Base confidence
        
        if signal_type == 'buy':
            # Strong uptrend
            if bar['trend'] == 'bullish':
                confidence += 0.15
            
            # MACD bullish and above signal
            if bar['macd'] > bar['macd_signal'] and bar['macd_histogram'] > 0:
                confidence += 0.15
            
            # High volume
            if bar.get('relative_volume', 1.0) > 1.5:
                confidence += 0.1
            
            # RSI in good range (not extreme)
            if 40 < bar['rsi'] < 60:
                confidence += 0.1
                
        else:  # sell
            # Strong downtrend
            if bar['trend'] == 'bearish':
                confidence += 0.15
            
            # MACD bearish
            if bar['macd'] < bar['macd_signal'] and bar['macd_histogram'] < 0:
                confidence += 0.15
            
            # High volume
            if bar.get('relative_volume', 1.0) > 1.5:
                confidence += 0.1
            
            # RSI in good range
            if 40 < bar['rsi'] < 60:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def get_current_trend(self, data: pd.DataFrame) -> str:
        """
        קבלת המגמה הנוכחית
        
        Returns:
            'bullish', 'bearish', או 'neutral'
        """
        if len(data) == 0:
            return 'neutral'
        
        current = data.iloc[-1]
        return current.get('trend', 'neutral')
    
    def __repr__(self):
        return (f"EMA Cross Strategy (Fast: {self.fast_ema}, "
                f"Slow: {self.slow_ema}) - {self.enabled}")
