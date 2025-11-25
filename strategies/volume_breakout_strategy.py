"""
Volume Breakout Strategy
=======================
אסטרטגיית פריצות נפח (Volume Breakout)

אסטרטגיה המזהה פריצות מחיר המלוות בנפח גבוה:
- נפח חריג מעיד על עניין מוסדי/כוח השוק
- פריצה עם נפח = סבירות גבוהה להמשך תנועה
- ללא נפח = פריצה שקרית (false breakout)

תנאי כניסה:
1. נפח מעל סף (1.5x-2x ממוצע)
2. פריצת טווח/התנגדות/תמיכה
3. אישור תנע (momentum confirmation)
4. אישור נרות (candle confirmation)
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .base_strategy import (
    BaseStrategy, TradingSignal, SignalType, SignalStrength
)
from indicators.custom_indicators import TechnicalIndicators
from indicators.volume_analysis import VolumeAnalysis, VolumeBreakoutDetector


class VolumeBreakoutStrategy(BaseStrategy):
    """אסטרטגיית פריצות נפח"""
    
    def __init__(self, config: Dict):
        """
        אתחול אסטרטגיה
        
        Args:
            config: קונפיגורציה מקובץ trading_config.yaml
        """
        super().__init__(name="Volume_Breakout", config=config)
        
        # Strategy parameters (OPTIMIZED)
        self.volume_threshold = config.get('volume_threshold', 2.0)  # From 1.5 - stronger signal
        self.confirmation_candles = config.get('confirmation_candles', 2)  # From 3 - faster
        self.min_volume = config.get('min_volume', 100000)
        
        # Breakout detection parameters (OPTIMIZED)
        self.lookback_period = config.get('lookback_period', 15)  # From 20 - shorter range
        self.breakout_percent = config.get('breakout_percent', 1.0)  # From 0.5% - clearer break
        
        # Minimum price movement for valid breakout (OPTIMIZED)
        self.min_move_percent = config.get('price_change_threshold', 1.5)  # From 1.0% - bigger moves
        
        # Use multiple timeframes
        self.use_multi_timeframe = config.get('use_multi_timeframe', False)
        
        # Initialize indicators
        self.indicators = TechnicalIndicators()
        self.volume_analyzer = VolumeAnalysis()
        # Note: VolumeBreakoutDetector is a static class, will use its methods directly
        
    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ניתוח נתונים וזיהוי פריצות פוטנציאליות
        
        Args:
            data: DataFrame עם OHLCV
            
        Returns:
            DataFrame עם אינדיקטורים
        """
        if data is None or len(data) < self.lookback_period + 10:
            raise ValueError(f"Need at least {self.lookback_period + 10} bars for analysis")
        
        df = data.copy()
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['relative_volume'] = df['volume'] / df['volume_sma']
        
        # Volume indicators
        df['obv'] = VolumeAnalysis.obv(df)
        df['cmf'] = VolumeAnalysis.chaikin_money_flow(df, period=20)
        df['ad_line'] = VolumeAnalysis.accumulation_distribution(df)
        
        # Price range analysis
        df['high_20'] = df['high'].rolling(window=self.lookback_period).max()
        df['low_20'] = df['low'].rolling(window=self.lookback_period).min()
        df['range_20'] = df['high_20'] - df['low_20']
        
        # Breakout levels
        df['breakout_long'] = df['high_20'] * (1 + self.breakout_percent / 100)
        df['breakout_short'] = df['low_20'] * (1 - self.breakout_percent / 100)
        
        # Price position in range
        df['range_position'] = (
            (df['close'] - df['low_20']) / df['range_20']
        ) * 100
        
        # Momentum indicators
        df['rsi'] = TechnicalIndicators.rsi(df['close'], period=14)
        df['atr'] = TechnicalIndicators.atr(df, period=14)
        
        # Price rate of change
        df['roc'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
        
        # Volatility
        bb_result = TechnicalIndicators.bollinger_bands(df['close'], period=20, std=2)
        df['bb_upper'] = bb_result['upper']
        df['bb_middle'] = bb_result['middle']
        df['bb_lower'] = bb_result['lower']
        df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle']) * 100
        
        # Trend
        df['ema_20'] = TechnicalIndicators.ema(df['close'], period=20)
        df['ema_50'] = TechnicalIndicators.ema(df['close'], period=50)
        df['trend'] = np.where(df['ema_20'] > df['ema_50'], 'bullish', 'bearish')
        
        # Mark high volume bars
        df['high_volume'] = df['relative_volume'] > self.volume_threshold
        
        # Detect volume spikes
        df['volume_spike'] = VolumeAnalysis.detect_high_volume(
            df['volume'], threshold=self.volume_threshold
        )
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        זיהוי פריצות נפח ויצירת סיגנלים
        
        Args:
            data: DataFrame עם אינדיקטורים
            
        Returns:
            רשימת סיגנלים
        """
        signals = []
        
        if len(data) < self.confirmation_candles + 1:
            return signals
        
        # Use breakout detector to find breakouts
        breakout_signals = VolumeBreakoutDetector.detect_breakout(
            df=data,
            volume_threshold=self.volume_threshold,
            lookback_period=self.lookback_period
        )
        
        if breakout_signals is None or len(breakout_signals) == 0:
            return signals
        
        # Check recent breakouts (last few bars)
        for i in range(max(0, len(data) - self.confirmation_candles), len(data)):
            idx = data.index[i]
            breakout_value = breakout_signals.iloc[i]
            
            if breakout_value == 0:
                continue  # No breakout
            
            # Get current bar
            current = data.iloc[i]
            
            if breakout_value == 1:  # Bullish breakout
                signal = self._generate_long_signal(current, data, idx)
                if signal:
                    signals.append(signal)
                    
            elif breakout_value == -1:  # Bearish breakout
                signal = self._generate_short_signal(current, data, idx)
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _generate_long_signal(self, current: pd.Series, data: pd.DataFrame, 
                             idx) -> Optional[TradingSignal]:
        """
        יצירת סיגנל קנייה לפריצה שורית
        
        Args:
            current: הנר הנוכחי
            data: כל הנתונים
            idx: אינדקס הנר
            
        Returns:
            סיגנל או None
        """
        # Apply filters
        passed_filters = True
        filter_reasons = []
        
        current_price = current['close']
        current_volume = current.get('relative_volume', 1.0)
        
        # Filter 1: High volume confirmation
        if current_volume < self.volume_threshold:
            return None
        
        # Filter 2: Price actually broke above range
        if current_price <= current['high_20']:
            return None
        
        # Filter 3: Minimum price movement
        move_pct = ((current_price - current['low_20']) / current['low_20']) * 100
        if move_pct < self.min_move_percent:
            filter_reasons.append(f"Small move ({move_pct:.2f}%)")
            passed_filters = False
        
        # Filter 4: Not extremely overbought
        if current['rsi'] > 80:
            filter_reasons.append(f"RSI too high ({current['rsi']:.1f})")
            passed_filters = False
        
        # Filter 5: Positive momentum
        if current.get('roc', 0) < 0:
            filter_reasons.append("Negative momentum")
            passed_filters = False
        
        # Filter 6: Volume flow confirmation
        if current.get('cmf', 0) < -0.1:
            filter_reasons.append("Negative CMF (distribution)")
            # Not fatal, just reduces confidence
        
        if not passed_filters:
            return None
        
        # Calculate stop loss - below recent low
        stop_loss = current['low_20'] * 0.99  # 1% below range low
        
        # Use ATR-based stop if tighter
        atr_stop = self.calculate_stop_loss(
            current_price, SignalType.BUY, atr=current['atr']
        )
        stop_loss = max(stop_loss, atr_stop)  # Use tighter stop
        
        # Calculate take profit
        take_profit = self.calculate_take_profit(
            current_price, stop_loss, SignalType.BUY, risk_reward_ratio=3.0
        )
        
        # Calculate confidence
        confidence = self._calculate_breakout_confidence(
            current, data, 'bullish'
        )
        strength = self.get_signal_strength(confidence)
        
        signal = TradingSignal(
            timestamp=idx if hasattr(idx, 'to_pydatetime') else datetime.now(),
            symbol="",
            signal_type=SignalType.BUY,
            strength=strength,
            price=current_price,
            strategy_name=self.name,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            indicators={
                'relative_volume': current_volume,
                'breakout_level': current['breakout_long'],
                'range_high': current['high_20'],
                'range_low': current['low_20'],
                'rsi': current['rsi'],
                'cmf': current.get('cmf'),
                'roc': current.get('roc'),
                'atr': current['atr'],
                'trend': current['trend']
            },
            reason=f"Bullish volume breakout: {current_volume:.2f}x volume, broke above ${current['high_20']:.2f}",
            confidence=confidence
        )
        
        self.add_signal_to_history(signal)
        return signal
    
    def _generate_short_signal(self, current: pd.Series, data: pd.DataFrame,
                              idx) -> Optional[TradingSignal]:
        """
        יצירת סיגנל מכירה לפריצה דובית
        
        Args:
            current: הנר הנוכחי
            data: כל הנתונים
            idx: אינדקס הנר
            
        Returns:
            סיגנל או None
        """
        # Apply filters
        passed_filters = True
        filter_reasons = []
        
        current_price = current['close']
        current_volume = current.get('relative_volume', 1.0)
        
        # Filter 1: High volume confirmation
        if current_volume < self.volume_threshold:
            return None
        
        # Filter 2: Price actually broke below range
        if current_price >= current['low_20']:
            return None
        
        # Filter 3: Minimum price movement
        move_pct = ((current['high_20'] - current_price) / current['high_20']) * 100
        if move_pct < self.min_move_percent:
            filter_reasons.append(f"Small move ({move_pct:.2f}%)")
            passed_filters = False
        
        # Filter 4: Not extremely oversold
        if current['rsi'] < 20:
            filter_reasons.append(f"RSI too low ({current['rsi']:.1f})")
            passed_filters = False
        
        # Filter 5: Negative momentum
        if current.get('roc', 0) > 0:
            filter_reasons.append("Positive momentum")
            passed_filters = False
        
        # Filter 6: Volume flow confirmation
        if current.get('cmf', 0) > 0.1:
            filter_reasons.append("Positive CMF (accumulation)")
            # Not fatal, just reduces confidence
        
        if not passed_filters:
            return None
        
        # Calculate stop loss - above recent high
        stop_loss = current['high_20'] * 1.01  # 1% above range high
        
        # Use ATR-based stop if tighter
        atr_stop = self.calculate_stop_loss(
            current_price, SignalType.SELL, atr=current['atr']
        )
        stop_loss = min(stop_loss, atr_stop)  # Use tighter stop
        
        # Calculate take profit
        take_profit = self.calculate_take_profit(
            current_price, stop_loss, SignalType.SELL, risk_reward_ratio=3.0
        )
        
        # Calculate confidence
        confidence = self._calculate_breakout_confidence(
            current, data, 'bearish'
        )
        strength = self.get_signal_strength(confidence)
        
        signal = TradingSignal(
            timestamp=idx if hasattr(idx, 'to_pydatetime') else datetime.now(),
            symbol="",
            signal_type=SignalType.SELL,
            strength=strength,
            price=current_price,
            strategy_name=self.name,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            indicators={
                'relative_volume': current_volume,
                'breakout_level': current['breakout_short'],
                'range_high': current['high_20'],
                'range_low': current['low_20'],
                'rsi': current['rsi'],
                'cmf': current.get('cmf'),
                'roc': current.get('roc'),
                'atr': current['atr'],
                'trend': current['trend']
            },
            reason=f"Bearish volume breakout: {current_volume:.2f}x volume, broke below ${current['low_20']:.2f}",
            confidence=confidence
        )
        
        self.add_signal_to_history(signal)
        return signal
    
    def _calculate_breakout_confidence(self, current: pd.Series, 
                                       data: pd.DataFrame, 
                                       direction: str) -> float:
        """
        חישוב רמת ביטחון בפריצה
        
        Args:
            current: נר נוכחי
            data: כל הנתונים
            direction: 'bullish' או 'bearish'
            
        Returns:
            רמת ביטחון (0-1)
        """
        confidence = 0.5  # Base confidence
        
        if direction == 'bullish':
            # Very high volume
            if current.get('relative_volume', 1.0) > 2.0:
                confidence += 0.2
            elif current.get('relative_volume', 1.0) > 1.5:
                confidence += 0.1
            
            # Strong momentum
            if current.get('roc', 0) > 2.0:
                confidence += 0.15
            
            # Positive money flow
            if current.get('cmf', 0) > 0.1:
                confidence += 0.1
            
            # Trend alignment
            if current['trend'] == 'bullish':
                confidence += 0.05
            
        else:  # bearish
            # Very high volume
            if current.get('relative_volume', 1.0) > 2.0:
                confidence += 0.2
            elif current.get('relative_volume', 1.0) > 1.5:
                confidence += 0.1
            
            # Strong negative momentum
            if current.get('roc', 0) < -2.0:
                confidence += 0.15
            
            # Negative money flow
            if current.get('cmf', 0) < -0.1:
                confidence += 0.1
            
            # Trend alignment
            if current['trend'] == 'bearish':
                confidence += 0.05
        
        return min(confidence, 1.0)
    
    def __repr__(self):
        return (f"Volume Breakout Strategy (threshold: {self.volume_threshold}x) "
                f"- {self.enabled}")
