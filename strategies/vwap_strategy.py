"""
VWAP Strategy
============
אסטרטגיית VWAP (Volume Weighted Average Price)

VWAP מייצג את המחיר הממוצע המשוקלל בנפח ומשמש כאינדיקטור למחיר "הוגן":
- מעל VWAP: מחיר חזק, קונים שולטים
- מתחת VWAP: מחיר חלש, מוכרים שולטים

אסטרטגיה:
- קנה: מחיר חוצה מעל VWAP עם נפח גבוה
- מכור: מחיר חוצה מתחת VWAP עם נפח גבוה

שימוש במרחק מ-VWAP כפילטר למניעת כניסות רחוקות מדי
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from .base_strategy import (
    BaseStrategy, TradingSignal, SignalType, SignalStrength
)
from indicators.custom_indicators import TechnicalIndicators


class VWAPStrategy(BaseStrategy):
    """אסטרטגיית VWAP"""
    
    def __init__(self, config: Dict):
        """
        אתחול אסטרטגיה
        
        Args:
            config: קונפיגורציה מקובץ trading_config.yaml
        """
        super().__init__(name="VWAP", config=config)
        
        # Strategy parameters
        self.deviation_percent = config.get('deviation_percent', 0.5)  # 0.5%
        self.min_volume = config.get('min_volume', 100000)
        self.volume_threshold = config.get('volume_threshold', 1.3)  # 1.3x average
        
        # Maximum distance from VWAP to enter (percentage)
        self.max_distance_percent = config.get('max_distance_percent', 2.0)  # 2%
        
        # Minimum distance to generate signal (avoid noise near VWAP)
        self.min_distance_percent = config.get('min_distance_percent', 0.2)  # 0.2%
        
        # Use standard deviation bands
        self.use_std_bands = config.get('use_std_bands', True)
        self.std_multiplier = config.get('std_multiplier', 2.0)
        
        # Initialize indicators
        self.indicators = TechnicalIndicators()
        
    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        חישוב VWAP ואינדיקטורים נלווים
        
        Args:
            data: DataFrame עם OHLCV
            
        Returns:
            DataFrame עם אינדיקטורים
        """
        if data is None or len(data) < 20:
            raise ValueError("Need at least 20 bars for VWAP analysis")
        
        df = data.copy()
        
        # Calculate VWAP
        df['vwap'] = TechnicalIndicators.vwap(df)
        
        # Calculate distance from VWAP
        df['vwap_distance'] = df['close'] - df['vwap']
        df['vwap_distance_pct'] = (df['vwap_distance'] / df['vwap']) * 100
        
        # VWAP standard deviation bands (if enabled)
        if self.use_std_bands:
            # Calculate cumulative variance for VWAP bands
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap_variance'] = (
                (typical_price - df['vwap']) ** 2 * df['volume']
            ).cumsum() / df['volume'].cumsum()
            df['vwap_std'] = np.sqrt(df['vwap_variance'])
            
            df['vwap_upper'] = df['vwap'] + (self.std_multiplier * df['vwap_std'])
            df['vwap_lower'] = df['vwap'] - (self.std_multiplier * df['vwap_std'])
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['relative_volume'] = df['volume'] / df['volume_sma']
        
        # Price position relative to VWAP
        df['above_vwap'] = df['close'] > df['vwap']
        df['below_vwap'] = df['close'] < df['vwap']
        
        # Calculate RSI for additional filtering
        df['rsi'] = TechnicalIndicators.rsi(df['close'], period=14)
        
        # ATR for stop loss
        df['atr'] = TechnicalIndicators.atr(df, period=14)
        
        # Trend determination using EMAs
        df['ema_20'] = TechnicalIndicators.ema(df['close'], period=20)
        df['ema_50'] = TechnicalIndicators.ema(df['close'], period=50)
        df['trend'] = np.where(df['ema_20'] > df['ema_50'], 'bullish', 'bearish')
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        זיהוי חציות VWAP ויצירת סיגנלים
        
        Args:
            data: DataFrame עם אינדיקטורים
            
        Returns:
            רשימת סיגנלים
        """
        signals = []
        
        if len(data) < 2:
            return signals
        
        # Get current and previous bars
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        current_price = current['close']
        current_vwap = current['vwap']
        prev_price = previous['close']
        prev_vwap = previous['vwap']
        
        current_volume = current.get('relative_volume', 1.0)
        current_atr = current['atr']
        current_rsi = current['rsi']
        
        # Calculate distance from VWAP
        distance_pct = abs(current['vwap_distance_pct'])
        
        # --- BULLISH CROSS: Price crossed above VWAP ---
        if prev_price <= prev_vwap and current_price > current_vwap:
            
            # Apply filters
            passed_filters = True
            filter_reasons = []
            
            # Filter 1: Distance from VWAP
            if distance_pct < self.min_distance_percent:
                passed_filters = False
                filter_reasons.append(f"Too close to VWAP ({distance_pct:.2f}%)")
            elif distance_pct > self.max_distance_percent:
                passed_filters = False
                filter_reasons.append(f"Too far from VWAP ({distance_pct:.2f}%)")
            
            # Filter 2: Volume confirmation
            if current_volume < self.volume_threshold:
                passed_filters = False
                filter_reasons.append(f"Low volume ({current_volume:.2f}x)")
            
            # Filter 3: Not overbought
            if current_rsi > 70:
                passed_filters = False
                filter_reasons.append(f"RSI overbought ({current_rsi:.1f})")
            
            # Filter 4: Trend alignment (prefer uptrend)
            if current['trend'] == 'bearish':
                # Still allow but reduce confidence
                pass
            
            if passed_filters:
                # Calculate stop loss - place below VWAP or use ATR
                stop_loss = min(
                    current_vwap * 0.995,  # 0.5% below VWAP
                    self.calculate_stop_loss(current_price, SignalType.BUY, atr=current_atr)
                )
                
                take_profit = self.calculate_take_profit(
                    current_price, stop_loss, SignalType.BUY, risk_reward_ratio=2.0
                )
                
                # Calculate confidence
                confidence = self._calculate_confidence(
                    current, previous, signal_type='buy'
                )
                strength = self.get_signal_strength(confidence)
                
                signal = TradingSignal(
                    timestamp=current.name if hasattr(current.name, 'to_pydatetime') else datetime.now(),
                    symbol="",
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    strategy_name=self.name,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    indicators={
                        'vwap': current_vwap,
                        'distance_pct': current['vwap_distance_pct'],
                        'relative_volume': current_volume,
                        'rsi': current_rsi,
                        'atr': current_atr,
                        'trend': current['trend']
                    },
                    reason=f"Bullish VWAP cross: Price crossed above VWAP ({distance_pct:.2f}% above)",
                    confidence=confidence
                )
                signals.append(signal)
                self.add_signal_to_history(signal)
        
        # --- BEARISH CROSS: Price crossed below VWAP ---
        elif prev_price >= prev_vwap and current_price < current_vwap:
            
            # Apply filters
            passed_filters = True
            filter_reasons = []
            
            # Filter 1: Distance from VWAP
            if distance_pct < self.min_distance_percent:
                passed_filters = False
                filter_reasons.append(f"Too close to VWAP ({distance_pct:.2f}%)")
            elif distance_pct > self.max_distance_percent:
                passed_filters = False
                filter_reasons.append(f"Too far from VWAP ({distance_pct:.2f}%)")
            
            # Filter 2: Volume confirmation
            if current_volume < self.volume_threshold:
                passed_filters = False
                filter_reasons.append(f"Low volume ({current_volume:.2f}x)")
            
            # Filter 3: Not oversold
            if current_rsi < 30:
                passed_filters = False
                filter_reasons.append(f"RSI oversold ({current_rsi:.1f})")
            
            # Filter 4: Trend alignment (prefer downtrend)
            if current['trend'] == 'bullish':
                # Still allow but reduce confidence
                pass
            
            if passed_filters:
                # Calculate stop loss - place above VWAP or use ATR
                stop_loss = max(
                    current_vwap * 1.005,  # 0.5% above VWAP
                    self.calculate_stop_loss(current_price, SignalType.SELL, atr=current_atr)
                )
                
                take_profit = self.calculate_take_profit(
                    current_price, stop_loss, SignalType.SELL, risk_reward_ratio=2.0
                )
                
                confidence = self._calculate_confidence(
                    current, previous, signal_type='sell'
                )
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
                        'vwap': current_vwap,
                        'distance_pct': current['vwap_distance_pct'],
                        'relative_volume': current_volume,
                        'rsi': current_rsi,
                        'atr': current_atr,
                        'trend': current['trend']
                    },
                    reason=f"Bearish VWAP cross: Price crossed below VWAP ({distance_pct:.2f}% below)",
                    confidence=confidence
                )
                signals.append(signal)
                self.add_signal_to_history(signal)
        
        # --- MEAN REVERSION: Price at VWAP bands ---
        # Optional: Generate signals when price touches bands and reverses
        if self.use_std_bands and len(data) >= 3:
            self._check_band_reversal(data, signals)
        
        return signals
    
    def _check_band_reversal(self, data: pd.DataFrame, signals: List[TradingSignal]):
        """
        בדיקת היפוך ממחזור בנד VWAP (mean reversion)
        
        Args:
            data: DataFrame עם נתונים
            signals: רשימת סיגנלים להוספה
        """
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        if 'vwap_upper' not in current or 'vwap_lower' not in current:
            return
        
        current_price = current['close']
        prev_price = previous['close']
        
        # Reversal from upper band (overbought)
        if (prev_price >= current['vwap_upper'] and 
            current_price < current['vwap_upper'] and
            current['rsi'] > 60):
            
            # Potential mean reversion sell signal
            # (Not implemented fully - would need additional confirmation)
            pass
        
        # Reversal from lower band (oversold)
        elif (prev_price <= current['vwap_lower'] and 
              current_price > current['vwap_lower'] and
              current['rsi'] < 40):
            
            # Potential mean reversion buy signal
            # (Not implemented fully - would need additional confirmation)
            pass
    
    def _calculate_confidence(self, current: pd.Series, previous: pd.Series, 
                             signal_type: str) -> float:
        """
        חישוב רמת ביטחון בסיגנל
        
        Args:
            current: נר נוכחי
            previous: נר קודם
            signal_type: 'buy' או 'sell'
            
        Returns:
            רמת ביטחון (0-1)
        """
        confidence = 0.5  # Base confidence
        
        distance_pct = abs(current['vwap_distance_pct'])
        
        if signal_type == 'buy':
            # Trend alignment
            if current['trend'] == 'bullish':
                confidence += 0.15
            
            # Strong volume
            if current.get('relative_volume', 1.0) > 1.5:
                confidence += 0.15
            
            # Good distance from VWAP (not too close, not too far)
            if 0.3 <= distance_pct <= 1.0:
                confidence += 0.1
            
            # RSI in bullish range
            if 45 < current['rsi'] < 65:
                confidence += 0.1
            
        else:  # sell
            # Trend alignment
            if current['trend'] == 'bearish':
                confidence += 0.15
            
            # Strong volume
            if current.get('relative_volume', 1.0) > 1.5:
                confidence += 0.15
            
            # Good distance from VWAP
            if 0.3 <= distance_pct <= 1.0:
                confidence += 0.1
            
            # RSI in bearish range
            if 35 < current['rsi'] < 55:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def get_vwap_stats(self, data: pd.DataFrame) -> Dict:
        """
        קבלת סטטיסטיקות VWAP
        
        Returns:
            מילון עם נתוני VWAP
        """
        if len(data) == 0:
            return {}
        
        current = data.iloc[-1]
        
        stats = {
            'current_price': current['close'],
            'vwap': current['vwap'],
            'distance': current['vwap_distance'],
            'distance_pct': current['vwap_distance_pct'],
            'above_vwap': current['above_vwap'],
            'relative_volume': current.get('relative_volume', 1.0)
        }
        
        if self.use_std_bands:
            stats.update({
                'vwap_upper': current.get('vwap_upper'),
                'vwap_lower': current.get('vwap_lower'),
                'vwap_std': current.get('vwap_std')
            })
        
        return stats
    
    def __repr__(self):
        return f"VWAP Strategy (deviation: {self.deviation_percent}%) - {self.enabled}"
