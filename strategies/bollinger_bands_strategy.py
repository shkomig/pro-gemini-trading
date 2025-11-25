"""
Bollinger Bands Strategy
========================
Mean reversion strategy using Bollinger Bands.

Strategy Logic:
- BUY: Price touches or breaks below lower band (oversold)
- SELL: Price touches or breaks above upper band (overbought)
- Bands: 20-period SMA ± 2 standard deviations

Classic Technical Analysis:
- Lower band (Mean - 2σ): Support, potential buy
- Upper band (Mean + 2σ): Resistance, potential sell/exit
- Works best in ranging markets
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_strategy import BaseStrategy, TradingSignal, SignalType, SignalStrength


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy - Mean Reversion
    
    Buys when price is oversold (at lower band).
    Exits when price reaches middle band or upper band.
    """
    
    def __init__(self, config: Dict):
        super().__init__("BollingerBands", config)
        
        # Bollinger Bands parameters
        self.period = config.get('period', 20)  # SMA period
        self.num_std = config.get('num_std', 2.0)  # Standard deviations
        
        # Entry/Exit parameters
        self.entry_threshold = config.get('entry_threshold', 0.0)  # % below lower band
        self.exit_at_middle = config.get('exit_at_middle', True)  # Exit at middle band
        
        # Risk management
        self.stop_loss_percent = config.get('stop_loss_percent', 3.0)
        self.take_profit_at_upper = config.get('take_profit_at_upper', True)
        
        # Filters
        self.volume_filter = config.get('volume_filter', True)
        self.min_volume = config.get('min_volume', 100000)
        self.rsi_confirmation = config.get('rsi_confirmation', True)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        
        # Bandwidth filter (avoid tight squeezes)
        self.min_bandwidth_percent = config.get('min_bandwidth_percent', 2.0)
        
        print(f"[OK] Bollinger Bands Strategy initialized:")
        print(f"  - Period: {self.period}, Std Dev: {self.num_std}")
        print(f"  - Exit at middle: {self.exit_at_middle}")
        print(f"  - RSI confirmation: {self.rsi_confirmation}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands and related indicators
        """
        df = data.copy()
        
        # Calculate SMA (middle band)
        df['bb_middle'] = df['close'].rolling(window=self.period).mean()
        
        # Calculate standard deviation
        df['bb_std'] = df['close'].rolling(window=self.period).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (self.num_std * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (self.num_std * df['bb_std'])
        
        # Calculate %B (position within bands)
        # %B = (close - lower) / (upper - lower)
        # %B > 1: above upper band, %B < 0: below lower band, %B = 0.5: at middle
        df['percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Calculate Bandwidth (volatility measure)
        df['bb_bandwidth'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100)
        
        # RSI for confirmation
        if self.rsi_confirmation:
            df['rsi'] = self._calculate_rsi(df['close'], period=14)
        
        # Volume indicators
        if self.volume_filter:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """
        Generate Bollinger Bands signals
        """
        if len(data) < self.period + 2:
            return None
        
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Check if we have valid data
        if pd.isna(current['bb_upper']) or pd.isna(current['bb_lower']):
            return None
        
        # Bandwidth filter (avoid tight squeezes with low volatility)
        if current['bb_bandwidth'] < self.min_bandwidth_percent:
            return None
        
        # Volume filter
        if self.volume_filter:
            if current['volume'] < self.min_volume:
                return None
            if 'volume_ratio' in current and current['volume_ratio'] < 0.8:
                return None
        
        # === LONG SIGNAL (Price at or below lower band) ===
        if current['percent_b'] <= 0.0:  # At or below lower band
            # RSI confirmation (oversold)
            if self.rsi_confirmation:
                if 'rsi' in current and not pd.isna(current['rsi']):
                    if current['rsi'] > self.rsi_oversold:
                        return None  # Not oversold enough
                else:
                    return None
            
            return 'long'
        
        # === EXIT SIGNAL (Price at middle or upper band) ===
        elif current['percent_b'] >= 0.5:  # At or above middle band
            if self.exit_at_middle:
                return 'exit'
            elif current['percent_b'] >= 1.0:  # At or above upper band
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
            
            # Stop loss: percentage based or below lower band
            stop_loss = min(
                entry_price * (1 - self.stop_loss_percent / 100),
                current['bb_lower'] * 0.98  # 2% below lower band
            )
            
            # Take profit: middle band or upper band
            if self.take_profit_at_upper:
                take_profit = current['bb_upper']
            else:
                take_profit = current['bb_middle']
            
            # Calculate risk/reward
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
            risk_reward = reward / risk if risk > 0 else 0
            
            # Distance below lower band (% below)
            distance_below = ((current['bb_lower'] - entry_price) / entry_price * 100)
            
            # Confidence based on:
            # 1. How far below lower band (more extreme = higher confidence)
            # 2. RSI oversold level
            # 3. Volume
            confidence = 0.5
            
            if distance_below > 1.0:  # More than 1% below lower band
                confidence += 0.2
            
            if self.rsi_confirmation and 'rsi' in current:
                if current['rsi'] < 25:  # Very oversold
                    confidence += 0.2
                elif current['rsi'] < self.rsi_oversold:
                    confidence += 0.1
            
            if self.volume_filter and 'volume_ratio' in current:
                if current['volume_ratio'] > 1.5:  # High volume
                    confidence += 0.1
            
            confidence = min(confidence, 1.0)
            
            return {
                'signal': signal,
                'price': entry_price,
                'bb_upper': current['bb_upper'],
                'bb_middle': current['bb_middle'],
                'bb_lower': current['bb_lower'],
                'percent_b': current['percent_b'],
                'bandwidth': current['bb_bandwidth'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk': risk,
                'reward': reward,
                'risk_reward': risk_reward,
                'rsi': current['rsi'] if 'rsi' in current else None,
                'confidence': confidence,
                'reason': self._get_reason(signal, current)
            }
        
        elif signal == 'exit':
            return {
                'signal': 'exit',
                'price': current['close'],
                'percent_b': current['percent_b'],
                'reason': self._get_reason(signal, current)
            }
        
        return {'signal': None}
    
    def _get_reason(self, signal: str, current) -> str:
        """Generate human-readable reason"""
        if signal == 'long':
            percent_b = current['percent_b']
            rsi_str = f", RSI: {current['rsi']:.1f}" if 'rsi' in current and not pd.isna(current['rsi']) else ""
            return f"Price below lower band (%B: {percent_b:.3f}{rsi_str})"
        elif signal == 'exit':
            percent_b = current['percent_b']
            if percent_b >= 1.0:
                return f"Price at upper band (%B: {percent_b:.3f})"
            else:
                return f"Price at middle band (%B: {percent_b:.3f})"
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
                'bb_upper': analysis.get('bb_upper'),
                'bb_middle': analysis.get('bb_middle'),
                'bb_lower': analysis.get('bb_lower'),
                'percent_b': analysis.get('percent_b'),
                'bandwidth': analysis.get('bandwidth'),
                'rsi': analysis.get('rsi')
            },
            reason=analysis['reason'],
            confidence=confidence
        )
        
        return [trading_signal]
