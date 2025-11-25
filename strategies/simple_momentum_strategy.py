"""
Simple Momentum Strategy - Aggressive for Volatile Stocks
==========================================================
אסטרטגיה פשוטה ואגרסיבית למניות נדחפות

Strategy Logic:
- BUY: Price up > 0.5% in last 5 minutes + volume spike
- EXIT: Price down > 0.3% from entry OR profit target 1%

Works on specific volatile stocks only: MSTR, LCID
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
from .base_strategy import BaseStrategy, TradingSignal, SignalType, SignalStrength


class SimpleMomentumStrategy(BaseStrategy):
    """
    Simple Momentum Strategy - For Volatile Stocks
    
    Designed for high-volatility stocks with quick momentum moves.
    Uses short lookback and tight stops.
    """
    
    def __init__(self, config: Dict):
        super().__init__("SimpleMomentum", config)
        
        # Target stocks only
        self.target_symbols = config.get('target_symbols', ['MSTR', 'LCID'])
        
        # Momentum parameters (AGGRESSIVE)
        self.price_change_threshold = config.get('price_change_threshold', 0.5)  # 0.5% move
        self.lookback_minutes = config.get('lookback_minutes', 5)  # 5 minute lookback
        self.volume_spike_threshold = config.get('volume_spike_threshold', 1.2)  # 1.2x avg volume
        
        # Risk management (TIGHT)
        self.stop_loss_percent = config.get('stop_loss_percent', 0.3)  # 0.3% stop
        self.take_profit_percent = config.get('take_profit_percent', 1.0)  # 1% target
        
        print(f"[OK] Simple Momentum Strategy initialized:")
        print(f"  - Target Stocks: {self.target_symbols}")
        print(f"  - Trigger: {self.price_change_threshold}% price change in {self.lookback_minutes} min")
        print(f"  - Stop Loss: {self.stop_loss_percent}%")
        print(f"  - Take Profit: {self.take_profit_percent}%")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate simple momentum indicators
        """
        df = data.copy()
        
        # Price change % over lookback
        df['price_change_pct'] = ((df['close'] - df['close'].shift(self.lookback_minutes)) / 
                                   df['close'].shift(self.lookback_minutes) * 100)
        
        # Volume ratio
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Simple trend (5-period vs 20-period)
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['trend'] = np.where(df['ema_5'] > df['ema_20'], 'up', 'down')
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate simple momentum signals
        """
        signals = []
        
        if len(data) < self.lookback_minutes + 5:
            return signals
        
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Get current values
        current_price = current['close']
        price_change = current.get('price_change_pct', 0)
        volume_ratio = current.get('volume_ratio', 1.0)
        trend = current.get('trend', 'neutral')
        
        # Check if valid data
        if pd.isna(price_change) or pd.isna(volume_ratio):
            return signals
        
        # === LONG SIGNAL (Upward momentum + volume) ===
        if (price_change > self.price_change_threshold and 
            volume_ratio > self.volume_spike_threshold and
            trend == 'up'):
            
            # Calculate stop loss and take profit
            stop_loss = current_price * (1 - self.stop_loss_percent / 100)
            take_profit = current_price * (1 + self.take_profit_percent / 100)
            
            # High confidence for strong moves
            confidence = min(95, 70 + (price_change * 5))  # 70% base + 5% per 0.1%
            strength = SignalStrength.STRONG if confidence > 80 else SignalStrength.MODERATE
            
            timestamp = current.name.to_pydatetime() if hasattr(current.name, 'to_pydatetime') else datetime.now()
            signal = TradingSignal(
                timestamp=timestamp,
                symbol="",
                signal_type=SignalType.BUY,
                strength=strength,
                price=current_price,
                strategy_name=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators={
                    'price_change_pct': price_change,
                    'volume_ratio': volume_ratio,
                    'trend': trend,
                },
                reason=f"Strong momentum: +{price_change:.2f}% with {volume_ratio:.1f}x volume",
                confidence=confidence
            )
            signals.append(signal)
            self.add_signal_to_history(signal)
        
        # === EXIT SIGNAL (Momentum fading or reversal) ===
        elif (price_change < -self.stop_loss_percent or 
              trend == 'down'):
            
            confidence = 75
            
            timestamp = current.name.to_pydatetime() if hasattr(current.name, 'to_pydatetime') else datetime.now()
            signal = TradingSignal(
                timestamp=timestamp,
                symbol="",
                signal_type=SignalType.SELL,
                strength=SignalStrength.MODERATE,
                price=current_price,
                strategy_name=self.name,
                entry_price=current_price,
                stop_loss=None,
                take_profit=None,
                indicators={
                    'price_change_pct': price_change,
                    'trend': trend,
                },
                reason=f"Momentum fading: {price_change:.2f}% or trend down",
                confidence=confidence
            )
            signals.append(signal)
            self.add_signal_to_history(signal)
        
        return signals
    
    def should_analyze_symbol(self, symbol: str) -> bool:
        """
        Only analyze target symbols
        """
        return symbol in self.target_symbols
    
    def analyze(self, data: pd.DataFrame) -> Dict:
        """
        Main analysis method
        """
        df = self.calculate_indicators(data)
        signals = self.generate_signals(df)
        
        signal_type = None
        if signals:
            latest_signal = signals[-1]
            if latest_signal.signal_type == SignalType.BUY:
                signal_type = 'long'
            elif latest_signal.signal_type == SignalType.SELL:
                signal_type = 'exit'
        
        current = df.iloc[-1]
        
        return {
            'signal': signal_type,
            'price': current['close'],
            'indicators': {
                'price_change_pct': current.get('price_change_pct', 0),
                'volume_ratio': current.get('volume_ratio', 1.0),
                'trend': current.get('trend', 'neutral'),
            },
            'signals': signals
        }
