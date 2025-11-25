import pandas as pd
import numpy as np
from src.strategy.base_strategy import BaseStrategy

class VolumeBreakoutStrategy(BaseStrategy):
    """
    Volume Confirmed Breakout Strategy
    
    Detects breakouts above/below support/resistance with volume confirmation.
    """

    def __init__(self, name: str, lookback_period: int = 20, volume_spike_multiplier: float = 1.2):
        super().__init__(name)
        self.lookback_period = lookback_period
        self.volume_spike_multiplier = volume_spike_multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data

        # Calculate rolling metrics
        data['rolling_high'] = data['high'].rolling(window=self.lookback_period).max()
        data['rolling_low'] = data['low'].rolling(window=self.lookback_period).min()
        data['avg_volume'] = data['volume'].rolling(window=self.lookback_period).mean()

        data['signal'] = 0
        
        # Buy Signal: Close > Previous High AND Volume > Avg Volume * Multiplier
        # We use shift(1) for rolling high to compare against the *previous* period's high
        buy_condition = (
            (data['close'] > data['rolling_high'].shift(1)) & 
            (data['volume'] > data['avg_volume'] * self.volume_spike_multiplier)
        )
        data.loc[buy_condition, 'signal'] = 1
        
        # Sell Signal: Close < Previous Low AND Volume > Avg Volume * Multiplier
        sell_condition = (
            (data['close'] < data['rolling_low'].shift(1)) & 
            (data['volume'] > data['avg_volume'] * self.volume_spike_multiplier)
        )
        data.loc[sell_condition, 'signal'] = -1
        
        return data
