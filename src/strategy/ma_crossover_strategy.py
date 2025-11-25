import pandas as pd
import ta
from src.strategy.base_strategy import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    """
    A simple Moving Average Crossover strategy.
    """

    def __init__(self, name: str, short_window: int, long_window: int, rsi_period: int = 14, rsi_upper: int = 70, rsi_lower: int = 30):
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals based on the MA crossover logic with RSI filter.

        Args:
            data (pd.DataFrame): DataFrame with historical market data.

        Returns:
            pd.DataFrame: The input DataFrame with an added 'signal' column.
        """
        if f'SMA_{self.short_window}' not in data.columns:
            data[f'SMA_{self.short_window}'] = ta.trend.sma_indicator(data['close'], window=self.short_window)
        
        if f'SMA_{self.long_window}' not in data.columns:
            data[f'SMA_{self.long_window}'] = ta.trend.sma_indicator(data['close'], window=self.long_window)
            
        # Calculate RSI
        if 'RSI' not in data.columns:
            data['RSI'] = ta.momentum.rsi(data['close'], window=self.rsi_period)
        
        # Generate signals
        data['signal'] = 0
        
        # Buy Signal: Short MA > Long MA AND RSI < Upper Threshold (Not Overbought)
        buy_condition = (
            (data[f'SMA_{self.short_window}'] > data[f'SMA_{self.long_window}']) &
            (data['RSI'] < self.rsi_upper)
        )
        data.loc[buy_condition, 'signal'] = 1
        
        # Sell Signal: Short MA < Long MA AND RSI > Lower Threshold (Not Oversold)
        sell_condition = (
            (data[f'SMA_{self.short_window}'] < data[f'SMA_{self.long_window}']) &
            (data['RSI'] > self.rsi_lower)
        )
        data.loc[sell_condition, 'signal'] = -1
        
        return data
