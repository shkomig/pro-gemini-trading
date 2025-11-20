import pandas as pd
import ta
from src.strategy.base_strategy import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    """
    A simple Moving Average Crossover strategy.
    """

    def __init__(self, name: str, short_window: int, long_window: int):
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals based on the MA crossover logic.

        Args:
            data (pd.DataFrame): DataFrame with historical market data.

        Returns:
            pd.DataFrame: The input DataFrame with an added 'signal' column.
        """
        if f'SMA_{self.short_window}' not in data.columns:
            data[f'SMA_{self.short_window}'] = ta.trend.sma_indicator(data['close'], window=self.short_window)
        
        if f'SMA_{self.long_window}' not in data.columns:
            data[f'SMA_{self.long_window}'] = ta.trend.sma_indicator(data['close'], window=self.long_window)
        
        # Generate signals
        data['signal'] = 0
        data.loc[data[f'SMA_{self.short_window}'] > data[f'SMA_{self.long_window}'], 'signal'] = 1
        data.loc[data[f'SMA_{self.short_window}'] < data[f'SMA_{self.long_window}'], 'signal'] = -1
        
        return data
