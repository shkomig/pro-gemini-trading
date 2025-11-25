import pandas as pd
import ta
from src.strategy.base_strategy import BaseStrategy

class VWAPStrategy(BaseStrategy):
    """
    VWAP Strategy
    
    Buy: Price crosses above VWAP
    Sell: Price crosses below VWAP
    """

    def __init__(self, name: str, window: int = 14):
        super().__init__(name)
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data

        # Calculate VWAP using ta library
        # Note: True VWAP resets daily. This uses a rolling window approximation if 'window' is set,
        # or we can implement a cumulative one. For simplicity and robustness with 'ta', we use the library.
        vwap = ta.volume.VolumeWeightedAveragePrice(
            high=data['high'], low=data['low'], close=data["close"], volume=data['volume'], window=self.window
        )
        data["vwap"] = vwap.volume_weighted_average_price()

        data['signal'] = 0
        
        # Buy: Price crosses above VWAP
        data.loc[data['close'] > data['vwap'], 'signal'] = 1
        
        # Sell: Price crosses below VWAP
        data.loc[data['close'] < data['vwap'], 'signal'] = -1
        
        return data
