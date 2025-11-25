import pandas as pd
import ta
from src.strategy.base_strategy import BaseStrategy

class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy - Mean Reversion
    
    Buys when price is oversold (at lower band).
    Sells when price is overbought (at upper band).
    """

    def __init__(self, name: str, period: int = 20, std_dev: float = 2.0):
        super().__init__(name)
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data

        # Initialize Indicator
        indicator_bb = ta.volatility.BollingerBands(close=data["close"], window=self.period, window_dev=self.std_dev)

        # Add Bollinger Bands features
        data["bb_high"] = indicator_bb.bollinger_hband()
        data["bb_low"] = indicator_bb.bollinger_lband()

        # Generate signals
        data['signal'] = 0
        
        # Buy Signal: Close price crosses below lower band (Oversold)
        data.loc[data['close'] < data['bb_low'], 'signal'] = 1
        
        # Sell Signal: Close price crosses above upper band (Overbought)
        data.loc[data['close'] > data['bb_high'], 'signal'] = -1
        
        return data
