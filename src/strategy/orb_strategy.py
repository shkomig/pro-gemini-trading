import pandas as pd
import numpy as np
from src.strategy.base_strategy import BaseStrategy

class ORBStrategy(BaseStrategy):
    """
    Opening Range Breakout (ORB) Strategy
    
    Trades breakouts of the first n-minutes trading range.
    """

    def __init__(self, name: str, opening_range_minutes: int = 30):
        super().__init__(name)
        self.opening_range_minutes = opening_range_minutes

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])

        data['signal'] = 0
        
        # Group by date to handle multiple days (if data spans multiple days)
        for date, group in data.groupby(data['date'].dt.date):
            if len(group) < 2:
                continue
            
            start_time = group['date'].min()
            end_range_time = start_time + pd.Timedelta(minutes=self.opening_range_minutes)
            
            opening_range = group[group['date'] <= end_range_time]
            
            if opening_range.empty:
                continue
                
            orb_high = opening_range['high'].max()
            orb_low = opening_range['low'].min()
            
            # Identify breakouts after the range
            post_opening = group[group['date'] > end_range_time]
            
            if post_opening.empty:
                continue

            # Vectorized signal generation for this group
            # Buy: Close > ORB High
            buy_indices = post_opening[post_opening['close'] > orb_high].index
            data.loc[buy_indices, 'signal'] = 1
            
            # Sell: Close < ORB Low
            sell_indices = post_opening[post_opening['close'] < orb_low].index
            data.loc[sell_indices, 'signal'] = -1
            
        return data
