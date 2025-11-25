import pandas as pd
import os
from datetime import datetime
from src.logger.logger import log

class TradeLogger:
    def __init__(self, log_file="data/trade_history.csv"):
        self.log_file = log_file
        self.ensure_log_file_exists()

    def ensure_log_file_exists(self):
        if not os.path.exists(os.path.dirname(self.log_file)):
            os.makedirs(os.path.dirname(self.log_file))
        
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(columns=[
                "timestamp", "symbol", "strategy", "action", "price", 
                "quantity", "order_type", "tp_price", "sl_price"
            ])
            df.to_csv(self.log_file, index=False)

    def log_trade(self, symbol, strategy, action, price, quantity, order_type, tp_price=None, sl_price=None):
        try:
            new_entry = {
                "timestamp": datetime.now(),
                "symbol": symbol,
                "strategy": strategy,
                "action": action,
                "price": price,
                "quantity": quantity,
                "order_type": order_type,
                "tp_price": tp_price,
                "sl_price": sl_price
            }
            
            df = pd.DataFrame([new_entry])
            # Append to CSV without writing header
            df.to_csv(self.log_file, mode='a', header=False, index=False)
            log.info(f"Logged trade for {symbol} ({strategy}) to {self.log_file}")
        except Exception as e:
            log.error(f"Failed to log trade: {e}")
