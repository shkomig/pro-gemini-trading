from src.logger.logger import log
from src.ib_client.ib_client import IBClient

class TradeManager:
    def __init__(self, ib_client: IBClient):
        self.ib_client = ib_client

    def process_signals(self, signals_df, symbol: str):
        """Processes trading signals and places orders."""
        if signals_df.empty:
            log.warning("Received empty signals DataFrame. No action taken.")
            return

        latest_signal = signals_df.iloc[-1]
        signal = latest_signal['signal']

        log.info(f"Processing signal for {symbol}. Latest signal: {signal}")

        # This is a simplified logic. A real system would manage positions, risk, etc.
        if signal == 1:
            log.info(f"Buy signal detected for {symbol}. Placing market order.")
            # In a real scenario, you'd check if you already have a position.
            self.ib_client.place_order(symbol=symbol, quantity=10, action="BUY") # Using a fixed quantity for now
        elif signal == -1:
            log.info(f"Sell signal detected for {symbol}. Placing market order.")
            # In a real scenario, you'd check if you have a position to sell.
            self.ib_client.place_order(symbol=symbol, quantity=10, action="SELL") # Using a fixed quantity for now
        else:
            log.info(f"Neutral signal (0) for {symbol}. No action taken.")
