import time
from ib_insync import IB, Stock, MarketOrder, util
from loguru import logger
import pandas as pd
import numpy as np

class IBClient:
    def __init__(self, host: str, port: int, client_id: int, max_retries: int = 3, retry_delay: int = 5, simulation_mode: bool = False):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.simulation_mode = simulation_mode

    def connect(self):
        """Establishes connection to TWS or IB Gateway with retries."""
        if self.simulation_mode:
            logger.info("Simulation mode is active. Skipping real connection.")
            logger.success("Simulated connection successful.")
            return

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Connecting to IB Gateway at {self.host}:{self.port} with client ID {self.client_id} (Attempt {attempt + 1}/{self.max_retries})...")
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                logger.success("Successfully connected to IB Gateway.")
                return
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
        logger.error("Failed to connect to IB Gateway after multiple retries.")
        raise ConnectionError("Could not connect to IB Gateway.")

    def disconnect(self):
        """Disconnects from TWS or IB Gateway."""
        if self.simulation_mode:
            logger.info("Simulation mode is active. Skipping real disconnection.")
            logger.success("Simulated disconnection successful.")
            return
            
        logger.info("Disconnecting from IB Gateway...")
        self.ib.disconnect()
        logger.success("Successfully disconnected from IB Gateway.")

    def get_account_summary(self):
        """Retrieves account summary."""
        if self.simulation_mode:
            logger.info("Simulation mode: Returning dummy account summary.")
            return {}
        return self.ib.accountSummary()

    def get_historical_data(self, symbol: str, duration: str, bar_size: str) -> pd.DataFrame:
        """Requests historical data for a given contract."""
        if self.simulation_mode:
            logger.info(f"Simulation mode: Generating dummy historical data for {symbol}.")
            days = 365 # Default to 1 year of daily data
            if duration.endswith('Y'):
                days = int(duration.split(' ')[0]) * 365
            elif duration.endswith('M'):
                days = int(duration.split(' ')[0]) * 30
            elif duration.endswith('W'):
                days = int(duration.split(' ')[0]) * 7
            elif duration.endswith('D'):
                days = int(duration.split(' ')[0])
                
            dates = pd.date_range(start="2023-01-01", periods=days, freq="D")
            prices = 100 + np.random.randn(days).cumsum()
            df = pd.DataFrame({'date': dates, 'close': prices})
            # df.set_index('date', inplace=True)
            return df

        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.reqHeadTimeStamp(contract, whatToShow='TRADES', useRTH=True)

        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES', 
            useRTH=True,
            keepUpToDate=False
        )

        if not bars:
            logger.warning(f"No historical data returned for {symbol}.")
            return pd.DataFrame()

        # Convert to pandas DataFrame
        df = util.df(bars)
        return df

    def place_order(self, symbol: str, quantity: float, action: str):
        """Places a market order."""
        if self.simulation_mode:
            logger.info(f"Simulation mode: Pretending to place order for {quantity} {symbol} {action}.")
            return None # Or a dummy trade object

        contract = Stock(symbol, 'SMART', 'USD')
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)
        logger.info(f"Placed order: {trade}")
        return trade

    def get_open_trades(self):
        """Retrieves open trades."""
        if self.simulation_mode:
            logger.info("Simulation mode: Returning no open trades.")
            return []
        return self.ib.openTrades()
