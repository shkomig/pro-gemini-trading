import time
from ib_insync import IB, Stock, Forex, Future, MarketOrder, LimitOrder, StopOrder, util
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
        self.sec_type = 'STK' # Default
        self.exchange = 'SMART' # Default
        self.currency = 'USD' # Default

    def set_contract_details(self, sec_type: str, exchange: str, currency: str):
        self.sec_type = sec_type
        self.exchange = exchange
        self.currency = currency

    def _create_contract(self, symbol: str):
        if self.sec_type == 'STK':
            return Stock(symbol, self.exchange, self.currency)
        elif self.sec_type == 'CASH':
            return Forex(symbol)
        elif self.sec_type == 'FUT':
            return Future(symbol, '202412', self.exchange) # Simplified for example, usually needs expiry
        else:
            return Stock(symbol, self.exchange, self.currency)

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
                
                # Explicitly set Market Data Type to 1 (Real-Time) to use the shared subscription
                self.ib.reqMarketDataType(1)
                logger.info("Set Market Data Type to 1 (Real-Time).")
                
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

        contract = self._create_contract(symbol)
        self.ib.reqHeadTimeStamp(contract, whatToShow='TRADES', useRTH=True)

        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='MIDPOINT' if self.sec_type == 'CASH' else 'TRADES', 
            useRTH=False,
            keepUpToDate=False
        )

        if not bars:
            logger.warning(f"No historical data returned for {symbol}.")
            return pd.DataFrame()

        # Convert to pandas DataFrame
        df = util.df(bars)
        return df

    def get_current_price(self, symbol: str) -> float:
        """Fetches the current market price using streaming data."""
        if self.simulation_mode:
            return 100.0

        contract = self._create_contract(symbol)
        # Request streaming data (snapshot=False) to utilize the Streaming Bundle
        ticker = self.ib.reqMktData(contract, '', False, False)
        
        start_time = time.time()
        # Wait for a valid bid, ask, or last price
        while (ticker.last != ticker.last and ticker.bid != ticker.bid and ticker.ask != ticker.ask) and (time.time() - start_time) < 4:
             self.ib.sleep(0.1)
        
        # Prefer Last price, then Midpoint (Bid/Ask average)
        price = 0.0
        if not np.isnan(ticker.last):
            price = ticker.last
        elif not np.isnan(ticker.bid) and not np.isnan(ticker.ask):
            price = (ticker.bid + ticker.ask) / 2
        else:
             # If still no data, try close
             if not np.isnan(ticker.close):
                 price = ticker.close
        
        # Cancel the subscription to save resources
        self.ib.cancelMktData(contract)

        if price == 0.0:
             logger.warning(f"Could not get market price for {symbol}")
             
        return price

    def place_order(self, symbol: str, quantity: float, action: str, limit_price: float = None):
        """Places an order. Uses LimitOrder if limit_price is provided, else MarketOrder."""
        if self.simulation_mode:
            logger.info(f"Simulation mode: Pretending to place order for {quantity} {symbol} {action} at {limit_price}.")
            return None 

        contract = self._create_contract(symbol)
        if limit_price:
            order = LimitOrder(action, quantity, limit_price)
        else:
            order = MarketOrder(action, quantity)
            
        order.outsideRth = True
        trade = self.ib.placeOrder(contract, order)
        logger.info(f"Placed order: {trade}")
        return trade

    def place_bracket_order(self, symbol: str, quantity: float, action: str, limit_price: float, take_profit_price: float, stop_loss_price: float):
        """Places a bracket order (Entry + Take Profit + Stop Loss)."""
        if self.simulation_mode:
            logger.info(f"Simulation mode: Placing bracket order for {symbol}. Entry: {limit_price}, TP: {take_profit_price}, SL: {stop_loss_price}")
            return []

        contract = self._create_contract(symbol)
        
        # Parent Order (Entry)
        parent = LimitOrder(action, quantity, limit_price)
        parent.orderId = self.ib.client.getReqId()
        parent.transmit = False
        parent.outsideRth = True

        # Take Profit Order
        tp_action = "SELL" if action == "BUY" else "BUY"
        take_profit = LimitOrder(tp_action, quantity, take_profit_price)
        take_profit.parentId = parent.orderId
        take_profit.transmit = False
        take_profit.outsideRth = True

        # Stop Loss Order
        stop_loss = StopOrder(tp_action, quantity, stop_loss_price)
        stop_loss.parentId = parent.orderId
        stop_loss.transmit = True # Transmit the whole bracket
        stop_loss.outsideRth = True

        orders = [parent, take_profit, stop_loss]
        trades = [self.ib.placeOrder(contract, o) for o in orders]
        
        logger.info(f"Placed bracket order for {symbol}. Parent ID: {parent.orderId}")
        return trades

    def get_open_trades(self):
        """Retrieves open trades."""
        if self.simulation_mode:
            logger.info("Simulation mode: Returning no open trades.")
            return []
        return self.ib.openTrades()

    def get_position(self, symbol: str) -> float:
        """Retrieves the current position for a given symbol."""
        if self.simulation_mode:
            return 0.0
        
        # V3.1: Force refresh positions from IB Gateway
        self.ib.reqPositions()
        self.ib.sleep(0.5)  # Wait for position update
        
        positions = self.ib.positions()
        for position in positions:
            if position.contract.symbol == symbol:
                return position.position
        return 0.0

    def get_all_positions(self):
        """Retrieves all open positions."""
        if self.simulation_mode:
            return []
        
        # V3.1: Force refresh positions from IB Gateway
        self.ib.reqPositions()
        self.ib.sleep(0.5)  # Wait for position update
        
        return self.ib.positions()

    def has_open_order(self, symbol: str) -> bool:
        """Checks if there are any open orders for the given symbol."""
        if self.simulation_mode:
            return False
            
        trades = self.ib.openTrades()
        for trade in trades:
            if trade.contract.symbol == symbol:
                return True
        return False
