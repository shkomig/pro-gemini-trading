# מדריך מקורות נתונים ו-APIs למסחר

## 1. APIs מומלצים לנתוני שוק

### 1.1 Alpha Vantage

**סקירה כללית:**
Alpha Vantage הוא ספק נתוני שוק רשמי של NASDAQ, מציע APIs חינמיים ובתשלום לנתונים פיננסיים.

**מה כולל:**
- נתוני מניות real-time והיסטוריים
- מט"ח (Forex)
- קריפטו
- אינדיקטורים טכניים מובנים
- נתונים פונדמנטליים

**Free Tier:**
- 5 בקשות לדקה
- 500 בקשות ליום

**מחיר Premium:**
- $49.99/חודש - 30 calls/minute
- $99.99/חודש - 75 calls/minute
- $249.99/חודש - 150+ calls/minute

**קוד דוגמה:**
```python
import requests
import pandas as pd

class AlphaVantageAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_daily_data(self, symbol, outputsize='full'):
        """קבלת נתונים יומיים"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['Time Series (Daily)']).T
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    
    def get_intraday_data(self, symbol, interval='5min'):
        """נתונים תוך-יומיים"""
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'outputsize': 'full',
            'apikey': self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return self._parse_time_series(response.json())
    
    def get_technical_indicator(self, symbol, indicator, interval='daily', 
                                 time_period=20):
        """אינדיקטור טכני"""
        params = {
            'function': indicator,  # e.g., 'RSI', 'MACD', 'SMA'
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'series_type': 'close',
            'apikey': self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    def get_fundamentals(self, symbol):
        """נתונים פונדמנטליים"""
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()

# Usage
api = AlphaVantageAPI('YOUR_API_KEY')
df = api.get_daily_data('AAPL')
print(df.head())
```

---

### 1.2 Finnhub

**סקירה כללית:**
Finnhub מספק נתוני שוק real-time עם תמיכה בניתוח סנטימנט מבוסס AI.

**מה כולל:**
- נתוני מניות ומט"ח real-time
- אינדיקטורים כלכליים
- דוחות רווחים
- עסקאות Insider
- ניתוח סנטימנט AI

**Free Tier:**
- 60 בקשות לדקה
- מרבית הנתונים הבסיסיים

**Premium:**
- מ-$79/חודש
- נתונים מתקדמים יותר

**קוד דוגמה:**
```python
import finnhub
import pandas as pd
from datetime import datetime, timedelta

class FinnhubAPI:
    def __init__(self, api_key):
        self.client = finnhub.Client(api_key=api_key)
    
    def get_quote(self, symbol):
        """מחיר נוכחי"""
        return self.client.quote(symbol)
    
    def get_candles(self, symbol, resolution='D', days_back=365):
        """נתוני נרות"""
        end = int(datetime.now().timestamp())
        start = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        data = self.client.stock_candles(symbol, resolution, start, end)
        
        df = pd.DataFrame({
            'Open': data['o'],
            'High': data['h'],
            'Low': data['l'],
            'Close': data['c'],
            'Volume': data['v'],
            'Timestamp': pd.to_datetime(data['t'], unit='s')
        })
        df.set_index('Timestamp', inplace=True)
        return df
    
    def get_company_news(self, symbol, days_back=7):
        """חדשות חברה"""
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        news = self.client.company_news(symbol, start, end)
        return pd.DataFrame(news)
    
    def get_sentiment(self, symbol):
        """ניתוח סנטימנט"""
        return self.client.news_sentiment(symbol)
    
    def get_insider_transactions(self, symbol):
        """עסקאות Insider"""
        return self.client.stock_insider_transactions(symbol)
    
    def get_earnings_calendar(self, symbol):
        """לוח דוחות רווחים"""
        return self.client.earnings_calendar()
    
    def get_recommendation_trends(self, symbol):
        """המלצות אנליסטים"""
        return self.client.recommendation_trends(symbol)

# Usage
api = FinnhubAPI('YOUR_API_KEY')
quote = api.get_quote('AAPL')
print(f"AAPL Current Price: ${quote['c']}")
```

---

### 1.3 Alpaca Markets

**סקירה כללית:**
Alpaca מציעה API למסחר ונתוני שוק, עם דגש על אלגו-טריידינג.

**מה כולל:**
- מסחר אוטומטי (Paper + Live)
- נתוני מניות real-time
- אופציות
- קריפטו
- 7+ שנות נתונים היסטוריים

**תמחור:**
- **Free:** 200 API calls/minute, נתונים מ-IEX בלבד
- **Unlimited:** $99/month - unlimited calls, כל הבורסות

**קוד דוגמה:**
```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime, timedelta

class AlpacaTrader:
    def __init__(self, api_key, secret_key, paper=True):
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
    
    def get_historical_data(self, symbol, days=365, timeframe='day'):
        """נתונים היסטוריים"""
        tf_map = {
            'minute': TimeFrame.Minute,
            'hour': TimeFrame.Hour,
            'day': TimeFrame.Day
        }
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf_map.get(timeframe, TimeFrame.Day),
            start=datetime.now() - timedelta(days=days),
            end=datetime.now()
        )
        
        bars = self.data_client.get_stock_bars(request_params)
        df = bars.df
        return df
    
    def get_account(self):
        """מידע על החשבון"""
        account = self.trading_client.get_account()
        return {
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'buying_power': float(account.buying_power),
            'equity': float(account.equity)
        }
    
    def get_positions(self):
        """פוזיציות פתוחות"""
        positions = self.trading_client.get_all_positions()
        return [{
            'symbol': p.symbol,
            'qty': float(p.qty),
            'market_value': float(p.market_value),
            'unrealized_pl': float(p.unrealized_pl),
            'unrealized_plpc': float(p.unrealized_plpc)
        } for p in positions]
    
    def place_market_order(self, symbol, qty, side='buy'):
        """ביצוע פקודת שוק"""
        order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
        
        market_order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY
        )
        
        order = self.trading_client.submit_order(market_order)
        return order
    
    def place_limit_order(self, symbol, qty, limit_price, side='buy'):
        """ביצוע פקודת לימיט"""
        from alpaca.trading.requests import LimitOrderRequest
        
        order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
        
        limit_order = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price
        )
        
        order = self.trading_client.submit_order(limit_order)
        return order
    
    def place_bracket_order(self, symbol, qty, take_profit, stop_loss):
        """פקודה עם TP ו-SL"""
        from alpaca.trading.requests import (
            MarketOrderRequest, TakeProfitRequest, StopLossRequest
        )
        
        bracket_order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class='bracket',
            take_profit=TakeProfitRequest(limit_price=take_profit),
            stop_loss=StopLossRequest(stop_price=stop_loss)
        )
        
        return self.trading_client.submit_order(bracket_order)

# Usage
trader = AlpacaTrader('API_KEY', 'SECRET_KEY', paper=True)
print(trader.get_account())
df = trader.get_historical_data('AAPL', days=30)
print(df.tail())
```

---

### 1.4 Interactive Brokers (TWS API)

**סקירה כללית:**
Interactive Brokers מספקת API מקצועי למסחר ב-150+ שווקים גלובליים.

**מה כולל:**
- מסחר גלובלי (מניות, אופציות, futures, forex, bonds)
- נתוני שוק real-time
- נתונים היסטוריים
- ניתוח פורטפוליו

**דרישות:**
- חשבון IB פעיל
- TWS או IB Gateway מותקן
- מינימום $10,000 להפעלת API

**קוד דוגמה:**
```python
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import threading
import time
import pandas as pd

class IBClient(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}
        self.positions = []
        self.order_id = 0
        
    def error(self, reqId, errorCode, errorString, advancedOrderReject=None):
        if errorCode not in [2104, 2106, 2158]:  # Filter info messages
            print(f"Error {errorCode}: {errorString}")
    
    def nextValidId(self, orderId):
        self.order_id = orderId
        print(f"Next valid order ID: {orderId}")
    
    def historicalData(self, reqId, bar):
        if reqId not in self.data:
            self.data[reqId] = []
        self.data[reqId].append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })
    
    def historicalDataEnd(self, reqId, start, end):
        print(f"Historical data complete for request {reqId}")
    
    def position(self, account, contract, position, avgCost):
        self.positions.append({
            'symbol': contract.symbol,
            'position': position,
            'avg_cost': avgCost
        })
    
    def positionEnd(self):
        print("Position data complete")

class IBTrader:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.client = IBClient()
        self.client.connect(host, port, client_id)
        
        # Start message thread
        thread = threading.Thread(target=self.client.run, daemon=True)
        thread.start()
        time.sleep(2)  # Wait for connection
    
    def create_stock_contract(self, symbol, exchange='SMART', currency='USD'):
        """יצירת contract למניה"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = 'STK'
        contract.exchange = exchange
        contract.currency = currency
        return contract
    
    def get_historical_data(self, symbol, duration='1 Y', bar_size='1 day'):
        """קבלת נתונים היסטוריים"""
        contract = self.create_stock_contract(symbol)
        req_id = 1
        
        self.client.data[req_id] = []
        self.client.reqHistoricalData(
            req_id, contract, '', duration, bar_size,
            'TRADES', 1, 1, False, []
        )
        
        time.sleep(5)  # Wait for data
        
        df = pd.DataFrame(self.client.data[req_id])
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        return df
    
    def get_positions(self):
        """קבלת פוזיציות"""
        self.client.positions = []
        self.client.reqPositions()
        time.sleep(2)
        return pd.DataFrame(self.client.positions)
    
    def place_market_order(self, symbol, quantity, action='BUY'):
        """ביצוע פקודת שוק"""
        contract = self.create_stock_contract(symbol)
        
        order = Order()
        order.action = action
        order.orderType = 'MKT'
        order.totalQuantity = quantity
        
        self.client.placeOrder(self.client.order_id, contract, order)
        self.client.order_id += 1
    
    def place_limit_order(self, symbol, quantity, limit_price, action='BUY'):
        """ביצוע פקודת לימיט"""
        contract = self.create_stock_contract(symbol)
        
        order = Order()
        order.action = action
        order.orderType = 'LMT'
        order.totalQuantity = quantity
        order.lmtPrice = limit_price
        
        self.client.placeOrder(self.client.order_id, contract, order)
        self.client.order_id += 1
    
    def place_bracket_order(self, symbol, quantity, limit_price, 
                           take_profit, stop_loss):
        """פקודת bracket עם TP ו-SL"""
        contract = self.create_stock_contract(symbol)
        
        # Parent order (entry)
        parent = Order()
        parent.orderId = self.client.order_id
        parent.action = 'BUY'
        parent.orderType = 'LMT'
        parent.totalQuantity = quantity
        parent.lmtPrice = limit_price
        parent.transmit = False
        
        # Take profit
        tp_order = Order()
        tp_order.orderId = self.client.order_id + 1
        tp_order.action = 'SELL'
        tp_order.orderType = 'LMT'
        tp_order.totalQuantity = quantity
        tp_order.lmtPrice = take_profit
        tp_order.parentId = parent.orderId
        tp_order.transmit = False
        
        # Stop loss
        sl_order = Order()
        sl_order.orderId = self.client.order_id + 2
        sl_order.action = 'SELL'
        sl_order.orderType = 'STP'
        sl_order.totalQuantity = quantity
        sl_order.auxPrice = stop_loss
        sl_order.parentId = parent.orderId
        sl_order.transmit = True  # Transmit all orders
        
        self.client.placeOrder(parent.orderId, contract, parent)
        self.client.placeOrder(tp_order.orderId, contract, tp_order)
        self.client.placeOrder(sl_order.orderId, contract, sl_order)
        
        self.client.order_id += 3
    
    def disconnect(self):
        self.client.disconnect()

# Usage
trader = IBTrader(port=7497)  # TWS paper trading port
df = trader.get_historical_data('AAPL', duration='6 M')
print(df.tail())
```

---

### 1.5 Yahoo Finance (yfinance)

**סקירה כללית:**
ספריית Python חינמית לקבלת נתונים מ-Yahoo Finance.

**מה כולל:**
- נתוני מניות (כל העולם)
- אופציות
- קריפטו
- אינדקסים
- נתונים פונדמנטליים

**יתרונות:**
- חינמי לחלוטין
- קל לשימוש
- אין צורך ב-API key

**חסרונות:**
- לא מתאים לנתונים real-time
- עלול להיות לא יציב

**קוד דוגמה:**
```python
import yfinance as yf
import pandas as pd

class YFinanceAPI:
    @staticmethod
    def get_stock_data(symbol, period='1y', interval='1d'):
        """נתונים היסטוריים"""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        return df
    
    @staticmethod
    def get_multiple_stocks(symbols, period='1y'):
        """נתונים למספר מניות"""
        data = yf.download(symbols, period=period, group_by='ticker')
        return data
    
    @staticmethod
    def get_fundamentals(symbol):
        """נתונים פונדמנטליים"""
        ticker = yf.Ticker(symbol)
        return {
            'info': ticker.info,
            'financials': ticker.financials,
            'balance_sheet': ticker.balance_sheet,
            'cashflow': ticker.cashflow,
            'earnings': ticker.earnings,
            'recommendations': ticker.recommendations
        }
    
    @staticmethod
    def get_options_chain(symbol):
        """שרשרת אופציות"""
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        
        all_options = {}
        for exp in expirations[:3]:  # First 3 expirations
            opt = ticker.option_chain(exp)
            all_options[exp] = {
                'calls': opt.calls,
                'puts': opt.puts
            }
        return all_options
    
    @staticmethod
    def get_institutional_holders(symbol):
        """בעלי מניות מוסדיים"""
        ticker = yf.Ticker(symbol)
        return ticker.institutional_holders
    
    @staticmethod
    def get_analyst_targets(symbol):
        """יעדי מחיר של אנליסטים"""
        ticker = yf.Ticker(symbol)
        return ticker.analyst_price_targets

# Usage
api = YFinanceAPI()
df = api.get_stock_data('AAPL', period='1y')
print(df.tail())

fundamentals = api.get_fundamentals('AAPL')
print(f"PE Ratio: {fundamentals['info'].get('trailingPE')}")
```

---

## 2. ספריות Python לניתוח טכני

### 2.1 TA-Lib

**סקירה כללית:**
ספרייה ותיקה ויציבה (20+ שנים) עם 200+ אינדיקטורים.

**התקנה:**
```bash
# Ubuntu/Debian
sudo apt-get install ta-lib
pip install ta-lib

# macOS
brew install ta-lib
pip install ta-lib

# Windows - download from ta-lib.org
pip install TA-Lib
```

**קוד דוגמה מקיף:**
```python
import talib
import pandas as pd
import numpy as np

class TechnicalAnalysis:
    def __init__(self, df):
        self.df = df
        self.results = pd.DataFrame(index=df.index)
    
    def add_all_indicators(self):
        """הוספת כל האינדיקטורים"""
        self.add_trend_indicators()
        self.add_momentum_indicators()
        self.add_volatility_indicators()
        self.add_volume_indicators()
        self.add_pattern_recognition()
        return self.results
    
    def add_trend_indicators(self):
        """אינדיקטורי מגמה"""
        close = self.df['Close']
        high = self.df['High']
        low = self.df['Low']
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            self.results[f'SMA_{period}'] = talib.SMA(close, period)
            self.results[f'EMA_{period}'] = talib.EMA(close, period)
        
        # Special MAs
        self.results['DEMA_20'] = talib.DEMA(close, 20)
        self.results['TEMA_20'] = talib.TEMA(close, 20)
        self.results['KAMA_20'] = talib.KAMA(close, 20)
        self.results['T3_20'] = talib.T3(close, 20)
        self.results['WMA_20'] = talib.WMA(close, 20)
        
        # Parabolic SAR
        self.results['SAR'] = talib.SAR(high, low)
        
        # ADX Family
        self.results['ADX'] = talib.ADX(high, low, close)
        self.results['ADXR'] = talib.ADXR(high, low, close)
        self.results['PLUS_DI'] = talib.PLUS_DI(high, low, close)
        self.results['MINUS_DI'] = talib.MINUS_DI(high, low, close)
        
        # Aroon
        self.results['AROON_UP'], self.results['AROON_DOWN'] = talib.AROON(high, low)
        self.results['AROON_OSC'] = talib.AROONOSC(high, low)
    
    def add_momentum_indicators(self):
        """אינדיקטורי מומנטום"""
        close = self.df['Close']
        high = self.df['High']
        low = self.df['Low']
        volume = self.df['Volume']
        
        # RSI
        for period in [7, 14, 21]:
            self.results[f'RSI_{period}'] = talib.RSI(close, period)
        
        # MACD
        macd, signal, hist = talib.MACD(close, 12, 26, 9)
        self.results['MACD'] = macd
        self.results['MACD_Signal'] = signal
        self.results['MACD_Hist'] = hist
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close)
        self.results['STOCH_K'] = slowk
        self.results['STOCH_D'] = slowd
        
        fastk, fastd = talib.STOCHF(high, low, close)
        self.results['STOCHF_K'] = fastk
        self.results['STOCHF_D'] = fastd
        
        # StochRSI
        fastk, fastd = talib.STOCHRSI(close)
        self.results['STOCHRSI_K'] = fastk
        self.results['STOCHRSI_D'] = fastd
        
        # Williams %R
        self.results['WILLR'] = talib.WILLR(high, low, close)
        
        # CCI
        self.results['CCI'] = talib.CCI(high, low, close)
        
        # MFI
        self.results['MFI'] = talib.MFI(high, low, close, volume)
        
        # Momentum
        self.results['MOM'] = talib.MOM(close, 10)
        self.results['ROC'] = talib.ROC(close, 10)
        self.results['ROCP'] = talib.ROCP(close, 10)
        
        # Ultimate Oscillator
        self.results['ULTOSC'] = talib.ULTOSC(high, low, close)
        
        # Balance of Power
        self.results['BOP'] = talib.BOP(
            self.df['Open'], high, low, close
        )
    
    def add_volatility_indicators(self):
        """אינדיקטורי תנודתיות"""
        close = self.df['Close']
        high = self.df['High']
        low = self.df['Low']
        
        # ATR
        for period in [7, 14, 21]:
            self.results[f'ATR_{period}'] = talib.ATR(high, low, close, period)
        
        self.results['NATR'] = talib.NATR(high, low, close)
        self.results['TRANGE'] = talib.TRANGE(high, low, close)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, 20, 2, 2)
        self.results['BB_Upper'] = upper
        self.results['BB_Middle'] = middle
        self.results['BB_Lower'] = lower
        self.results['BB_Width'] = (upper - lower) / middle
        self.results['BB_Position'] = (close - lower) / (upper - lower)
    
    def add_volume_indicators(self):
        """אינדיקטורי נפח"""
        close = self.df['Close']
        high = self.df['High']
        low = self.df['Low']
        volume = self.df['Volume']
        
        # OBV
        self.results['OBV'] = talib.OBV(close, volume)
        
        # AD
        self.results['AD'] = talib.AD(high, low, close, volume)
        
        # ADOSC
        self.results['ADOSC'] = talib.ADOSC(high, low, close, volume)
    
    def add_pattern_recognition(self):
        """זיהוי תבניות נרות"""
        open = self.df['Open']
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close']
        
        patterns = [
            'CDLDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLENGULFING',
            'CDLHARAMI', 'CDLMORNINGSTAR', 'CDLEVENINGSTAR',
            'CDLSHOOTINGSTAR', 'CDLINVERTEDHAMMER', 'CDLSPINNINGTOP',
            'CDLMARUBOZU', 'CDL3WHITESOLDIERS', 'CDL3BLACKCROWS',
            'CDLDOJISTAR', 'CDLPIERCING', 'CDLDARKCLOUDCOVER'
        ]
        
        for pattern in patterns:
            func = getattr(talib, pattern)
            self.results[pattern] = func(open, high, low, close)
    
    def generate_signals(self):
        """יצירת סיגנלים מסחריים"""
        signals = pd.DataFrame(index=self.df.index)
        
        # RSI Signals
        signals['RSI_Oversold'] = (self.results['RSI_14'] < 30).astype(int)
        signals['RSI_Overbought'] = (self.results['RSI_14'] > 70).astype(int)
        
        # MACD Signal
        signals['MACD_Bullish'] = (
            (self.results['MACD'] > self.results['MACD_Signal']) &
            (self.results['MACD'].shift(1) <= self.results['MACD_Signal'].shift(1))
        ).astype(int)
        
        signals['MACD_Bearish'] = (
            (self.results['MACD'] < self.results['MACD_Signal']) &
            (self.results['MACD'].shift(1) >= self.results['MACD_Signal'].shift(1))
        ).astype(int)
        
        # MA Crossover
        signals['Golden_Cross'] = (
            (self.results['SMA_50'] > self.results['SMA_200']) &
            (self.results['SMA_50'].shift(1) <= self.results['SMA_200'].shift(1))
        ).astype(int)
        
        signals['Death_Cross'] = (
            (self.results['SMA_50'] < self.results['SMA_200']) &
            (self.results['SMA_50'].shift(1) >= self.results['SMA_200'].shift(1))
        ).astype(int)
        
        # BB Signals
        signals['BB_Oversold'] = (
            self.df['Close'] < self.results['BB_Lower']
        ).astype(int)
        
        signals['BB_Overbought'] = (
            self.df['Close'] > self.results['BB_Upper']
        ).astype(int)
        
        return signals

# Usage
df = pd.read_csv('AAPL.csv', parse_dates=['Date'], index_col='Date')
ta = TechnicalAnalysis(df)
indicators = ta.add_all_indicators()
signals = ta.generate_signals()

print(indicators.tail())
print(signals.tail())
```

---

### 2.2 Pandas-TA

**סקירה כללית:**
ספרייה מודרנית המבוססת על Pandas, עם API נוח יותר.

**התקנה:**
```bash
pip install pandas-ta
```

**קוד דוגמה:**
```python
import pandas as pd
import pandas_ta as ta

# Load data
df = pd.read_csv('AAPL.csv', parse_dates=['Date'], index_col='Date')

# Calculate all indicators at once
df.ta.strategy('All')  # Adds all available indicators

# Or specific indicators
df['RSI'] = df.ta.rsi(length=14)
df['MACD'] = df.ta.macd()['MACD_12_26_9']
df['SMA_20'] = df.ta.sma(length=20)
df['BB'] = df.ta.bbands()

# Custom strategy
custom_strategy = ta.Strategy(
    name="Custom",
    ta=[
        {"kind": "rsi", "length": 14},
        {"kind": "macd", "fast": 12, "slow": 26},
        {"kind": "bbands", "length": 20},
        {"kind": "atr", "length": 14},
        {"kind": "adx", "length": 14}
    ]
)

df.ta.strategy(custom_strategy)
print(df.tail())
```

---

## 3. טבלת השוואה מסכמת

| API/ספרייה | מחיר | Real-Time | Historical | מסחר | יתרון עיקרי |
|:-----------|:-----|:----------|:-----------|:-----|:------------|
| Alpha Vantage | Free-$249/mo | ✅ | ✅ | ❌ | נתונים רשמיים מ-NASDAQ |
| Finnhub | Free-$79+/mo | ✅ | ✅ | ❌ | Sentiment AI |
| Alpaca | Free-$99/mo | ✅ | ✅ | ✅ | Paper + Live Trading |
| IB TWS API | Commission | ✅ | ✅ | ✅ | 150+ שווקים גלובליים |
| Yahoo Finance | Free | 15-min delay | ✅ | ❌ | חינמי, קל לשימוש |
| TA-Lib | Free | N/A | N/A | N/A | 200+ אינדיקטורים |
| Pandas-TA | Free | N/A | N/A | N/A | API מודרני |

---

## 4. המלצות לפי סוג משתמש

**למתחילים:**
- Yahoo Finance (yfinance) לנתונים
- Pandas-TA לניתוח טכני
- Alpaca Paper Trading לתרגול

**לסוחרים פעילים:**
- Finnhub/Alpha Vantage לנתונים
- TA-Lib לניתוח מתקדם
- Alpaca/IB למסחר אמיתי

**למוסדות/מתקדמים:**
- IB TWS API
- Bloomberg Terminal (לא נסקר כאן)
- Custom data feeds

---

## סיכום

מסמך זה מספק מדריך מקיף ל-APIs ומקורות נתונים למסחר, כולל:

1. **APIs לנתוני שוק:** Alpha Vantage, Finnhub, Alpaca, IB, Yahoo Finance
2. **ספריות לניתוח טכני:** TA-Lib, Pandas-TA
3. **קוד מעשי ומוכן לשימוש** לכל API
4. **השוואות ותמחור**
5. **המלצות לפי רמת משתמש**

הבחירה ב-API תלויה בדרישות הספציפיות שלך: תקציב, סוג מסחר, צורך בנתונים real-time, ושווקים רלוונטיים.
