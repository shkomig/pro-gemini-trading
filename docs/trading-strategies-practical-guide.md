# מדריך אסטרטגיות מסחר - טבלאות ודוגמאות מעשיות

## 1. השוואת אסטרטגיות מסחר אלגוריתמי

### 1.1 טבלת השוואה מקיפה

| אסטרטגיה | סוג שוק מתאים | דרישת מהירות | רמת מורכבות | תשואה שנתית ממוצעת | סיכון |
|:---------|:--------------|:-------------|:-------------|:-------------------|:------|
| Momentum/Trend Following | שווקים במגמה | בינונית | נמוכה-בינונית | 25-40% | בינוני |
| Mean Reversion | שווקים בטווח | בינונית | בינונית | 15-30% | בינוני |
| Breakout | שווקים תנודתיים | גבוהה | בינונית | 20-50%+ | גבוה |
| Arbitrage | כל השווקים | גבוהה מאוד | גבוהה | 10-25% | נמוך |
| HFT | שווקים נזילים | קריטית (µs) | גבוהה מאוד | 50-200%+ | גבוה |
| Grid Trading | שווקים בטווח | נמוכה | נמוכה | 15-25% | בינוני-גבוה |
| Sentiment-Based | כל השווקים | בינונית | גבוהה | 20-35% | בינוני |
| ML/AI Driven | כל השווקים | משתנה | גבוהה מאוד | 30-80%+ | משתנה |

### 1.2 בחירת אסטרטגיה לפי פרופיל משתמש

| פרופיל סוחר | אסטרטגיה מומלצת | סיבה |
|:-----------|:----------------|:-----|
| מתחיל | Moving Average Crossover | פשוטה להבנה ויישום |
| Day Trader | Momentum + Breakout | מתאימות למסחר תוך-יומי |
| Swing Trader | Mean Reversion + Trend | מיטוב למסגרות זמן של ימים-שבועות |
| משקיע מוסדי | Statistical Arbitrage | דורשת הון גדול וקשרי שוק |
| מפתח מתקדם | ML/RL Agents | מקסום יתרון טכנולוגי |

---

## 2. פרמטרי סריקה מומלצים לפי סגנון מסחר

### 2.1 סריקה ל-Day Trading

```yaml
Pre-Market Gap Scanner:
  min_gap_percent: 3%
  min_volume: 500,000
  min_atr: $0.50
  max_price: $100
  min_price: $5
  news_catalyst: required
  float: <20M shares (preferred)

Momentum Scanner:
  rsi_above: 60
  price_above_vwap: true
  volume_ratio: >2x (vs 10-day avg)
  macd_crossover: recent
  sector_strength: top 3

Breakout Scanner:
  price_near_high: within 3% of 52-week high
  consolidation_days: 5-20
  volume_increase: >50% vs avg
  support_resistance_test: 3+ touches
```

### 2.2 סריקה ל-Swing Trading

```yaml
Pullback Scanner:
  trend: uptrend (price > 20 SMA > 50 SMA)
  pullback_to: 20-day SMA
  rsi_range: 30-50
  volume: declining during pullback
  recovery_candle: bullish

Breakout Anticipation:
  pattern: ascending triangle, cup & handle
  time_in_pattern: 3-8 weeks
  volume_pattern: decreasing then increasing
  relative_strength: >1 vs SPY

Oversold Reversal:
  rsi: <30
  price_at: support level
  volume_spike: yes
  divergence: bullish (price/RSI)
```

---

## 3. דוגמאות קוד מעשיות

### 3.1 סורק מומנטום בסיסי (Python)

```python
import pandas as pd
import talib
import yfinance as yf

class MomentumScanner:
    def __init__(self, symbols, period='6mo'):
        self.symbols = symbols
        self.period = period
        
    def scan(self):
        results = []
        
        for symbol in self.symbols:
            try:
                data = yf.download(symbol, period=self.period, progress=False)
                if len(data) < 50:
                    continue
                
                # Calculate indicators
                data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
                data['MACD'], data['MACD_Signal'], _ = talib.MACD(data['Close'])
                data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
                data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
                data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'])
                
                latest = data.iloc[-1]
                
                # Momentum criteria
                conditions = {
                    'price_above_sma20': latest['Close'] > latest['SMA_20'],
                    'sma20_above_sma50': latest['SMA_20'] > latest['SMA_50'],
                    'rsi_bullish': 50 < latest['RSI'] < 70,
                    'macd_positive': latest['MACD'] > latest['MACD_Signal'],
                    'good_volatility': latest['ATR'] > 0.5
                }
                
                if all(conditions.values()):
                    results.append({
                        'Symbol': symbol,
                        'Price': round(latest['Close'], 2),
                        'RSI': round(latest['RSI'], 2),
                        'ATR': round(latest['ATR'], 2),
                        'MACD': round(latest['MACD'], 4),
                        'Score': sum(conditions.values())
                    })
                    
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                
        return pd.DataFrame(results).sort_values('RSI', ascending=False)

# Usage
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
scanner = MomentumScanner(symbols)
momentum_stocks = scanner.scan()
print(momentum_stocks)
```

### 3.2 מערכת Backtesting פשוטה

```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd

class RSI_Strategy(Strategy):
    """
    אסטרטגיית RSI עם סינון מגמה
    קנייה: RSI < 30 + מחיר מעל SMA 200
    מכירה: RSI > 70 או Stop Loss
    """
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    sma_period = 200
    stop_loss_pct = 0.05  # 5%
    take_profit_pct = 0.15  # 15%
    
    def init(self):
        close = self.data.Close
        self.rsi = self.I(lambda x: talib.RSI(x, self.rsi_period), close)
        self.sma = self.I(SMA, close, self.sma_period)
        
    def next(self):
        price = self.data.Close[-1]
        
        # אין פוזיציה - חפש כניסה
        if not self.position:
            # תנאי כניסה: RSI oversold + מעל SMA (מגמה עולה)
            if self.rsi[-1] < self.rsi_oversold and price > self.sma[-1]:
                # חשב Stop Loss ו-Take Profit
                sl = price * (1 - self.stop_loss_pct)
                tp = price * (1 + self.take_profit_pct)
                self.buy(sl=sl, tp=tp)
                
        # יש פוזיציה - חפש יציאה
        else:
            # יציאה כאשר RSI overbought
            if self.rsi[-1] > self.rsi_overbought:
                self.position.close()

# Load data
data = pd.read_csv('AAPL.csv', parse_dates=['Date'], index_col='Date')

# Run backtest
bt = Backtest(data, RSI_Strategy, 
              cash=100000, 
              commission=0.002,
              exclusive_orders=True)

results = bt.run()
print(results)

# Optimize parameters
optimization_results = bt.optimize(
    rsi_period=range(10, 20, 2),
    rsi_oversold=range(25, 35, 5),
    rsi_overbought=range(65, 75, 5),
    maximize='Sharpe Ratio'
)
print(optimization_results)

# Plot
bt.plot()
```

### 3.3 מודל ML לחיזוי כיוון מחיר

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import talib

class MLTradingModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.feature_names = []
        
    def create_features(self, df):
        """יצירת Features טכניים"""
        features = pd.DataFrame(index=df.index)
        
        # Momentum indicators
        features['RSI_14'] = talib.RSI(df['Close'], 14)
        features['RSI_7'] = talib.RSI(df['Close'], 7)
        features['MACD'], features['MACD_Signal'], _ = talib.MACD(df['Close'])
        features['MACD_Hist'] = features['MACD'] - features['MACD_Signal']
        features['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])
        features['MOM_10'] = talib.MOM(df['Close'], 10)
        
        # Trend indicators
        features['SMA_10'] = talib.SMA(df['Close'], 10)
        features['SMA_20'] = talib.SMA(df['Close'], 20)
        features['SMA_50'] = talib.SMA(df['Close'], 50)
        features['EMA_10'] = talib.EMA(df['Close'], 10)
        features['Price_SMA_Ratio'] = df['Close'] / features['SMA_20']
        
        # Volatility indicators
        features['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
        features['BB_Upper'], features['BB_Middle'], features['BB_Lower'] = \
            talib.BBANDS(df['Close'])
        features['BB_Width'] = (features['BB_Upper'] - features['BB_Lower']) / features['BB_Middle']
        features['BB_Position'] = (df['Close'] - features['BB_Lower']) / \
                                  (features['BB_Upper'] - features['BB_Lower'])
        
        # Volume indicators
        features['Volume_SMA'] = df['Volume'].rolling(20).mean()
        features['Volume_Ratio'] = df['Volume'] / features['Volume_SMA']
        features['OBV'] = talib.OBV(df['Close'], df['Volume'])
        
        # Price patterns
        features['Returns_1'] = df['Close'].pct_change(1)
        features['Returns_5'] = df['Close'].pct_change(5)
        features['Returns_10'] = df['Close'].pct_change(10)
        features['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'RSI_lag_{lag}'] = features['RSI_14'].shift(lag)
            features[f'Returns_lag_{lag}'] = features['Returns_1'].shift(lag)
            
        return features.dropna()
    
    def create_target(self, df, horizon=5, threshold=0.02):
        """
        יצירת Target - 1 אם המחיר עלה מעל threshold בהורייזון, אחרת 0
        """
        future_returns = df['Close'].shift(-horizon) / df['Close'] - 1
        target = (future_returns > threshold).astype(int)
        return target
    
    def prepare_data(self, df, horizon=5, threshold=0.02):
        """הכנת נתונים לאימון"""
        features = self.create_features(df)
        target = self.create_target(df, horizon, threshold)
        
        # יישור אינדקסים
        common_idx = features.index.intersection(target.dropna().index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, X, y):
        """אימון המודל"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = self.model.predict(X_test)
        
        print("=== Model Performance ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Features:")
        print(importance.head(10))
        
        return X_test, y_test, y_pred
    
    def predict(self, features):
        """חיזוי על נתונים חדשים"""
        probabilities = self.model.predict_proba(features)
        predictions = self.model.predict(features)
        return predictions, probabilities[:, 1]

# Usage
df = pd.read_csv('AAPL.csv', parse_dates=['Date'], index_col='Date')

model = MLTradingModel()
X, y = model.prepare_data(df, horizon=5, threshold=0.02)
X_test, y_test, predictions = model.train(X, y)
```

---

## 4. טבלאות השוואת כלים ופלטפורמות

### 4.1 סורקי מניות - השוואה

| כלי | מחיר חודשי | Real-Time | AI/ML | Alerts | Mobile | התאמה ל- |
|:----|:-----------|:----------|:------|:-------|:-------|:---------|
| Trade Ideas | $89-$254 | ✅ | ✅ (Holly) | ✅ | ❌ | Day Trading |
| TradingView | Free-$56 | ✅ | ❌ | ✅ | ✅ | כל הסגנונות |
| Benzinga Pro | $37-$197 | ✅ | ❌ | ✅ | ✅ (iOS) | News-Based |
| Finviz | Free/$39 | ✅ (Elite) | ❌ | ✅ | ❌ | Swing Trading |
| TC2000 | $25-$100 | ✅ | ❌ | ✅ | ✅ | Technical Analysis |
| TrendSpider | $82-$183 | ✅ | ✅ | ✅ | ✅ | Pattern Recognition |
| Scanz | Premium | ✅ | ❌ | ✅ | ❌ | Penny Stocks |

### 4.2 APIs לנתוני שוק - השוואה

| API | Free Tier | Real-Time | Historical | נכסים | תמיכת Python |
|:----|:----------|:----------|:-----------|:------|:-------------|
| Alpha Vantage | 5/min | ✅ | ✅ | Stocks, Forex, Crypto | מצוינת |
| Finnhub | 60/min | ✅ | ✅ | Stocks, Forex | טובה |
| Alpaca | 200/min | ✅ | 7+ years | US Stocks, Options, Crypto | מצוינת |
| Yahoo Finance | Unlimited | 15-min delay | ✅ | Global | מצוינת (yfinance) |
| IB API | Unlimited* | ✅ | ✅ | 150+ Markets | טובה |
| Polygon.io | Limited | ✅ | ✅ | US Markets | טובה |

*דורש חשבון IB פעיל

### 4.3 ספריות Backtesting - השוואה

| ספרייה | רישיון | קושי שימוש | מהירות | Live Trading | תיעוד |
|:-------|:-------|:-----------|:-------|:-------------|:------|
| Backtrader | MIT | בינוני | טובה | ✅ | מעולה |
| Backtesting.py | MIT | קל | מעולה | ❌ | טוב |
| Vectorbt | MIT | בינוני-קשה | מעולה | ❌ | טוב |
| Zipline | Apache 2.0 | קשה | טובה | ❌ | טוב |
| bt | MIT | קל | טובה | ❌ | טוב |
| QSTrader | MIT | קשה | טובה | ✅ | בינוני |

---

## 5. מדדי ביצועים - נוסחאות וחישובים

### 5.1 Sharpe Ratio

```
Sharpe Ratio = (Rp - Rf) / σp

כאשר:
Rp = תשואה ממוצעת של התיק
Rf = תשואה ללא סיכון (Treasury rate)
σp = סטיית תקן של תשואות התיק
```

**פירוש:**
- > 1.0: טוב
- > 2.0: מעולה
- > 3.0: יוצא דופן

### 5.2 Maximum Drawdown

```
Max Drawdown = (Peak - Trough) / Peak × 100%

כאשר:
Peak = הערך הגבוה ביותר שהושג
Trough = הערך הנמוך ביותר לאחר ה-Peak
```

**רמות מומלצות:**
- < 10%: סיכון נמוך
- 10-20%: סיכון בינוני
- > 20%: סיכון גבוה

### 5.3 Profit Factor

```
Profit Factor = Gross Profit / Gross Loss

כאשר:
Gross Profit = סכום כל העסקאות הרווחיות
Gross Loss = סכום כל העסקאות המפסידות
```

**פירוש:**
- > 1.0: רווחי
- > 1.5: טוב
- > 2.0: מעולה

### 5.4 Win Rate ו-Risk-Reward

```
Win Rate = Winning Trades / Total Trades × 100%

Required Win Rate = 1 / (1 + R:R)

לדוגמה:
- R:R של 1:2 דורש Win Rate של 33%+
- R:R של 1:3 דורש Win Rate של 25%+
```

### 5.5 Kelly Criterion

```
Kelly % = (W × R - L) / R

כאשר:
W = הסתברות לזכייה
L = הסתברות להפסד (1 - W)
R = יחס Average Win / Average Loss

לדוגמה:
W = 55%, L = 45%, R = 1.5
Kelly = (0.55 × 1.5 - 0.45) / 1.5 = 25%

המלצה: להשתמש ב-Half Kelly (12.5% בדוגמה)
```

---

## 6. Checklists מעשיות

### 6.1 לפני כניסה לעסקה

```
□ האם המגמה הכללית תומכת בכיוון העסקה?
□ האם יש אישור נפח (Volume)?
□ האם RSI לא בקיצון (לא מעל 80 לקנייה)?
□ האם הגדרתי Stop-Loss?
□ האם הגדרתי Target?
□ האם יחס R:R הוא לפחות 1:2?
□ האם גודל הפוזיציה תואם לחוק 1%?
□ האם אין אירועים חדשותיים צפויים?
□ האם בדקתי Spread ו-Liquidity?
□ האם זה תואם את תוכנית המסחר שלי?
```

### 6.2 סקירה יומית (Pre-Market)

```
□ בדיקת Futures והכוון הכללי
□ סריקת Gap Ups/Downs
□ זיהוי מניות עם חדשות
□ סקירת Watchlist
□ בדיקת אירועים כלכליים (Calendar)
□ סקירת סקטורים חזקים/חלשים
□ עדכון רמות תמיכה/התנגדות
□ הגדרת יעדים ליום
```

### 6.3 סקירה שבועית

```
□ סיכום עסקאות השבוע
□ חישוב מדדי ביצועים
□ ניתוח עסקאות מפסידות
□ ניתוח עסקאות מוצלחות
□ זיהוי דפוסים חוזרים
□ עדכון Watchlist לשבוע הבא
□ סקירת ביצועי האסטרטגיה
□ התאמות נדרשות
```

---

## 7. תרחישי מסחר - Case Studies

### 7.1 תרחיש Momentum Breakout

**מצב:**
- NVDA נסחרת ב-$450
- Consolidation של 2 שבועות בין $440-$455
- Volume ממוצע: 50M
- RSI: 55
- מעל כל ה-SMAs

**טריגר כניסה:**
- פריצה מעל $455 עם Volume > 75M
- RSI עולה מעל 60

**ביצוע:**
```
Entry: $456
Stop Loss: $445 (2.4% סיכון)
Target 1: $470 (3% רווח) - 50% position
Target 2: $490 (7.5% רווח) - 50% position
Position Size: 1% capital risk / $11 = shares
```

### 7.2 תרחיש Mean Reversion

**מצב:**
- AAPL ירדה 8% בשבוע
- RSI: 28 (Oversold)
- נגעה ברצועת Bollinger תחתונה
- מעל SMA 200
- Volume גבוה בירידה

**טריגר כניסה:**
- נר היפוך (Hammer/Doji) ברמת תמיכה
- RSI מתחיל לעלות

**ביצוע:**
```
Entry: רצועת Bollinger אמצעית כיעד ראשוני
Stop Loss: מתחת לשפל האחרון
Target: BB Middle או רמת התנגדות קרובה
R:R: לפחות 1:2
```

---

## 8. טיפים מתקדמים

### 8.1 אופטימיזציה ללא Overfitting

```python
# Walk-Forward Optimization
def walk_forward_optimization(data, strategy, n_splits=5):
    """
    חלוקת הנתונים לתקופות אימון ובדיקה מתגלגלות
    """
    split_size = len(data) // n_splits
    results = []
    
    for i in range(n_splits - 1):
        train_start = 0
        train_end = (i + 1) * split_size
        test_start = train_end
        test_end = test_start + split_size
        
        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]
        
        # Optimize on train
        best_params = optimize(train_data, strategy)
        
        # Test on out-of-sample
        result = backtest(test_data, strategy, best_params)
        results.append(result)
    
    return pd.DataFrame(results)
```

### 8.2 Position Sizing דינמי

```python
def dynamic_position_size(equity, volatility, base_risk=0.01):
    """
    התאמת גודל פוזיציה לתנודתיות השוק
    """
    # VIX-based adjustment
    if volatility > 30:
        risk_multiplier = 0.5  # High volatility - reduce risk
    elif volatility > 20:
        risk_multiplier = 0.75
    else:
        risk_multiplier = 1.0
    
    adjusted_risk = base_risk * risk_multiplier
    position_value = equity * adjusted_risk
    
    return position_value
```

### 8.3 Multi-Timeframe Confirmation

```python
def multi_timeframe_signal(symbol, timeframes=['1d', '4h', '1h']):
    """
    אישור סיגנל במספר מסגרות זמן
    """
    signals = {}
    
    for tf in timeframes:
        data = get_data(symbol, timeframe=tf)
        
        # Calculate indicators
        trend = 1 if data['SMA_20'][-1] > data['SMA_50'][-1] else -1
        momentum = 1 if data['RSI'][-1] > 50 else -1
        
        signals[tf] = (trend + momentum) / 2
    
    # Aggregate signal
    final_signal = sum(signals.values()) / len(signals)
    
    return {
        'signals': signals,
        'final': final_signal,
        'strength': abs(final_signal)
    }
```

---

## סיכום

מסמך זה מספק ארגז כלים מעשי לסוחר האלגוריתמי, כולל:
- השוואות מקיפות של אסטרטגיות וכלים
- קוד מעשי ומוכן לשימוש
- נוסחאות וחישובים חיוניים
- Checklists לעבודה יומיומית
- תרחישי מסחר אמיתיים

הקבצים הנלווים מספקים מידע נוסף על:
- אסטרטגיות AI/ML מתקדמות
- ניהול סיכונים
- אינדיקטורים טכניים
- מקורות נתונים ו-APIs
