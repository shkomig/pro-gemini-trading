# מדריך AI ולמידת מכונה במסחר - Deep Dive

## 1. סקירת טכנולוגיות AI במסחר

### 1.1 למידה עמוקה (Deep Learning)

#### LSTM - Long Short-Term Memory

**מהות הטכנולוגיה:**
LSTM הוא סוג של רשת נוירונים חוזרת (RNN) שתוכננה לזכור מידע לאורך זמן. במסחר, היא מסוגלת ללמוד דפוסים מורכבים מסדרות זמן של מחירי מניות.

**ארכיטקטורה טיפוסית:**
```
Input Layer → LSTM Layer (64-128 units) → 
LSTM Layer (32-64 units) → Dense Layer → 
Output Layer (Prediction)
```

**תוצאות מחקריות מוכחות:**
- דיוק תחזית של 93%-97.7% על מניות NASDAQ
- PNJ: 97.7% דיוק (הגבוה ביותר)
- MSN, TPB: ~97% דיוק
- MAPE של 2.65%-2.72% (לעומת 20.66% ב-ARIMA)

**קוד דוגמה:**
```python
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class LSTMPricePredictor:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def create_sequences(self, data):
        """יצירת רצפים לאימון LSTM"""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """בניית מודל LSTM"""
        model = Sequential([
            LSTM(units=128, return_sequences=True, 
                 input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=64, return_sequences=True),
            Dropout(0.2),
            LSTM(units=32, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, df, epochs=50, batch_size=32):
        """אימון המודל"""
        # Scale data
        data = self.scaler.fit_transform(df['Close'].values.reshape(-1, 1))
        
        # Create sequences
        X, y = self.create_sequences(data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build and train
        self.model = self.build_model((X_train.shape[1], 1))
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        return history
    
    def predict(self, recent_data):
        """חיזוי מחיר עתידי"""
        scaled_data = self.scaler.transform(
            recent_data['Close'].values.reshape(-1, 1)
        )
        X = np.array([scaled_data[-self.lookback:, 0]])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        prediction = self.model.predict(X)
        return self.scaler.inverse_transform(prediction)[0, 0]

# Usage
predictor = LSTMPricePredictor(lookback=60)
history = predictor.train(df, epochs=50)
next_price = predictor.predict(df)
print(f"Predicted next price: ${next_price:.2f}")
```

#### CNN - Convolutional Neural Networks

**יישום במסחר:**
- זיהוי תבניות גרפיות (Chart Patterns)
- עיבוד תמונות של גרפים
- שילוב עם LSTM לדיוק גבוה יותר

**Hybrid LSTM-CNN:**
```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

def build_hybrid_model(lookback, n_features):
    model = Sequential([
        # CNN layers for feature extraction
        Conv1D(filters=64, kernel_size=3, activation='relu',
               input_shape=(lookback, n_features)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # LSTM layers for sequence learning
        LSTM(units=50, return_sequences=True),
        LSTM(units=50),
        
        # Output
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model
```

### 1.2 Reinforcement Learning (למידה מחזקת)

#### עקרונות בסיסיים

**מרכיבי מערכת RL למסחר:**
1. **Agent (סוכן):** מקבל החלטות מסחר
2. **Environment (סביבה):** השוק הפיננסי
3. **State (מצב):** נתוני שוק נוכחיים + פוזיציות
4. **Action (פעולה):** קנייה, מכירה, או החזקה
5. **Reward (תגמול):** רווח/הפסד מהעסקה

**אלגוריתמים מרכזיים:**

**DQN (Deep Q-Network):**
```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class DQNTrader:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size  # 0=hold, 1=buy, 2=sell
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state)
                target = reward + self.gamma * torch.max(
                    self.model(next_state_tensor)
                ).item()
            
            state_tensor = torch.FloatTensor(state)
            target_f = self.model(state_tensor).detach().numpy()
            target_f[action] = target
            
            # Train step
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate
            )
            criterion = nn.MSELoss()
            
            output = self.model(state_tensor)
            loss = criterion(output, torch.FloatTensor(target_f))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

**PPO (Proximal Policy Optimization):**
- יציב יותר מ-DQN
- מתאים לאקשנים רציפים
- פופולרי בקרנות גידור

**A2C (Advantage Actor-Critic):**
- שילוב של Policy Gradient ו-Value Function
- מהיר יותר באימון
- מתאים ל-multi-asset trading

#### תוצאות מחקריות

**MVMM (Multi-Agent Virtual Market Model):**
- שיפור רווחיות של 12%
- Sharpe Ratio גבוה יותר
- Maximum Drawdown נמוך יותר
- הוכח על שוק המניות האמריקאי ו-futures סיניים

### 1.3 NLP וניתוח סנטימנט

#### טכנולוגיות מובילות

**Transformer Models (BERT, GPT):**
- הבנת הקשר ברמה גבוהה
- זיהוי סרקזם וניואנסים
- ניתוח real-time

**Pipeline לניתוח סנטימנט:**
```python
from transformers import pipeline
from textblob import TextBlob
import feedparser
import pandas as pd

class SentimentAnalyzer:
    def __init__(self):
        # Load FinBERT for financial sentiment
        self.finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
        
    def analyze_text(self, text):
        """ניתוח סנטימנט של טקסט בודד"""
        # FinBERT analysis
        finbert_result = self.finbert(text[:512])[0]  # max 512 tokens
        
        # TextBlob for additional metrics
        blob = TextBlob(text)
        
        return {
            'finbert_sentiment': finbert_result['label'],
            'finbert_score': finbert_result['score'],
            'polarity': blob.sentiment.polarity,  # -1 to 1
            'subjectivity': blob.sentiment.subjectivity  # 0 to 1
        }
    
    def analyze_news(self, symbol, sources=None):
        """ניתוח סנטימנט מחדשות"""
        if sources is None:
            sources = [
                f'https://news.google.com/rss/search?q={symbol}+stock',
                f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}'
            ]
        
        articles = []
        for source in sources:
            try:
                feed = feedparser.parse(source)
                for entry in feed.entries[:10]:
                    sentiment = self.analyze_text(
                        entry.title + " " + entry.get('summary', '')
                    )
                    articles.append({
                        'title': entry.title,
                        'published': entry.get('published'),
                        **sentiment
                    })
            except Exception as e:
                print(f"Error parsing {source}: {e}")
        
        return pd.DataFrame(articles)
    
    def get_aggregate_sentiment(self, symbol):
        """חישוב סנטימנט מצרפי"""
        df = self.analyze_news(symbol)
        
        if df.empty:
            return {'sentiment': 'neutral', 'score': 0}
        
        # Weighted average
        positive = (df['finbert_sentiment'] == 'positive').sum()
        negative = (df['finbert_sentiment'] == 'negative').sum()
        total = len(df)
        
        sentiment_score = (positive - negative) / total
        avg_polarity = df['polarity'].mean()
        
        if sentiment_score > 0.2:
            sentiment = 'bullish'
        elif sentiment_score < -0.2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': sentiment_score,
            'polarity': avg_polarity,
            'articles_analyzed': total
        }

# Usage
analyzer = SentimentAnalyzer()
result = analyzer.get_aggregate_sentiment('AAPL')
print(f"Sentiment: {result['sentiment']}, Score: {result['score']:.2f}")
```

#### מקרי שימוש מוכחים

**Bollen et al. (2011):**
- ניתוח מיליוני ציוצי Twitter
- מתאם בין מצב רוח ציבורי ל-DJIA
- דיוק תחזית משופר

**JP Morgan Chase:**
- NLP לניתוח חדשות ורשתות חברתיות
- שיפור ניהול סיכונים
- זיהוי מוקדם של איומים שוקיים

**Accern:**
- אלגוריתמים מונעי סנטימנט
- עלו על מודלים מסורתיים
- הסתגלות real-time לתנאי שוק

---

## 2. Feature Engineering למודלי ML

### 2.1 סוגי Features

#### Features טכניים
```python
import talib
import pandas as pd
import numpy as np

def create_technical_features(df):
    """יצירת Features טכניים מקיפים"""
    features = pd.DataFrame(index=df.index)
    
    # === Momentum Indicators ===
    features['RSI_14'] = talib.RSI(df['Close'], 14)
    features['RSI_7'] = talib.RSI(df['Close'], 7)
    features['RSI_21'] = talib.RSI(df['Close'], 21)
    
    # MACD
    macd, signal, hist = talib.MACD(df['Close'], 12, 26, 9)
    features['MACD'] = macd
    features['MACD_Signal'] = signal
    features['MACD_Hist'] = hist
    features['MACD_Divergence'] = macd - signal
    
    # Stochastic
    features['STOCH_K'], features['STOCH_D'] = talib.STOCH(
        df['High'], df['Low'], df['Close']
    )
    
    # Williams %R
    features['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'])
    
    # ADX - Trend Strength
    features['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])
    features['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'])
    features['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'])
    
    # Momentum
    features['MOM_10'] = talib.MOM(df['Close'], 10)
    features['ROC_10'] = talib.ROC(df['Close'], 10)
    
    # === Trend Indicators ===
    features['SMA_5'] = talib.SMA(df['Close'], 5)
    features['SMA_10'] = talib.SMA(df['Close'], 10)
    features['SMA_20'] = talib.SMA(df['Close'], 20)
    features['SMA_50'] = talib.SMA(df['Close'], 50)
    features['SMA_200'] = talib.SMA(df['Close'], 200)
    
    features['EMA_5'] = talib.EMA(df['Close'], 5)
    features['EMA_10'] = talib.EMA(df['Close'], 10)
    features['EMA_20'] = talib.EMA(df['Close'], 20)
    
    # Price relative to MAs
    features['Price_SMA20_Ratio'] = df['Close'] / features['SMA_20']
    features['Price_SMA50_Ratio'] = df['Close'] / features['SMA_50']
    features['SMA_Cross'] = (features['SMA_20'] > features['SMA_50']).astype(int)
    
    # === Volatility Indicators ===
    features['ATR_14'] = talib.ATR(df['High'], df['Low'], df['Close'], 14)
    features['ATR_Percent'] = features['ATR_14'] / df['Close'] * 100
    
    upper, middle, lower = talib.BBANDS(df['Close'], 20, 2, 2)
    features['BB_Upper'] = upper
    features['BB_Middle'] = middle
    features['BB_Lower'] = lower
    features['BB_Width'] = (upper - lower) / middle
    features['BB_Position'] = (df['Close'] - lower) / (upper - lower)
    
    # Historical Volatility
    features['HV_10'] = df['Close'].pct_change().rolling(10).std() * np.sqrt(252)
    features['HV_20'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    
    # === Volume Indicators ===
    features['Volume_SMA_10'] = df['Volume'].rolling(10).mean()
    features['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
    features['Volume_Ratio'] = df['Volume'] / features['Volume_SMA_20']
    features['OBV'] = talib.OBV(df['Close'], df['Volume'])
    features['AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
    features['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], 14)
    
    # === Price Patterns ===
    features['Returns_1d'] = df['Close'].pct_change(1)
    features['Returns_5d'] = df['Close'].pct_change(5)
    features['Returns_10d'] = df['Close'].pct_change(10)
    features['Returns_20d'] = df['Close'].pct_change(20)
    
    features['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    features['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    features['Close_Open_Range'] = (df['Close'] - df['Open']) / df['Open']
    
    # Distance from high/low
    features['Dist_52W_High'] = df['Close'] / df['High'].rolling(252).max() - 1
    features['Dist_52W_Low'] = df['Close'] / df['Low'].rolling(252).min() - 1
    
    return features
```

#### Features זמניים
```python
def create_time_features(df):
    """יצירת Features מבוססי זמן"""
    features = pd.DataFrame(index=df.index)
    
    # Basic time features
    features['Hour'] = df.index.hour
    features['DayOfWeek'] = df.index.dayofweek
    features['DayOfMonth'] = df.index.day
    features['Month'] = df.index.month
    features['Quarter'] = df.index.quarter
    features['WeekOfYear'] = df.index.isocalendar().week
    
    # Cyclical encoding (prevents discontinuity)
    features['Hour_Sin'] = np.sin(2 * np.pi * features['Hour'] / 24)
    features['Hour_Cos'] = np.cos(2 * np.pi * features['Hour'] / 24)
    features['Day_Sin'] = np.sin(2 * np.pi * features['DayOfWeek'] / 7)
    features['Day_Cos'] = np.cos(2 * np.pi * features['DayOfWeek'] / 7)
    features['Month_Sin'] = np.sin(2 * np.pi * features['Month'] / 12)
    features['Month_Cos'] = np.cos(2 * np.pi * features['Month'] / 12)
    
    # Trading session
    features['Is_US_Open'] = ((features['Hour'] >= 9) & 
                              (features['Hour'] < 16)).astype(int)
    features['Is_Pre_Market'] = ((features['Hour'] >= 4) & 
                                  (features['Hour'] < 9)).astype(int)
    features['Is_After_Hours'] = ((features['Hour'] >= 16) & 
                                   (features['Hour'] < 20)).astype(int)
    
    # Special days
    features['Is_Monday'] = (features['DayOfWeek'] == 0).astype(int)
    features['Is_Friday'] = (features['DayOfWeek'] == 4).astype(int)
    features['Is_Month_End'] = (features['DayOfMonth'] >= 25).astype(int)
    features['Is_Month_Start'] = (features['DayOfMonth'] <= 5).astype(int)
    
    return features
```

#### Features מתקדמים
```python
def create_advanced_features(df, features):
    """יצירת Features מתקדמים"""
    
    # === Lagged Features ===
    for col in ['RSI_14', 'MACD', 'Returns_1d', 'Volume_Ratio']:
        for lag in [1, 2, 3, 5, 10]:
            features[f'{col}_Lag_{lag}'] = features[col].shift(lag)
    
    # === Rolling Statistics ===
    for window in [5, 10, 20]:
        features[f'Returns_Mean_{window}'] = features['Returns_1d'].rolling(window).mean()
        features[f'Returns_Std_{window}'] = features['Returns_1d'].rolling(window).std()
        features[f'Returns_Skew_{window}'] = features['Returns_1d'].rolling(window).skew()
        features[f'Returns_Kurt_{window}'] = features['Returns_1d'].rolling(window).kurt()
    
    # === Interaction Features ===
    features['RSI_MACD_Interaction'] = features['RSI_14'] * features['MACD_Hist']
    features['Volume_Volatility'] = features['Volume_Ratio'] * features['ATR_Percent']
    features['Momentum_Trend'] = features['MOM_10'] * features['ADX']
    
    # === Regime Features ===
    features['Is_Uptrend'] = (features['SMA_20'] > features['SMA_50']).astype(int)
    features['Is_High_Volatility'] = (features['ATR_Percent'] > 
                                       features['ATR_Percent'].rolling(50).mean()).astype(int)
    features['Is_High_Volume'] = (features['Volume_Ratio'] > 1.5).astype(int)
    
    # === Pattern Recognition ===
    # Candlestick patterns (using TA-Lib)
    patterns = ['CDLDOJI', 'CDLHAMMER', 'CDLENGULFING', 'CDLMORNINGSTAR']
    for pattern in patterns:
        func = getattr(talib, pattern)
        features[pattern] = func(df['Open'], df['High'], df['Low'], df['Close'])
    
    return features
```

### 2.2 בחירת Features

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import shap

class FeatureSelector:
    def __init__(self, n_features=50):
        self.n_features = n_features
        self.selected_features = None
        
    def random_forest_importance(self, X, y):
        """בחירה לפי חשיבות Random Forest"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance.head(self.n_features)['feature'].tolist()
    
    def mutual_information(self, X, y):
        """בחירה לפי מידע הדדי"""
        selector = SelectKBest(mutual_info_classif, k=self.n_features)
        selector.fit(X, y)
        
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        return scores.head(self.n_features)['feature'].tolist()
    
    def shap_importance(self, X, y):
        """בחירה לפי SHAP values"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)
        
        importance = np.abs(shap_values[1]).mean(axis=0)
        
        scores = pd.DataFrame({
            'feature': X.columns,
            'shap_importance': importance
        }).sort_values('shap_importance', ascending=False)
        
        return scores.head(self.n_features)['feature'].tolist()
    
    def select_features(self, X, y, method='ensemble'):
        """בחירת Features סופית"""
        if method == 'rf':
            self.selected_features = self.random_forest_importance(X, y)
        elif method == 'mi':
            self.selected_features = self.mutual_information(X, y)
        elif method == 'shap':
            self.selected_features = self.shap_importance(X, y)
        elif method == 'ensemble':
            # שילוב כל השיטות
            rf_features = set(self.random_forest_importance(X, y))
            mi_features = set(self.mutual_information(X, y))
            
            # Features שנבחרו בשתי השיטות
            common_features = rf_features.intersection(mi_features)
            self.selected_features = list(common_features)[:self.n_features]
        
        return self.selected_features
```

---

## 3. ארכיטקטורות מתקדמות

### 3.1 Ensemble Models

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class EnsembleTradingModel:
    def __init__(self):
        self.models = {}
        self.ensemble = None
        
    def build_voting_ensemble(self):
        """Voting Ensemble - ממוצע החלטות"""
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100),
            'xgb': XGBClassifier(n_estimators=100),
            'lgbm': LGBMClassifier(n_estimators=100)
        }
        
        self.ensemble = VotingClassifier(
            estimators=list(self.models.items()),
            voting='soft'  # weighted by probability
        )
        return self.ensemble
    
    def build_stacking_ensemble(self):
        """Stacking Ensemble - מודל meta"""
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100)),
            ('xgb', XGBClassifier(n_estimators=100)),
            ('lgbm', LGBMClassifier(n_estimators=100))
        ]
        
        self.ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(),
            cv=5
        )
        return self.ensemble
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """אימון והערכה"""
        self.ensemble.fit(X_train, y_train)
        
        y_pred = self.ensemble.predict(X_test)
        y_prob = self.ensemble.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score
        )
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_prob)
        }
        
        return metrics
```

### 3.2 Attention Mechanism למסחר

```python
import torch
import torch.nn as nn

class AttentionTradingModel(nn.Module):
    """מודל עם מנגנון Attention לחיזוי מסחר"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size)
        )
        
    def attention_weights(self, lstm_output):
        """חישוב משקולות Attention"""
        # lstm_output: (batch, seq_len, hidden_size * 2)
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return attention_weights
    
    def forward(self, x):
        # LSTM encoding
        lstm_output, _ = self.lstm(x)  # (batch, seq_len, hidden_size * 2)
        
        # Attention
        attn_weights = self.attention_weights(lstm_output)
        context = torch.sum(attn_weights * lstm_output, dim=1)  # (batch, hidden_size * 2)
        
        # Output
        output = self.fc(context)
        return output, attn_weights

# Usage
model = AttentionTradingModel(
    input_size=50,  # number of features
    hidden_size=128,
    num_layers=2,
    output_size=3  # buy, hold, sell
)
```

---

## 4. טיפים מתקדמים ליישום

### 4.1 מניעת Overfitting

```python
def prevent_overfitting(model, X_train, y_train, X_val, y_val):
    """טכניקות למניעת Overfitting"""
    
    # 1. Early Stopping
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # 2. Cross-Validation
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_t, y_t)
        score = model.score(X_v, y_v)
        scores.append(score)
    
    print(f"CV Scores: {scores}")
    print(f"Mean: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    # 3. Regularization
    # L1/L2 regularization in model definition
    
    # 4. Dropout
    # Already in model architecture
    
    return model
```

### 4.2 Online Learning

```python
class OnlineLearningTrader:
    """מודל שמתעדכן בזמן אמת"""
    
    def __init__(self, base_model, update_frequency=100):
        self.model = base_model
        self.update_frequency = update_frequency
        self.buffer = []
        
    def predict(self, features):
        return self.model.predict(features)
    
    def update(self, features, actual_outcome):
        """עדכון המודל עם נתון חדש"""
        self.buffer.append((features, actual_outcome))
        
        if len(self.buffer) >= self.update_frequency:
            X = np.array([x[0] for x in self.buffer])
            y = np.array([x[1] for x in self.buffer])
            
            # Partial fit (for models that support it)
            if hasattr(self.model, 'partial_fit'):
                self.model.partial_fit(X, y)
            else:
                # Retrain on buffer
                self.model.fit(X, y)
            
            # Clear buffer (or keep some for stability)
            self.buffer = self.buffer[-50:]  # Keep recent samples
```

### 4.3 Model Monitoring

```python
class ModelMonitor:
    """ניטור ביצועי מודל בזמן אמת"""
    
    def __init__(self, model, threshold_drop=0.1):
        self.model = model
        self.threshold_drop = threshold_drop
        self.baseline_accuracy = None
        self.predictions_log = []
        
    def set_baseline(self, X_test, y_test):
        """קביעת baseline לביצועים"""
        predictions = self.model.predict(X_test)
        self.baseline_accuracy = (predictions == y_test).mean()
        print(f"Baseline accuracy: {self.baseline_accuracy:.4f}")
    
    def log_prediction(self, features, prediction, actual):
        """רישום תחזית"""
        self.predictions_log.append({
            'timestamp': pd.Timestamp.now(),
            'prediction': prediction,
            'actual': actual,
            'correct': prediction == actual
        })
    
    def check_performance(self, window=100):
        """בדיקת ביצועים אחרונים"""
        if len(self.predictions_log) < window:
            return None
        
        recent = self.predictions_log[-window:]
        recent_accuracy = sum(p['correct'] for p in recent) / len(recent)
        
        if recent_accuracy < self.baseline_accuracy - self.threshold_drop:
            self._alert_degradation(recent_accuracy)
        
        return recent_accuracy
    
    def _alert_degradation(self, current_accuracy):
        """התראה על ירידה בביצועים"""
        print(f"⚠️ ALERT: Model performance degradation detected!")
        print(f"Current: {current_accuracy:.4f}, Baseline: {self.baseline_accuracy:.4f}")
        print(f"Drop: {(self.baseline_accuracy - current_accuracy) * 100:.1f}%")
        # כאן ניתן להוסיף שליחת אימייל, SMS וכו'
```

---

## סיכום

מסמך זה מספק מדריך מעמיק לשילוב AI ולמידת מכונה במסחר, כולל:

1. **Deep Learning:**
   - LSTM לחיזוי סדרות זמן (93-97% דיוק)
   - CNN לזיהוי תבניות
   - מודלים היברידיים

2. **Reinforcement Learning:**
   - DQN, PPO, A2C
   - סביבות מסחר מותאמות
   - תוצאות מחקריות (+12% רווחיות)

3. **NLP וסנטימנט:**
   - FinBERT לניתוח פיננסי
   - עיבוד חדשות real-time
   - מקרי שימוש מוכחים

4. **Feature Engineering:**
   - Features טכניים (100+)
   - Features זמניים
   - בחירת Features אופטימלית

5. **ארכיטקטורות מתקדמות:**
   - Ensemble Models
   - Attention Mechanisms
   - Online Learning

המפתח להצלחה הוא שילוב נכון של טכנולוגיות, ניהול סיכונים קפדני, ומעקב מתמיד אחר ביצועי המודל.
