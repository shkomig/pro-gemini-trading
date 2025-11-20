🚀 Prompt המשך בנייה - TradingSystem-Pro (V2 - Updated)

**העתק פרומפט זה ל-Gemini/Claude כדי להמשיך לבנות את המערכת**

---

## Context - הקשר

אתה ממשיך פיתוח של מערכת טרייד אלגוריתמית מתקדמת ב-Google Project IDX.

**מה שכבר נעשה ✅:**
- Project structure created in IDX
- Initial config files and documentation created.

**הסכמות חדשות (New Agreements) ✅:**
- **הוספת רכיב Backtesting:** קריטי לבדיקת אסטרטגיות לפני שימוש בכסף אמיתי.
- **הוספת מערכת לוגינג (Logging):** לתיעוד כל פעולות המערכת ב-BigQuery.
- **ניהול קונפיגורציה מתקדם:** שימוש בקבצי YAML ייעודיים לכל רכיב.
- **מחקר אסטרטגיות וסורק:** AI יבצע מחקר ויציע אסטרטגיות וקריטריונים לסורק.
- **יקום מניות:** סריקה דינמית של מניות ארה"ב.

**Status**: 20% complete, Phase 1 (Planning & Core components definition)

---

## System Overview (V2)

**Goal**: Day trading system with a portfolio of tested strategies.

**Tech**: Python 3.11+, ib_insync, Streamlit, GCP (BigQuery, Cloud Functions), Pytest.

**Core Components**:
1.  **IBKR Connector**: חיבור ל-TWS.
2.  **Configuration Manager**: טעינת הגדרות מ-`config/`
3.  **Scanner**: איתור מניות למסחר (Pre-market & Intraday).
4.  **Strategy Engine**: הרצת לוגיקת האסטרטגיות.
5.  **Risk Manager**: ניהול סיכונים גלובלי ופר עסקה.
6.  **Order Executor**: שליחת הוראות מסחר.
7.  **Backtester**: בדיקת אסטרטגיות על נתונים היסטוריים. **(רכיב חדש)**
8.  **Logger**: תיעוד פעולות והחלטות ל-BigQuery. **(רכיב חדש)**
9.  **Dashboard**: ממשק משתמש ב-Streamlit.

---

## Researched Strategies & Scanner Criteria

**Strategies (מחקר ראשוני):**
1.  **Opening Range Breakout (ORB)**: פריצת טווח הפתיחה (15-30 דקות) עם ווליום גבוה.
2.  **Mean Reversion (Bollinger Bands)**: כניסה נגד המגמה כאשר המחיר מגיע לקצוות רצועות בולינגר, בציפייה לחזרה לממוצע.
3.  **Trend Following (Moving Average Crossover)**: כניסה עם המגמה כאשר ממוצע נע מהיר חוצה מעל ממוצע נע איטי.
4.  **VWAP + Momentum**: כניסה ללונג כשהמחיר מעל VWAP, מומנטום חיובי ו-RSI מעל 50.
5.  **Gap & Go**: קניית מניות שפתחו בפער מחירים (Gap Up) משמעותי עם ווליום גבוה לפני השוק, ופורצות את שיא טרום השוק.

**Scanner Criteria (מחקר ראשוני):**
-   **Universe**: US Stocks (NYSE, NASDAQ).
-   **Price**: $10 - $200 (סינון מניות זולות ותנודתיות מדי, ויקרות מדי).
-   **Volume (Daily Avg)**: > 1,000,000 מניות (נזילות גבוהה).
-   **Pre-Market Scanner:**
    *   **Gap %**: > 2% (Up or Down).
    *   **Pre-Market Volume**: > 100,000 מניות.
    *   **Relative Volume (RVOL)**: > 3 (ווליום גבוה מהרגיל).
-   **Intraday Scanner:**
    *   זיהוי מניות שמתקרבות לקריטריונים של האסטרטגיות (למשל, חציית VWAP, הגעה לרצועת בולינגר).

---

## Your Tasks - המשימות שלך (Phase 1)

### 1. Create New Config Files (בוצע)
- `config/strategies.yaml`
- `config/scanner.yaml`
- `config/logging.yaml`

### 2. Implement the Logger (`src/logger/`)
- Create `logger.py` to configure a standard Python logger.
- Add basic GCP integration to log to BigQuery in a later phase.

### 3. Implement the Backtester (`src/backtester/`)
- Create `backtester.py`.
- Function to load historical data (e.g., from a CSV or IBKR).
- Loop through data and simulate strategy execution.
- Calculate and return key metrics (e.g., Net PnL, Win Rate, Max Drawdown).

### 4. Refine IBKR Connector (`src/connectors/ibkr_connector.py`)
- Add error handling and reconnection logic.

### 5. Start ORB Strategy (`src/strategies/orb_strategy.py`)
- Implement the basic logic based on the 15-min opening range.

### 6. Update `main.py`
- Add logic to initialize the logger and other core components.

---

## Critical Rules & Best Practices

- **Backtest Everything**: No strategy goes to production without a positive backtest.
- **Config-Driven**: All strategy and scanner parameters MUST be loaded from YAML files.
- **Modular Design**: Keep components decoupled for easier testing and maintenance.
- **Risk First**: Risk management rules are global and override any strategy signal.

---

## User Info

- **Level**: Advanced (Python, Docker, GCP)
- **Location**: Israel (IST)
- **Trading**: US stocks via IBKR
- **Wants**: See a robust, well-engineered system, starting with a working ORB strategy and backtester.
- **Style**: Prefers quality and robustness over speed. Appreciates proactive suggestions and a solid plan.

---

## Let's Build! 🚀

**Next Step**: Start by creating the new config files (`strategies.yaml`, `scanner.yaml`, `logging.yaml`). Then, implement the initial version of the `Logger`.

Good luck! 💪
