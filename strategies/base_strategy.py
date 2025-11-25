"""
Base Strategy Class
==================
Abstract base class for all trading strategies.
Provides common interface and utility methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """סוגי סיגנלים"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


class SignalStrength(Enum):
    """חוזק הסיגנל"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3


@dataclass
class TradingSignal:
    """מבנה סיגנל מסחר"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    price: float
    strategy_name: str
    
    # Entry details
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Risk management
    position_size: Optional[int] = None
    risk_amount: Optional[float] = None
    risk_percent: Optional[float] = None
    
    # Additional context
    indicators: Optional[Dict] = None
    reason: Optional[str] = None
    confidence: Optional[float] = None
    
    def __str__(self):
        return (f"{self.signal_type.value} {self.symbol} @ ${self.price:.2f} "
                f"[{self.strength.value}] - {self.strategy_name}")


class BaseStrategy(ABC):
    """
    מחלקת בסיס לכל האסטרטגיות
    
    כל אסטרטגיה צריכה לממש:
    - analyze(): ניתוח הנתונים וחישוב אינדיקטורים
    - generate_signals(): יצירת סיגנלי קנייה/מכירה
    """
    
    def __init__(self, name: str, config: Dict):
        """
        אתחול אסטרטגיה
        
        Args:
            name: שם האסטרטגיה
            config: קונפיגורציה מקובץ YAML
        """
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        
        # Risk management parameters
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2%
        self.stop_loss_percent = config.get('stop_loss_percent', 0.01)  # 1%
        
        # State tracking
        self.last_signal: Optional[TradingSignal] = None
        self.signals_history: List[TradingSignal] = []
        
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ניתוח הנתונים וחישוב אינדיקטורים ספציפיים לאסטרטגיה
        
        Args:
            data: DataFrame עם נתוני OHLCV
            
        Returns:
            DataFrame עם אינדיקטורים מחושבים
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        יצירת סיגנלי מסחר על בסיס האינדיקטורים
        
        Args:
            data: DataFrame עם נתונים ואינדיקטורים
            
        Returns:
            רשימת סיגנלים
        """
        pass
    
    def calculate_stop_loss(self, entry_price: float, signal_type: SignalType,
                           atr: Optional[float] = None) -> float:
        """
        חישוב רמת Stop Loss
        
        Args:
            entry_price: מחיר כניסה
            signal_type: סוג הסיגנל (BUY/SELL)
            atr: Average True Range (אופציונלי)
            
        Returns:
            מחיר Stop Loss
        """
        if signal_type == SignalType.BUY:
            # For long positions: stop below entry
            if atr is not None:
                # ATR-based stop: entry - 2*ATR
                stop = entry_price - (2 * atr)
            else:
                # Percentage-based stop
                stop = entry_price * (1 - self.stop_loss_percent)
        else:
            # For short positions: stop above entry
            if atr is not None:
                stop = entry_price + (2 * atr)
            else:
                stop = entry_price * (1 + self.stop_loss_percent)
                
        return round(stop, 2)
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                             signal_type: SignalType, 
                             risk_reward_ratio: float = 2.0) -> float:
        """
        חישוב רמת Take Profit על בסיס יחס סיכון-תשואה
        
        Args:
            entry_price: מחיר כניסה
            stop_loss: מחיר Stop Loss
            signal_type: סוג הסיגנל
            risk_reward_ratio: יחס סיכון-תשואה רצוי (ברירת מחדל: 2.0)
            
        Returns:
            מחיר Take Profit
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if signal_type == SignalType.BUY:
            take_profit = entry_price + reward
        else:
            take_profit = entry_price - reward
            
        return round(take_profit, 2)
    
    def calculate_position_size(self, account_balance: float, entry_price: float,
                               stop_loss: float, max_position_value: Optional[float] = None) -> int:
        """
        חישוב גודל פוזיציה על בסיס ניהול סיכונים
        
        Args:
            account_balance: יתרת חשבון
            entry_price: מחיר כניסה
            stop_loss: מחיר Stop Loss
            max_position_value: ערך מקסימלי לפוזיציה (אופציונלי)
            
        Returns:
            מספר מניות
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        # Maximum amount to risk on this trade
        max_risk_amount = account_balance * self.max_risk_per_trade
        
        # Calculate shares based on risk
        shares = int(max_risk_amount / risk_per_share)
        
        # Apply position value limit if specified
        if max_position_value is not None:
            max_shares_by_value = int(max_position_value / entry_price)
            shares = min(shares, max_shares_by_value)
        
        # Ensure at least 1 share if risk allows
        return max(1, shares)
    
    def validate_signal(self, signal: TradingSignal, 
                       current_positions: Optional[Dict] = None) -> bool:
        """
        בדיקת תקינות סיגנל לפני ביצוע
        
        Args:
            signal: הסיגנל לבדיקה
            current_positions: פוזיציות פתוחות נוכחיות
            
        Returns:
            True אם הסיגנל תקין
        """
        # Basic validation
        if signal.price <= 0:
            return False
            
        if signal.position_size is not None and signal.position_size <= 0:
            return False
        
        # Check if we already have a position
        if current_positions and signal.symbol in current_positions:
            current_pos = current_positions[signal.symbol]
            
            # Don't open new long if already long
            if signal.signal_type == SignalType.BUY and current_pos > 0:
                return False
                
            # Don't open new short if already short
            if signal.signal_type == SignalType.SELL and current_pos < 0:
                return False
        
        return True
    
    def get_signal_strength(self, confidence: float) -> SignalStrength:
        """
        קביעת חוזק הסיגנל על בסיס רמת הביטחון
        
        Args:
            confidence: רמת ביטחון (0-1)
            
        Returns:
            חוזק הסיגנל
        """
        if confidence >= 0.75:
            return SignalStrength.STRONG
        elif confidence >= 0.5:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def add_signal_to_history(self, signal: TradingSignal):
        """הוספת סיגנל להיסטוריה"""
        self.signals_history.append(signal)
        self.last_signal = signal
    
    def get_recent_signals(self, n: int = 10) -> List[TradingSignal]:
        """
        קבלת הסיגנלים האחרונים
        
        Args:
            n: מספר סיגנלים
            
        Returns:
            רשימת סיגנלים אחרונים
        """
        return self.signals_history[-n:]
    
    def reset(self):
        """איפוס מצב האסטרטגיה"""
        self.last_signal = None
        self.signals_history.clear()
    
    def __repr__(self):
        status = "Enabled" if self.enabled else "Disabled"
        return f"{self.name} Strategy [{status}]"
