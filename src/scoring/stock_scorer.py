"""
Stock Scorer Module - V3.3
==========================
Multi-factor scoring system to identify the BEST trading opportunities.

Scoring Factors (Total: 100 points):
- Gap %: 20 points - Gap from previous close
- Relative Volume (RVOL): 25 points - Volume vs average
- ATR Strength: 15 points - Daily volatility potential  
- Price Momentum: 20 points - 5-bar momentum
- Signal Agreement: 20 points - Strategy consensus

Author: Trading System Pro
Version: 3.3
"""

import pandas as pd
import numpy as np
import pandas_ta as pta
from dataclasses import dataclass
from typing import List, Dict, Optional
from src.logger.logger import log


@dataclass
class StockScore:
    """Holds the scoring breakdown for a single stock."""
    symbol: str
    gap_score: float = 0.0
    rvol_score: float = 0.0
    atr_score: float = 0.0
    momentum_score: float = 0.0
    signal_score: float = 0.0
    total_score: float = 0.0
    
    # Raw values for logging
    gap_percent: float = 0.0
    rvol: float = 0.0
    atr_percent: float = 0.0
    momentum_percent: float = 0.0
    signal_count: int = 0
    
    def __repr__(self):
        return (f"StockScore({self.symbol}: Total={self.total_score:.1f} | "
                f"Gap={self.gap_score:.1f}({self.gap_percent:+.2f}%) | "
                f"RVOL={self.rvol_score:.1f}({self.rvol:.1f}x) | "
                f"ATR={self.atr_score:.1f}({self.atr_percent:.2f}%) | "
                f"Mom={self.momentum_score:.1f}({self.momentum_percent:+.2f}%) | "
                f"Sig={self.signal_score:.1f}({self.signal_count}))")


class StockScorer:
    """
    Multi-factor Stock Scoring System.
    
    Evaluates stocks based on multiple factors to identify
    the best trading opportunities.
    """
    
    # Scoring weights (total = 100)
    WEIGHTS = {
        'gap': 20,
        'rvol': 25,
        'atr': 15,
        'momentum': 20,
        'signal': 20
    }
    
    # Thresholds for scoring
    GAP_THRESHOLDS = {
        'min': 1.0,      # Minimum gap % to consider
        'good': 3.0,     # Good gap %
        'excellent': 5.0  # Excellent gap %
    }
    
    RVOL_THRESHOLDS = {
        'min': 1.5,      # Minimum relative volume
        'good': 3.0,     # Good RVOL
        'excellent': 5.0  # Excellent RVOL
    }
    
    ATR_THRESHOLDS = {
        'min': 2.0,      # Minimum ATR % of price
        'good': 4.0,     # Good ATR %
        'excellent': 6.0  # Excellent ATR %
    }
    
    MOMENTUM_THRESHOLDS = {
        'min': 0.5,      # Minimum momentum %
        'good': 1.5,     # Good momentum %
        'excellent': 3.0  # Excellent momentum %
    }

    def __init__(self, ib_client=None):
        """Initialize the stock scorer."""
        self.ib_client = ib_client
        log.info("[SCORER] Stock Scorer V3.3 initialized")

    def calculate_gap_percent(self, data: pd.DataFrame) -> float:
        """
        Calculate the gap percentage from previous close.
        
        Gap % = (Today Open - Yesterday Close) / Yesterday Close * 100
        """
        if len(data) < 2:
            return 0.0
        
        # Get today's open and yesterday's close
        today_open = data['open'].iloc[-1]
        yesterday_close = data['close'].iloc[-2]
        
        if yesterday_close == 0:
            return 0.0
            
        gap_percent = ((today_open - yesterday_close) / yesterday_close) * 100
        return gap_percent

    def calculate_relative_volume(self, data: pd.DataFrame, period: int = 20) -> float:
        """
        Calculate relative volume (RVOL).
        
        RVOL = Current Volume / Average Volume
        """
        if len(data) < period:
            return 1.0
        
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].iloc[-period:-1].mean()
        
        if avg_volume == 0:
            return 1.0
            
        rvol = current_volume / avg_volume
        return rvol

    def calculate_atr_percent(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate ATR as percentage of price.
        
        ATR % = (ATR / Current Price) * 100
        """
        if len(data) < period:
            return 0.0
        
        # Calculate ATR using pandas_ta
        atr = pta.atr(data['high'], data['low'], data['close'], length=period)
        
        if atr is None or atr.empty:
            return 0.0
            
        current_atr = atr.iloc[-1]
        current_price = data['close'].iloc[-1]
        
        if current_price == 0 or pd.isna(current_atr):
            return 0.0
            
        atr_percent = (current_atr / current_price) * 100
        return atr_percent

    def calculate_momentum(self, data: pd.DataFrame, period: int = 5) -> float:
        """
        Calculate price momentum over n bars.
        
        Momentum % = (Current Close - Close n bars ago) / Close n bars ago * 100
        """
        if len(data) < period + 1:
            return 0.0
        
        current_close = data['close'].iloc[-1]
        past_close = data['close'].iloc[-period-1]
        
        if past_close == 0:
            return 0.0
            
        momentum_percent = ((current_close - past_close) / past_close) * 100
        return momentum_percent

    def _score_linear(self, value: float, thresholds: dict, max_score: float, 
                      use_abs: bool = True) -> float:
        """
        Calculate a linear score based on thresholds.
        
        Score scales linearly from 0 at min to max_score at excellent.
        """
        if use_abs:
            value = abs(value)
            
        if value < thresholds['min']:
            return 0.0
        elif value >= thresholds['excellent']:
            return max_score
        elif value >= thresholds['good']:
            # Linear interpolation between good and excellent
            range_size = thresholds['excellent'] - thresholds['good']
            progress = (value - thresholds['good']) / range_size
            return max_score * (0.7 + 0.3 * progress)
        else:
            # Linear interpolation between min and good
            range_size = thresholds['good'] - thresholds['min']
            progress = (value - thresholds['min']) / range_size
            return max_score * (0.3 + 0.4 * progress)

    def score_stock(self, symbol: str, data: pd.DataFrame, 
                    signal_count: int = 0) -> StockScore:
        """
        Calculate the complete score for a stock.
        
        Args:
            symbol: Stock ticker
            data: OHLCV DataFrame
            signal_count: Number of strategies giving buy signal (0-5)
            
        Returns:
            StockScore with complete breakdown
        """
        score = StockScore(symbol=symbol)
        
        if data is None or data.empty or len(data) < 20:
            log.warning(f"[SCORER] {symbol}: Insufficient data for scoring")
            return score
        
        try:
            # 1. Gap Score (20 points)
            score.gap_percent = self.calculate_gap_percent(data)
            score.gap_score = self._score_linear(
                score.gap_percent, 
                self.GAP_THRESHOLDS, 
                self.WEIGHTS['gap'],
                use_abs=True  # Bullish or bearish gap both get points
            )
            
            # 2. Relative Volume Score (25 points)
            score.rvol = self.calculate_relative_volume(data)
            score.rvol_score = self._score_linear(
                score.rvol, 
                self.RVOL_THRESHOLDS, 
                self.WEIGHTS['rvol'],
                use_abs=False
            )
            
            # 3. ATR Score (15 points)
            score.atr_percent = self.calculate_atr_percent(data)
            score.atr_score = self._score_linear(
                score.atr_percent, 
                self.ATR_THRESHOLDS, 
                self.WEIGHTS['atr'],
                use_abs=False
            )
            
            # 4. Momentum Score (20 points)
            score.momentum_percent = self.calculate_momentum(data)
            # Only positive momentum for long trades
            if score.momentum_percent > 0:
                score.momentum_score = self._score_linear(
                    score.momentum_percent, 
                    self.MOMENTUM_THRESHOLDS, 
                    self.WEIGHTS['momentum'],
                    use_abs=False
                )
            else:
                score.momentum_score = 0.0
            
            # 5. Signal Agreement Score (20 points)
            score.signal_count = signal_count
            # Max 5 strategies, so each agreement = 4 points
            score.signal_score = min(signal_count * 4, self.WEIGHTS['signal'])
            
            # Calculate total score
            score.total_score = (
                score.gap_score + 
                score.rvol_score + 
                score.atr_score + 
                score.momentum_score + 
                score.signal_score
            )
            
            log.info(f"[SCORER] {score}")
            
        except Exception as e:
            log.error(f"[SCORER] Error scoring {symbol}: {e}")
            
        return score

    def rank_stocks(self, scores: List[StockScore], 
                    min_score: float = 30.0,
                    top_n: int = 5) -> List[StockScore]:
        """
        Rank stocks by score and return the top N.
        
        Args:
            scores: List of StockScore objects
            min_score: Minimum score threshold
            top_n: Number of top stocks to return
            
        Returns:
            Top N stocks sorted by score (descending)
        """
        # Filter by minimum score
        qualified = [s for s in scores if s.total_score >= min_score]
        
        # Sort by total score (descending)
        ranked = sorted(qualified, key=lambda x: x.total_score, reverse=True)
        
        # Return top N
        top_stocks = ranked[:top_n]
        
        log.info(f"[SCORER] Ranked {len(qualified)}/{len(scores)} stocks above {min_score} threshold")
        for i, stock in enumerate(top_stocks, 1):
            log.info(f"[SCORER] #{i}: {stock.symbol} = {stock.total_score:.1f} points")
            
        return top_stocks


class SignalRanker:
    """
    Ranks and filters trading signals from multiple strategies.
    """
    
    def __init__(self):
        log.info("[RANKER] Signal Ranker V3.3 initialized")

    def count_buy_signals(self, signals_by_strategy: Dict[str, pd.DataFrame]) -> int:
        """
        Count how many strategies are giving buy signals.
        
        Args:
            signals_by_strategy: Dict of strategy_name -> signals DataFrame
            
        Returns:
            Number of strategies with signal == 1 (buy) on last bar
        """
        buy_count = 0
        
        for strategy_name, signals_df in signals_by_strategy.items():
            if signals_df is not None and not signals_df.empty:
                last_signal = signals_df['signal'].iloc[-1]
                if last_signal == 1:
                    buy_count += 1
                    log.debug(f"[RANKER] {strategy_name}: BUY signal")
                    
        return buy_count

    def get_consensus_signal(self, signals_by_strategy: Dict[str, pd.DataFrame],
                             min_agreement: int = 2) -> int:
        """
        Get consensus signal from multiple strategies.
        
        Args:
            signals_by_strategy: Dict of strategy_name -> signals DataFrame
            min_agreement: Minimum strategies that must agree
            
        Returns:
            1 for buy, -1 for sell, 0 for no consensus
        """
        buy_count = 0
        sell_count = 0
        
        for strategy_name, signals_df in signals_by_strategy.items():
            if signals_df is not None and not signals_df.empty:
                last_signal = signals_df['signal'].iloc[-1]
                if last_signal == 1:
                    buy_count += 1
                elif last_signal == -1:
                    sell_count += 1
        
        if buy_count >= min_agreement:
            log.info(f"[RANKER] BUY consensus: {buy_count} strategies agree")
            return 1
        elif sell_count >= min_agreement:
            log.info(f"[RANKER] SELL consensus: {sell_count} strategies agree")
            return -1
        else:
            log.debug(f"[RANKER] No consensus: {buy_count} buy, {sell_count} sell")
            return 0
