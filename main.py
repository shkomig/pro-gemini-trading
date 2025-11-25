"""
Trading System Pro - V3.3
=========================
Smart Stock Selection with Multi-Factor Scoring

V3.3 Features:
- Enhanced multi-scan (Gap Up, Hot Volume, Most Active)
- Stock scoring system (Gap%, RVOL, ATR, Momentum, Signals)
- Signal consensus from multiple strategies
- Ranks and selects TOP 5 best opportunities

Author: Trading System Pro
Version: 3.3
"""

import yaml
import pandas as pd
import time
from typing import Dict, List
from src.logger.logger import log
from src.ib_client.ib_client import IBClient
from src.strategy.ma_crossover_strategy import MACrossoverStrategy
from src.strategy.bollinger_bands_strategy import BollingerBandsStrategy
from src.strategy.orb_strategy import ORBStrategy
from src.strategy.vwap_strategy import VWAPStrategy
from src.strategy.volume_breakout_strategy import VolumeBreakoutStrategy
from src.trade_manager.trade_manager import TradeManager
from src.scanner.scanner import Scanner
from src.scoring.stock_scorer import StockScorer, SignalRanker, StockScore

def load_strategies(strategy_config_path: str) -> list:
    """Loads trading strategies from a config file."""
    with open(strategy_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    strategies = []
    for strategy_definition in config['strategies']:
        strategy_type = strategy_definition['type']
        params = {
            'name': strategy_definition['name'],
            **strategy_definition['parameters']
        }

        if strategy_type == "MACrossoverStrategy":
            strategies.append(MACrossoverStrategy(**params))
        elif strategy_type == "BollingerBandsStrategy":
            strategies.append(BollingerBandsStrategy(**params))
        elif strategy_type == "ORBStrategy":
            strategies.append(ORBStrategy(**params))
        elif strategy_type == "VWAPStrategy":
            strategies.append(VWAPStrategy(**params))
        elif strategy_type == "VolumeBreakoutStrategy":
            strategies.append(VolumeBreakoutStrategy(**params))
        else:
            log.warning(f"Strategy type '{strategy_type}' is not supported.")
            
    return strategies


def analyze_stock(symbol: str, data: pd.DataFrame, strategies: list, 
                  scorer: StockScorer, ranker: SignalRanker) -> tuple:
    """
    Analyze a single stock with all strategies.
    
    Returns:
        Tuple of (StockScore, signals_by_strategy dict, consensus_signal)
    """
    signals_by_strategy: Dict[str, pd.DataFrame] = {}
    
    # Run all strategies
    for strategy in strategies:
        try:
            signals_df = strategy.generate_signals(data.copy())
            signals_by_strategy[strategy.name] = signals_df
        except Exception as e:
            log.warning(f"Strategy {strategy.name} failed on {symbol}: {e}")
            signals_by_strategy[strategy.name] = pd.DataFrame()
    
    # Count buy signals
    buy_count = ranker.count_buy_signals(signals_by_strategy)
    
    # Score the stock
    score = scorer.score_stock(symbol, data, signal_count=buy_count)
    
    # Get consensus signal
    consensus = ranker.get_consensus_signal(signals_by_strategy, min_agreement=2)
    
    return score, signals_by_strategy, consensus

def main():
    log.info("=" * 60)
    log.info("Initializing Trading System Pro V3.3 - Smart Stock Selection")
    log.info("=" * 60)

    # --- IB Client Setup ---
    with open("config/ib_config.yaml", 'r') as f:
        ib_config = yaml.safe_load(f)
    
    connection_params = ib_config['connection']
    simulation_mode = connection_params.pop('simulation_mode', False)
    ib_client = IBClient(**connection_params, simulation_mode=simulation_mode)
    data_request_params = ib_config['data_request']
    
    # Set contract details from config
    ib_client.set_contract_details(
        sec_type=data_request_params.get('sec_type', 'STK'),
        exchange=data_request_params.get('exchange', 'SMART'),
        currency=data_request_params.get('currency', 'USD')
    )

    try:
        ib_client.connect()
        trade_manager = TradeManager(ib_client)
        scanner = Scanner(ib_client, "config/scanner.yaml")
        
        # --- V3.3: Initialize Scoring System ---
        scorer = StockScorer(ib_client)
        ranker = SignalRanker()
        
        # --- Strategy Loading ---
        strategies = load_strategies("config/strategy_config.yaml")
        if not strategies:
            log.warning("No strategies loaded. Exiting.")
            return
        
        log.info(f"Loaded {len(strategies)} strategies")

        log.info("Starting continuous trading loop. Press Ctrl+C to stop.")
        while True:
            try:
                if not ib_client.ib.isConnected():
                    log.warning("Connection lost. Attempting to reconnect...")
                    try:
                        ib_client.connect()
                        log.info("Reconnected successfully.")
                    except Exception as conn_err:
                        log.error(f"Reconnection failed: {conn_err}")
                        time.sleep(30)
                        continue

                # --- V3.3: Enhanced Scanning ---
                log.info("-" * 40)
                log.info("[V3.3] Starting smart stock selection cycle")
                log.info("-" * 40)
                
                symbols = scanner.scan_market()
                if not symbols:
                    log.warning("Scanner returned no symbols. Using defaults.")
                    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
                
                # Limit analysis to top 30 candidates (from scanner priority order)
                symbols_to_analyze = symbols[:30]
                log.info(f"[V3.3] Analyzing top {len(symbols_to_analyze)} candidates from scan")
                
                # --- V3.3: Score and Rank All Candidates ---
                stock_scores: List[StockScore] = []
                stock_data: Dict[str, pd.DataFrame] = {}
                stock_signals: Dict[str, Dict[str, pd.DataFrame]] = {}
                stock_consensus: Dict[str, int] = {}
                
                for symbol in symbols_to_analyze:
                    # Get historical data
                    historical_data = ib_client.get_historical_data(
                        symbol=symbol,
                        duration=data_request_params['duration'],
                        bar_size=data_request_params['bar_size']
                    )
                    
                    if historical_data.empty:
                        log.debug(f"No data for {symbol}, skipping")
                        continue
                    
                    # Price filter
                    latest_close = historical_data['close'].iloc[-1]
                    universe_config = scanner.config.get('universe', {})
                    min_price = universe_config.get('min_price', 0)
                    max_price = universe_config.get('max_price', float('inf'))
                    
                    if not (min_price <= latest_close <= max_price):
                        log.debug(f"Skipping {symbol}: Price ${latest_close:.2f} out of range")
                        continue
                    
                    # Analyze with all strategies and score
                    score, signals, consensus = analyze_stock(
                        symbol, historical_data, strategies, scorer, ranker
                    )
                    
                    stock_scores.append(score)
                    stock_data[symbol] = historical_data
                    stock_signals[symbol] = signals
                    stock_consensus[symbol] = consensus
                
                # --- V3.3: Select TOP 5 Best Opportunities ---
                log.info("=" * 40)
                log.info("[V3.3] RANKING RESULTS")
                log.info("=" * 40)
                
                top_stocks = scorer.rank_stocks(
                    stock_scores, 
                    min_score=25.0,  # Minimum 25 points to qualify
                    top_n=5
                )
                
                if not top_stocks:
                    log.warning("[V3.3] No stocks qualified above minimum score threshold")
                else:
                    log.info(f"[V3.3] TOP {len(top_stocks)} SELECTED:")
                    for i, score in enumerate(top_stocks, 1):
                        log.info(f"  #{i} {score.symbol}: {score.total_score:.1f} pts | "
                                f"Gap={score.gap_percent:+.1f}% | RVOL={score.rvol:.1f}x | "
                                f"Signals={score.signal_count}")
                    
                    # --- Execute trades only for top stocks with buy consensus ---
                    for score in top_stocks:
                        symbol = score.symbol
                        consensus = stock_consensus.get(symbol, 0)
                        
                        if consensus == 1:  # BUY consensus
                            log.info(f"[V3.3] ✅ {symbol} has BUY consensus - executing trade")
                            
                            # Use the signals from the first agreeing strategy
                            for strategy_name, signals_df in stock_signals[symbol].items():
                                if not signals_df.empty and signals_df['signal'].iloc[-1] == 1:
                                    trade_manager.process_signals(signals_df, symbol, strategy_name)
                                    break  # Only one trade per symbol
                        else:
                            log.info(f"[V3.3] ⏳ {symbol} scored high but no BUY consensus yet")
                
                log.info("-" * 40)
                log.info("Sleeping for 10 seconds...")
                log.info("-" * 40)
                ib_client.ib.sleep(10)

            except Exception as e:
                log.error(f"An error occurred during the loop: {e}")
                import traceback
                log.error(traceback.format_exc())
                ib_client.ib.sleep(10)

    except KeyboardInterrupt:
        log.info("User stopped the trading loop.")
    except ConnectionError as e:
        log.critical(f"A critical error occurred: {e}")
    finally:
        if ib_client.ib.isConnected():
            ib_client.disconnect()

    log.info("Trading System Pro V3.3 finished.")

if __name__ == "__main__":
    main()
