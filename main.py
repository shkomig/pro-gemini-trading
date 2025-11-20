import yaml
import pandas as pd
from src.logger.logger import log
from src.ib_client.ib_client import IBClient
from src.strategy.ma_crossover_strategy import MACrossoverStrategy
from src.trade_manager.trade_manager import TradeManager

def load_strategies(strategy_config_path: str) -> list:
    """Loads trading strategies from a config file."""
    with open(strategy_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    strategies = []
    for strategy_definition in config['strategies']:
        strategy_type = strategy_definition['type']
        if strategy_type == "MACrossoverStrategy":
            params = {
                'name': strategy_definition['name'],
                **strategy_definition['parameters']
            }
            strategies.append(MACrossoverStrategy(**params))
        else:
            log.warning(f"Strategy type '{strategy_type}' is not supported.")
            
    return strategies

def main():
    log.info("Initializing Trading System Pro...")

    # --- IB Client Setup ---
    with open("config/ib_config.yaml", 'r') as f:
        ib_config = yaml.safe_load(f)
    
    connection_params = ib_config['connection']
    simulation_mode = connection_params.pop('simulation_mode', False)
    ib_client = IBClient(**connection_params, simulation_mode=simulation_mode)
    data_request_params = ib_config['data_request']
    symbol = data_request_params['symbol'] # Get the symbol for the trade manager

    try:
        ib_client.connect()
        trade_manager = TradeManager(ib_client)

        # --- Data Loading ---
        log.info(f"Requesting historical data for {symbol}...")
        historical_data = ib_client.get_historical_data(**data_request_params)

        if historical_data.empty:
            log.warning("No historical data received. Exiting.")
            return

        # --- Strategy Loading and Execution ---
        strategies = load_strategies("config/strategy_config.yaml")
        if not strategies:
            log.warning("No strategies loaded. Exiting.")
            return

        for strategy in strategies:
            log.info(f"Running strategy: {strategy.name}")
            signals_df = strategy.generate_signals(historical_data.copy()) 
            log.info(f"Generated signals:\n{signals_df.tail()}")
            trade_manager.process_signals(signals_df, symbol)

    except ConnectionError as e:
        log.critical(f"A critical error occurred: {e}")
    finally:
        if ib_client.ib.isConnected():
            ib_client.disconnect()

    log.info("Trading System Pro finished.")

if __name__ == "__main__":
    main()
