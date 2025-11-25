from src.ib_client.ib_client import IBClient
from src.logger.logger import log
import yaml
import time

def check_positions():
    log.info("Checking open positions...")

    with open("config/ib_config.yaml", 'r') as f:
        ib_config = yaml.safe_load(f)
    
    connection_params = ib_config['connection']
    simulation_mode = connection_params.pop('simulation_mode', False)
    
    # Use a different Client ID
    connection_params['client_id'] = 997
    
    ib_client = IBClient(**connection_params, simulation_mode=simulation_mode)
    
    try:
        ib_client.connect()
        time.sleep(1)

        # V3.1: Force refresh positions from IB Gateway
        ib_client.ib.reqPositions()
        time.sleep(1)  # Wait for position update

        positions = ib_client.ib.positions()
        if not positions:
            log.info("No open positions found.")
        else:
            log.info(f"Found {len(positions)} open positions:")
            for p in positions:
                log.info(f"Symbol: {p.contract.symbol}, Position: {p.position}, Avg Cost: {p.avgCost}")

        orders = ib_client.ib.openOrders()
        if not orders:
            log.info("No open orders found.")
        else:
            log.info(f"Found {len(orders)} open orders:")
            for o in orders:
                log.info(f"Order: {o}")

    except Exception as e:
        log.error(f"Error checking positions: {e}")
    finally:
        ib_client.disconnect()

if __name__ == "__main__":
    check_positions()
