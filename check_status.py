from src.ib_client.ib_client import IBClient
from src.logger.logger import log
import yaml
import time

def check_status():
    log.info("Checking account status...")

    with open("config/ib_config.yaml", 'r') as f:
        ib_config = yaml.safe_load(f)
    
    connection_params = ib_config['connection']
    simulation_mode = connection_params.pop('simulation_mode', False)
    
    ib_client = IBClient(**connection_params, simulation_mode=simulation_mode)
    
    try:
        ib_client.connect()
        time.sleep(1)

        positions = ib_client.ib.positions()
        open_orders = ib_client.ib.openOrders()
        
        print("\n--- POSITIONS ---")
        for p in positions:
            if p.position != 0:
                print(f"{p.contract.symbol}: {p.position}")
                
        print("\n--- OPEN ORDERS ---")
        for o in open_orders:
            print(f"{o.contract.symbol} {o.action} {o.totalQuantity} - Status: {o.orderState.status if hasattr(o, 'orderState') else 'Unknown'}")

    except Exception as e:
        log.error(f"Error: {e}")
    finally:
        ib_client.disconnect()

if __name__ == "__main__":
    check_status()
