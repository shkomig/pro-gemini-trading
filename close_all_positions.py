from src.ib_client.ib_client import IBClient
from src.logger.logger import log
from ib_insync import MarketOrder, LimitOrder, Stock, Contract
import yaml
import time
import math

def close_all_positions():
    log.info("Starting process to close ALL positions...")

    with open("config/ib_config.yaml", 'r') as f:
        ib_config = yaml.safe_load(f)
    
    connection_params = ib_config['connection']
    simulation_mode = connection_params.pop('simulation_mode', False)
    
    # Use a different Client ID to avoid conflict with the main bot
    connection_params['client_id'] = 999
    
    ib_client = IBClient(**connection_params, simulation_mode=simulation_mode)
    
    try:
        ib_client.connect()
        time.sleep(1)

        # 1. Cancel all open orders first
        log.info("Cancelling all open orders...")
        ib_client.ib.reqGlobalCancel()
        time.sleep(2)

        positions = ib_client.ib.positions()
        if not positions:
            log.info("No open positions found.")
            return

        log.info(f"Found {len(positions)} open positions. Closing them now...")

        for p in positions:
            # Create a fresh contract for SMART routing to avoid "Direct Routing" errors
            # We use the symbol from the position, but force exchange='SMART'
            contract = Stock(p.contract.symbol, 'SMART', 'USD')
            
            # Qualify contract to get conId etc
            ib_client.ib.qualifyContracts(contract)

            # Get current market price for Limit Order
            # We use Limit orders with aggressive pricing to ensure execution in Extended Hours
            ticker = ib_client.ib.reqMktData(contract, '', False, False)
            
            # Wait for data
            start_time = time.time()
            while (ticker.bid != ticker.bid or ticker.ask != ticker.ask) and time.time() - start_time < 2:
                ib_client.ib.sleep(0.1)
            
            action = 'SELL' if p.position > 0 else 'BUY'
            quantity = abs(p.position)
            
            # Determine Price
            # If Selling: Sell at Bid (or slightly lower to ensure fill)
            # If Buying: Buy at Ask (or slightly higher)
            price = 0.0
            current_price = ticker.last if (ticker.last and not math.isnan(ticker.last)) else ticker.close
            
            log.info(f"Ticker for {contract.symbol}: Bid={ticker.bid}, Ask={ticker.ask}, Last={ticker.last}, Close={ticker.close}")

            if action == 'SELL':
                # Try to get a valid bid, else use last/close - 5%
                ref_price = ticker.bid if (ticker.bid and not math.isnan(ticker.bid) and ticker.bid > 0) else current_price
                price = round(ref_price * 0.95, 2) # 5% below to ensure fill
            else:
                # Try to get a valid ask, else use last/close + 5%
                ref_price = ticker.ask if (ticker.ask and not math.isnan(ticker.ask) and ticker.ask > 0) else current_price
                price = round(ref_price * 1.05, 2) # 5% above to ensure fill

            # FORCE MARKET ORDER FOR NOW TO ENSURE CLOSE
            log.info(f"Attempting Market Order for {contract.symbol}...")
            order = MarketOrder(action, quantity)
            
            # If you really want Limit, uncomment below and comment out MarketOrder
            # if price == 0 or math.isnan(price):
            #     log.warning(f"Could not determine price for {contract.symbol}. Using Market Order as fallback.")
            #     order = MarketOrder(action, quantity)
            # else:
            #     log.info(f"Closing {contract.symbol}: {action} {quantity} @ Limit {price} (Ref: {current_price})")
            #     order = LimitOrder(action, quantity, price)

            order.outsideRth = True
            order.tif = 'GTC' # Good Till Cancelled

            trade = ib_client.ib.placeOrder(contract, order)
            log.info(f"Placed closing order for {contract.symbol}: {trade.orderStatus.status}")
            
            # Wait loop for this specific order
            for _ in range(10):
                ib_client.ib.sleep(0.5)
                log.info(f"Order status for {contract.symbol}: {trade.orderStatus.status}")
                if trade.orderStatus.status in ['Filled', 'Cancelled', 'Inactive']:
                    break


        log.info("All closing orders placed. Waiting for execution...")
        
        # Wait for trades to fill
        start_wait = time.time()
        while time.time() - start_wait < 30: # Wait up to 30 seconds
            ib_client.ib.sleep(1)
            open_trades = ib_client.ib.trades()
            pending = [t for t in open_trades if not t.isDone()]
            if not pending:
                log.info("All closing orders filled.")
                break
            log.info(f"Waiting for {len(pending)} orders to fill...")
        
        # Final check
        ib_client.ib.sleep(2)
        final_positions = ib_client.ib.positions()
        if final_positions:
            log.error(f"Failed to close all positions. Remaining: {len(final_positions)}")
            for p in final_positions:
                log.error(f"  {p.contract.symbol}: {p.position}")
        else:
            log.info("Successfully closed all positions.")
        
        if pending:
            log.warning(f"{len(pending)} orders did not fill within timeout.")

    except Exception as e:
        log.error(f"Error closing positions: {e}")
    finally:
        ib_client.disconnect()

if __name__ == "__main__":
    close_all_positions()
