from ib_insync import *
import pandas as pd

def check_portfolio():
    ib = IB()
    try:
        # Connect to the same port as the main app (7497 for paper trading)
        ib.connect('127.0.0.1', 7497, clientId=999) # Use a different clientId
        
        positions = ib.positions()
        portfolio = []
        
        print(f"Total Positions: {len(positions)}")
        
        total_unrealized_pnl = 0
        
        for pos in positions:
            contract = pos.contract
            # Request market data to get current price
            ticker = ib.reqMktData(contract, '', False, False)
            ib.sleep(1) # Wait for data
            
            market_price = ticker.marketPrice()
            if market_price != float('nan'):
                market_value = pos.position * market_price
                avg_cost = pos.avgCost * pos.position
                unrealized_pnl = market_value - avg_cost
            else:
                unrealized_pnl = 0 # Could not get price
            
            total_unrealized_pnl += unrealized_pnl
            
            portfolio.append({
                'Symbol': contract.symbol,
                'Position': pos.position,
                'Avg Cost': pos.avgCost,
                'Market Price': market_price,
                'Unrealized PnL': unrealized_pnl
            })
            
        df = pd.DataFrame(portfolio)
        if not df.empty:
            print(df.to_string())
            print("\n" + "="*30)
            print(f"Total Unrealized PnL: {total_unrealized_pnl:.2f}")
        else:
            print("No active positions.")
            
        # Also check account summary for Net Liquidation Value
        account_summary = ib.accountSummary()
        for item in account_summary:
            if item.tag == 'NetLiquidation':
                print(f"Net Liquidation Value: {item.value} {item.currency}")
            if item.tag == 'TotalCashValue':
                 print(f"Total Cash Value: {item.value} {item.currency}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        ib.disconnect()

if __name__ == "__main__":
    check_portfolio()
