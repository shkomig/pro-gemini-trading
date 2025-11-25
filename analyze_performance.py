import pandas as pd
import yaml
from src.ib_client.ib_client import IBClient
from src.logger.logger import log
from ib_insync import ExecutionFilter

def analyze_performance():
    print("--- Starting Performance Analysis ---")
    
    # 1. Load Trade History from CSV (Signals/Entries)
    try:
        csv_path = "data/trade_history.csv"
        history_df = pd.read_csv(csv_path)
        print(f"\n[Local Log Analysis]")
        print(f"Total Signals Logged: {len(history_df)}")
        if not history_df.empty:
            print("\nActivity by Strategy:")
            print(history_df['strategy'].value_counts())
            print("\nActivity by Symbol:")
            print(history_df['symbol'].value_counts())
    except FileNotFoundError:
        print("\n[Local Log Analysis] No local trade history file found (data/trade_history.csv).")
    except Exception as e:
        print(f"\n[Local Log Analysis] Error reading CSV: {e}")

    # 2. Connect to IBKR for Realized PnL
    print(f"\n[IBKR Account Analysis]")
    try:
        with open("config/ib_config.yaml", 'r') as f:
            ib_config = yaml.safe_load(f)
        
        params = ib_config['connection']
        # Ensure we don't use a duplicate client ID if possible, or just reuse the config
        # We'll assume the main bot isn't running or we use a different ID if needed.
        # For safety, let's increment the client ID temporarily for analysis
        params['client_id'] = params.get('client_id', 99) + 1
        
        client = IBClient(**params)
        client.connect()
        
        # Request Executions
        print("Fetching executions from IBKR...")
        exec_filter = ExecutionFilter()
        fills = client.ib.reqExecutions(exec_filter)
        
        if not fills:
            print("No executions found in IBKR history.")
        else:
            data = []
            for fill in fills:
                # We are interested in realized PnL which is usually in commissionReport
                # Note: realizedPNL is only available if the position is closed
                pnl = 0.0
                if fill.commissionReport:
                    pnl = fill.commissionReport.realizedPNL
                
                data.append({
                    'symbol': fill.contract.symbol,
                    'action': fill.execution.side,
                    'qty': fill.execution.shares,
                    'price': fill.execution.price,
                    'time': fill.execution.time,
                    'realized_pnl': pnl,
                    'commission': fill.commissionReport.commission if fill.commissionReport else 0.0
                })
            
            df = pd.DataFrame(data)
            
            # Filter for fills with non-zero PnL (Closed trades)
            closed_trades = df[df['realized_pnl'] != 0]
            
            total_pnl = df['realized_pnl'].sum()
            total_comm = df['commission'].sum()
            
            print(f"\n--- Financial Summary ---")
            print(f"Total Realized PnL: ${total_pnl:.2f}")
            print(f"Total Commissions:  ${total_comm:.2f}")
            print(f"Net Profit:         ${(total_pnl - total_comm):.2f}")
            
            if not closed_trades.empty:
                print(f"\n--- Closed Trades Statistics ---")
                wins = closed_trades[closed_trades['realized_pnl'] > 0]
                losses = closed_trades[closed_trades['realized_pnl'] < 0]
                
                win_rate = (len(wins) / len(closed_trades)) * 100
                print(f"Total Closed Trades: {len(closed_trades)}")
                print(f"Wins: {len(wins)}")
                print(f"Losses: {len(losses)}")
                print(f"Win Rate: {win_rate:.2f}%")
                
                if not wins.empty:
                    print(f"Avg Win: ${wins['realized_pnl'].mean():.2f}")
                if not losses.empty:
                    print(f"Avg Loss: ${losses['realized_pnl'].mean():.2f}")
            else:
                print("\nNo closed trades with realized PnL detected yet.")
                
            print("\n--- Recent Executions ---")
            print(df[['time', 'symbol', 'action', 'qty', 'price', 'realized_pnl']].tail(10))

        client.disconnect()

    except Exception as e:
        log.error(f"IBKR Analysis failed: {e}")

if __name__ == "__main__":
    analyze_performance()
