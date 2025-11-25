from ib_insync import *
import time

def onPendingTicker(ticker):
    print("Pending ticker event received")

def onTickerUpdate(ticker):
    print(f"Update for {ticker.contract.symbol}: Last={ticker.last} Bid={ticker.bid} Ask={ticker.ask}")

def onError(reqId, errorCode, errorString, contract):
    print(f"ERROR {errorCode} for {contract.symbol if contract else 'Unknown'}: {errorString}")

print("--- STARTING DIAGNOSTIC TEST ---")
ib = IB()
ib.errorEvent += onError

# Connect to the same port as the main system
print("Connecting to IB...")
try:
    # Try connecting to Live TWS (7496) first to verify subscription
    print("Attempting to connect to LIVE TWS (Port 7496) to verify data subscription...")
    ib.connect('127.0.0.1', 7496, clientId=99) 
except Exception as e:
    print(f"Live connection failed: {e}")
    print("Falling back to PAPER TWS (Port 7497)...")
    try:
        ib.connect('127.0.0.1', 7497, clientId=99)
    except Exception as e2:
        print(f"Paper connection failed: {e2}")
        exit()

print("Connected. Setting Market Data Type to 1 (Real-Time)...")
ib.reqMarketDataType(1)

# Define contracts
spy = Stock('SPY', 'SMART', 'USD')
soxl = Stock('SOXL', 'SMART', 'USD')

print("Requesting Market Data for SPY and SOXL...")
spy_ticker = ib.reqMktData(spy, '', False, False)
soxl_ticker = ib.reqMktData(soxl, '', False, False)

# Register callback
ib.pendingTickersEvent += onPendingTicker

# Loop for 10 seconds to catch data
start = time.time()
while time.time() - start < 10:
    ib.sleep(1)
    print(f"Waiting for data... (SPY Last: {spy_ticker.last}, SOXL Last: {soxl_ticker.last})")

print("Disconnecting.")
ib.disconnect()
