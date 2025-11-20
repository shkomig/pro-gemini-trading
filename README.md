# Gemini Pro Trading System

This is an automated trading system skeleton built with Python. It uses Interactive Brokers (IB) for market data and order execution, managed through the `ib_insync` library. The system is designed to be modular, allowing for easy addition of new trading strategies.

## Features

- **IBKR Integration**: Connects to Trader Workstation (TWS) or IB Gateway for live data and trading.
- **Simulation Mode**: Run the system with dummy data for testing and development without connecting to IB.
- **Strategy Framework**: A simple framework to implement and test trading strategies (e.g., Moving Average Crossover).
- **Trade Management**: A basic trade manager to process signals and place orders.
- **Configuration Driven**: Easily configure connection, data requests, and strategies via YAML files.
- **Robust Logging**: Logs are printed to the console and saved to a file in the `logs` directory.

---

## Prerequisites

1.  **Python 3.8+**
2.  **Interactive Brokers Trader Workstation (TWS)**: You must have an account (either a live or a paper trading account) and have TWS installed and running.

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shkomig/pro-gemini-trading.git
cd pro-gemini-trading
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv .venv
.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

## Configuration

### 1. Configure TWS for API Access (CRITICAL STEP)

For the application to connect to TWS, you MUST configure the API settings within TWS:

1.  Open Trader Workstation.
2.  Go to **File > Global Configuration...** (or **Edit > Global Configuration...** on older versions).
3.  In the left pane, expand **API** and click on **Settings**.
4.  Check the box **Enable ActiveX and Socket Clients**.
5.  Make sure the **Socket port** is set correctly. By default:
    *   `7496` is for live trading accounts.
    *   `7497` is for paper trading accounts.
    The `config/ib_config.yaml` is pre-configured for `7497`.
6.  **Important**: To allow the application to place trades, you must uncheck **Read-Only API**.
7.  Under **Trusted IP Addresses**, click **Create** and add `127.0.0.1`. This tells TWS to trust connection requests from your local machine without prompting for confirmation every time.

![TWS API Settings](https://www.interactivebrokers.com/images/web/tws-api-settings-v2.png) *(Image for reference)*

### 2. Application Configuration (`config/ib_config.yaml`)

This file controls the connection to IB and the main data request.

- `host`: The IP address where TWS is running (usually `127.0.0.1`).
- `port`: The TWS socket port (e.g., `7497` for paper trading).
- `simulation_mode`: Set to `false` to connect to TWS. Set to `true` to use simulated data and skip the connection.
- `symbol`: The stock symbol you want to trade (e.g., "AAPL").

### 3. Strategy Configuration (`config/strategy_config.yaml`)

This file defines which trading strategies to run.
You can add more strategies here as you develop them.

---

## Running the System

Make sure TWS is running and you are logged in. Then, execute the main script:

```bash
python main.py
```

The system will:
1.  Connect to TWS.
2.  Request historical data.
3.  Run the defined strategies to generate signals.
4.  Process the signals through the `TradeManager` to place orders (if any signal is generated).
5.  Log all its actions to the console and to `logs/trading_system.log`.
