from src.logger.logger import log
from src.ib_client.ib_client import IBClient
from src.analysis.trade_logger import TradeLogger

import yaml
import math
import pandas_ta as ta

class TradeManager:
    def __init__(self, ib_client: IBClient):
        self.ib_client = ib_client
        self.trade_logger = TradeLogger()
        self._load_config()

    def _load_config(self):
        try:
            with open("config/strategy_config.yaml", 'r') as f:
                config = yaml.safe_load(f)
                self.risk_config = config.get('risk_management', {})
                self.position_size_usd = self.risk_config.get('position_size_usd', 2000)
                self.risk_per_trade = self.risk_config.get('risk_per_trade', 50)
                self.max_positions = self.risk_config.get('max_positions', 5)
                self.global_trend_filter = self.risk_config.get('global_trend_filter', True)
        except Exception as e:
            log.error(f"Error loading strategy config: {e}. Using defaults.")
            self.position_size_usd = 2000
            self.risk_per_trade = 50
            self.max_positions = 5
            self.global_trend_filter = True

    def calculate_smart_position(self, symbol, current_price, df):
        """
        Calculates position size based on ATR volatility.
        Risk: $50 per trade (Fixed)
        Stop Loss: 1 * ATR (tight for day trading)
        Take Profit: 1.5 * ATR (1:1.5 Risk/Reward ratio)
        
        V3.2 Updates:
        - Reduced TP from 4x to 1.5x ATR (realistic for day trading)
        - Reduced SL from 2x to 1x ATR (tighter stops)
        - Maintains good risk/reward while being achievable intraday
        """
        try:
            # ============= V3.1 VALIDATION START =============
            # 1. Check minimum data requirement
            if len(df) < 15:
                log.warning(f"[ATR_FAIL] {symbol}: Not enough data ({len(df)} rows < 15). Skipping.")
                return None, None, None

            # 2. Calculate ATR
            df.ta.atr(length=14, append=True)
            
            # 3. Verify ATR column exists
            if 'ATRr_14' not in df.columns:
                log.error(f"[ATR_FAIL] {symbol}: ATR column not created. Skipping.")
                return None, None, None

            current_atr = df['ATRr_14'].iloc[-1]

            # 4. Strict ATR validation (V3.1 - minimum threshold)
            ATR_MIN_THRESHOLD = 0.05
            if math.isnan(current_atr) or current_atr < ATR_MIN_THRESHOLD:
                log.warning(f"[ATR_FAIL] {symbol}: ATR={current_atr} is invalid or below threshold ({ATR_MIN_THRESHOLD}). Skipping.")
                return None, None, None
            # ============= V3.1 VALIDATION END =============

            # V3.2: Day Trading Optimized SL/TP
            SL_ATR_MULTIPLIER = 1.0   # Tight stop for day trading
            TP_ATR_MULTIPLIER = 1.5   # Realistic intraday target (1:1.5 R/R)
            
            stop_loss_dist = current_atr * SL_ATR_MULTIPLIER
            take_profit_dist = current_atr * TP_ATR_MULTIPLIER

            stop_loss_price = round(current_price - stop_loss_dist, 2)
            take_profit_price = round(current_price + take_profit_dist, 2)

            # V3.1: Validate SL/TP are logical
            if stop_loss_price <= 0 or stop_loss_price >= current_price:
                log.warning(f"[SL_FAIL] {symbol}: Invalid SL price ({stop_loss_price}) for entry ({current_price}). Skipping.")
                return None, None, None
            
            if take_profit_price <= current_price:
                log.warning(f"[TP_FAIL] {symbol}: Invalid TP price ({take_profit_price}) for entry ({current_price}). Skipping.")
                return None, None, None

            # Position Sizing: Risk / Distance to Stop
            risk_per_share = current_price - stop_loss_price
            if risk_per_share <= 0:
                log.warning(f"[RISK_FAIL] {symbol}: Invalid risk per share. Skipping.")
                return None, None, None

            quantity = int(self.risk_per_trade / risk_per_share)
            
            # V3.1: Minimum quantity check
            if quantity < 1:
                log.warning(f"[QTY_FAIL] {symbol}: Calculated quantity is 0 (Risk: ${self.risk_per_trade}, Per Share: ${risk_per_share:.2f}). Skipping.")
                return None, None, None

            # ============= V3.1 DEBUG LOG =============
            log.info(f"[SMART_RISK] {symbol}: Price=${current_price:.2f} | ATR=${current_atr:.3f} | "
                     f"SL=${stop_loss_price:.2f} ({SL_ATR_MULTIPLIER}x ATR, -{stop_loss_dist:.2f}) | "
                     f"TP=${take_profit_price:.2f} ({TP_ATR_MULTIPLIER}x ATR, +{take_profit_dist:.2f}) | "
                     f"Risk/Share=${risk_per_share:.2f} | Qty={quantity}")
            # ==========================================
            
            return quantity, stop_loss_price, take_profit_price

        except Exception as e:
            log.error(f"[ATR_ERROR] {symbol}: Exception in Smart Risk calculation: {e}")
            return None, None, None

    def process_signals(self, signals_df, symbol: str, strategy_name: str = "Unknown"):
        """Processes trading signals and places orders."""
        if signals_df.empty:
            log.warning("Received empty signals DataFrame. No action taken.")
            return

        latest_signal = signals_df.iloc[-1]
        signal = latest_signal['signal']

        log.info(f"Processing signal for {symbol}. Latest signal: {signal}")

        current_position = self.ib_client.get_position(symbol)
        has_open_order = self.ib_client.has_open_order(symbol)
        
        log.info(f"Current position for {symbol}: {current_position}. Open order exists: {has_open_order}")

        # Check Max Positions Limit
        all_positions = self.ib_client.get_all_positions()
        # Filter for non-zero positions
        active_positions = [p for p in all_positions if p.position != 0]
        
        if len(active_positions) >= self.max_positions and current_position == 0:
            # Log the actual positions causing the limit
            position_symbols = [f"{p.contract.symbol}:{p.position}" for p in active_positions]
            log.warning(f"Max positions limit reached ({len(active_positions)}/{self.max_positions}). Active positions: {position_symbols}. Skipping buy for {symbol}.")
            return

        # Get current price for Limit Order calculation (essential for Pre-Market)
        current_price = self.ib_client.get_current_price(symbol)

        if signal == 1:
            if current_position == 0 and not has_open_order:
                if current_price > 0:
                    # Global Trend Filter Check
                    if self.global_trend_filter:
                        # Check if SMA_200 is in the signals dataframe
                        if 'SMA_200' in signals_df.columns:
                            sma_200 = latest_signal['SMA_200']
                            if not math.isnan(sma_200) and current_price < sma_200:
                                log.info(f"Trend Filter: {symbol} price ({current_price}) is below SMA 200 ({sma_200}). Skipping Buy.")
                                return
                        else:
                            log.warning(f"Trend Filter enabled but SMA_200 not found for {symbol}. Proceeding with caution.")

                    # Smart Risk Management (ATR Based)
                    quantity, stop_loss_price, take_profit_price = self.calculate_smart_position(symbol, current_price, signals_df)

                    limit_price = round(current_price * 1.005, 2) # 0.5% buffer for entry

                    # Fallback to Fixed Size if ATR fails
                    if quantity is None or quantity < 1:
                        log.warning(f"Smart Risk failed for {symbol}. Falling back to fixed position size.")
                        quantity = int(self.position_size_usd / current_price)
                        if quantity < 1:
                            log.warning(f"Calculated quantity for {symbol} is 0 (Price: {current_price}, Size: {self.position_size_usd}). Skipping.")
                            return

                        limit_price = round(current_price * 1.01, 2) # 1% buffer for fallback
                        # Calculate TP (1.5%) and SL (1%)
                        take_profit_price = round(limit_price * 1.015, 2)
                        stop_loss_price = round(limit_price * 0.99, 2)

                    log.info(f"Buy signal detected for {symbol}. Placing BRACKET order. Qty: {quantity}, Entry: {limit_price}, TP: {take_profit_price}, SL: {stop_loss_price}")
                    self.ib_client.place_bracket_order(symbol, quantity, "BUY", limit_price, take_profit_price, stop_loss_price)
                    
                    # Log the trade
                    self.trade_logger.log_trade(
                        symbol=symbol,
                        strategy=strategy_name,
                        action="BUY",
                        price=limit_price,
                        quantity=quantity,
                        order_type="BRACKET",
                        tp_price=take_profit_price,
                        sl_price=stop_loss_price
                    )
                else:
                    log.warning(f"Buy signal for {symbol}, but could not get price. Placing MARKET order.")
                    # Fallback to 10 shares if price is unknown (risky, but keeps existing logic for fallback)
                    self.ib_client.place_order(symbol=symbol, quantity=10, action="BUY")
                    
                    self.trade_logger.log_trade(
                        symbol=symbol,
                        strategy=strategy_name,
                        action="BUY",
                        price="MARKET",
                        quantity=10,
                        order_type="MARKET"
                    )
            elif has_open_order:
                 log.info(f"Buy signal detected for {symbol}, but there is already an open order. Skipping.")
            else:
                log.info(f"Buy signal detected for {symbol}, but position already exists ({current_position}). Skipping.")
        
        elif signal == -1:
            # DISABLED EXPLICIT SELL SIGNALS to prevent strategy conflicts (Churning)
            # We rely on Bracket Orders (SL/TP) to close positions.
            log.info(f"Sell signal detected for {symbol} from {strategy_name}. IGNORING to rely on SL/TP and prevent churning.")
            
            # if current_position > 0 and not has_open_order:
            #     if current_price > 0:
            #         limit_price = round(current_price * 0.99, 2) # 1% buffer
            #         log.info(f"Sell signal detected for {symbol}. Placing LIMIT order at {limit_price} (Current: {current_price}).")
            #         self.ib_client.place_order(symbol=symbol, quantity=current_position, action="SELL", limit_price=limit_price)
            #         
            #         self.trade_logger.log_trade(
            #             symbol=symbol,
            #             strategy=strategy_name,
            #             action="SELL",
            #             price=limit_price,
            #             quantity=current_position,
            #             order_type="LIMIT"
            #         )
            #     else:
            #         log.warning(f"Sell signal for {symbol}, but could not get price. Placing MARKET order.")
            #         self.ib_client.place_order(symbol=symbol, quantity=current_position, action="SELL")
            #         
            #         self.trade_logger.log_trade(
            #             symbol=symbol,
            #             strategy=strategy_name,
            #             action="SELL",
            #             price="MARKET",
            #             quantity=current_position,
            #             order_type="MARKET"
            #         )
            # elif has_open_order:
            #     log.info(f"Sell signal detected for {symbol}, but there is already an open order. Skipping.")
            # else:
            #     log.info(f"Sell signal detected for {symbol}, but no position to sell. Skipping.")
        
        else:
            log.info(f"Neutral signal (0) for {symbol}. No action taken.")
