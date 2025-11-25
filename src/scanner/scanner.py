"""
Enhanced Market Scanner - V3.3
==============================
Multi-scan system to find the best trading opportunities.

Scan Types:
1. TOP_OPEN_PERC_GAIN - Stocks with biggest gap up
2. TOP_OPEN_PERC_LOSE - Stocks with biggest gap down  
3. HOT_BY_VOLUME - Highest relative volume
4. MOST_ACTIVE - Most traded volume

Version: 3.3
"""

import yaml
from ib_insync import ScannerSubscription, TagValue
from src.logger.logger import log
from src.ib_client.ib_client import IBClient
from typing import List, Set


class Scanner:
    """
    Enhanced Market Scanner V3.3
    
    Runs multiple scan types to find the best opportunities:
    - Gap scanners (up and down)
    - Volume scanners (relative and absolute)
    - Combines and deduplicates results
    """
    
    # Blacklist of symbols to exclude (Inverse/Bear ETFs, leveraged ETFs, etc.)
    BLACKLIST = {
        # Inverse ETFs
        'SQQQ', 'SOXS', 'SPDN', 'TSLS', 'NVD', 'UVIX', 'TZA', 'SRTY', 
        'SPXU', 'SDOW', 'TECS', 'FAZ', 'LABD', 'YANG', 'DRV', 'SARK',
        'SH', 'PSQ', 'RWM', 'DOG', 'SDS', 'QID', 'SPXS', 'VIXY',
        # Problematic or illiquid
        'DRIP', 'KOLD', 'ERY', 'DUST', 'JDST', 'WEBS'
    }
    
    # Scan configurations
    SCAN_TYPES = [
        {
            'name': 'Gap Up',
            'code': 'TOP_OPEN_PERC_GAIN',
            'max_results': 15,
            'description': 'Stocks gapping up from previous close'
        },
        {
            'name': 'Hot Volume',
            'code': 'HOT_BY_VOLUME',
            'max_results': 15,
            'description': 'Stocks with unusual relative volume'
        },
        {
            'name': 'Most Active',
            'code': 'MOST_ACTIVE',
            'max_results': 10,
            'description': 'Most traded by volume'
        }
    ]

    def __init__(self, ib_client: IBClient, config_path: str):
        self.ib_client = ib_client
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        log.info("[SCANNER] Enhanced Scanner V3.3 initialized")

    def _run_single_scan(self, scan_code: str, scan_name: str, 
                         max_results: int = 20) -> List[str]:
        """Run a single IB scanner and return symbols."""
        
        universe_config = self.config.get('universe', {})
        min_price = universe_config.get('min_price', 5.0)
        max_price = universe_config.get('max_price', 100.0)
        min_volume = universe_config.get('min_avg_daily_volume', 500000)
        
        sub = ScannerSubscription(
            instrument='STK',
            locationCode='STK.US.MAJOR',
            scanCode=scan_code,
            abovePrice=min_price,
            belowPrice=max_price,
            aboveVolume=min_volume,
            numberOfRows=max_results
        )
        
        try:
            scan_data = self.ib_client.ib.reqScannerData(sub, [], [])
            
            symbols = []
            for data in scan_data:
                symbol = data.contractDetails.contract.symbol
                if symbol not in self.BLACKLIST and symbol not in symbols:
                    symbols.append(symbol)
            
            log.info(f"[SCANNER] {scan_name}: Found {len(symbols)} symbols")
            return symbols
            
        except Exception as e:
            log.error(f"[SCANNER] {scan_name} scan failed: {e}")
            return []

    def scan_market(self) -> List[str]:
        """
        Run multiple scans and combine results.
        
        Returns prioritized list of unique symbols.
        """
        log.info("[SCANNER] Running enhanced multi-scan...")
        
        all_symbols: List[str] = []
        seen: Set[str] = set()
        
        for scan_config in self.SCAN_TYPES:
            symbols = self._run_single_scan(
                scan_code=scan_config['code'],
                scan_name=scan_config['name'],
                max_results=scan_config['max_results']
            )
            
            # Add unique symbols (maintaining priority order)
            for symbol in symbols:
                if symbol not in seen:
                    all_symbols.append(symbol)
                    seen.add(symbol)
        
        log.info(f"[SCANNER] Combined scan: {len(all_symbols)} unique symbols")
        log.info(f"[SCANNER] Top candidates: {all_symbols[:20]}")
        
        return all_symbols

    def scan_gap_up(self, min_gap_percent: float = 3.0) -> List[str]:
        """Scan specifically for gap up stocks."""
        log.info(f"[SCANNER] Scanning for gap up stocks (min {min_gap_percent}%)")
        return self._run_single_scan('TOP_OPEN_PERC_GAIN', 'Gap Up', 25)
    
    def scan_hot_volume(self) -> List[str]:
        """Scan specifically for unusual volume stocks."""
        log.info("[SCANNER] Scanning for hot volume stocks")
        return self._run_single_scan('HOT_BY_VOLUME', 'Hot Volume', 25)
    
    def scan_most_active(self) -> List[str]:
        """Scan for most active stocks by volume."""
        log.info("[SCANNER] Scanning for most active stocks")
        return self._run_single_scan('MOST_ACTIVE', 'Most Active', 25)
