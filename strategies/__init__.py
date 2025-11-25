"""
Strategies Module
=================

Trading strategies implementation.
"""

from .base_strategy import (
    BaseStrategy,
    TradingSignal,
    SignalType,
    SignalStrength
)
from .ema_cross_strategy import EMACrossStrategy
from .vwap_strategy import VWAPStrategy
from .volume_breakout_strategy import VolumeBreakoutStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .orb_strategy import ORBStrategy
from .momentum_strategy import MomentumStrategy
from .bollinger_bands_strategy import BollingerBandsStrategy
from .pairs_trading_strategy import PairsTradingStrategy
from .rsi_divergence_strategy import RSIDivergenceStrategy
from .advanced_volume_breakout_strategy import VolumeBreakoutStrategy as AdvancedVolumeBreakoutStrategy

__all__ = [
    'BaseStrategy',
    'TradingSignal',
    'SignalType',
    'SignalStrength',
    'EMACrossStrategy',
    'VWAPStrategy',
    'VolumeBreakoutStrategy',
    'MeanReversionStrategy',
    'ORBStrategy',
    'MomentumStrategy',
    'BollingerBandsStrategy',
    'PairsTradingStrategy',
    'RSIDivergenceStrategy',
    'AdvancedVolumeBreakoutStrategy'
]
