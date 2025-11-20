from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals based on the input data.

        Args:
            data (pd.DataFrame): A DataFrame with historical market data,
                                 expected to have 'close' prices.

        Returns:
            pd.DataFrame: The input DataFrame with an added 'signal' column.
                          Signal can be 1 (buy), -1 (sell), or 0 (hold).
        """
        pass
