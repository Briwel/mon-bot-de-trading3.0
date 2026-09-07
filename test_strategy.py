import unittest
from strategy import TradingStrategy
import numpy as np

class TestTradingStrategy(unittest.TestCase):
    def setUp(self):
        self.config = {
            'indicators': {
                'ma_short_period': 2,
                'ma_long_period': 3,
                'rsi_period': 2,
                'atr_period': 2,
                'atr_multiplier': 1.0,
                'bb_period': 3,
                'bb_deviations': 2
            },
            'strategy': {
                'take_profit_percentage': 0.05,
                'transaction_fees': 0.001
            }
        }
        self.strategy = TradingStrategy(self.config)

    def test_calculate_indicators(self):
        # Create dummy OHLCV data (minimal for indicators)
        # Needs at least enough data for the longest period (ma_long=3 + cushion)
        ohlcv = [
            [0, 10, 12, 8, 10],
            [0, 10, 12, 8, 11],
            [0, 10, 12, 8, 12],
            [0, 10, 12, 8, 13],
            [0, 10, 12, 8, 14]
        ]
        indicators = self.strategy.calculate_indicators(ohlcv)
        self.assertIsNotNone(indicators)
        self.assertIn('ma_short_now', indicators)

    def test_buy_signal(self):
        # Setup dummy indicators where buy conditions are met
        indicators = {
            'ma_short_prev': 10,
            'ma_short_now': 12,
            'ma_long_prev': 11,
            'ma_long_now': 11,
            'rsi': 50,
            'last_price': 100,
            'upper_band': 110
        }
        current_state = {}
        self.assertTrue(self.strategy.get_buy_signal(indicators, current_state))

    def test_sell_signal(self):
        # Setup dummy indicators where sell conditions are met
        indicators = {
            'last_price': 120,
            'atr': 5
        }
        current_state = {
            'max_price_since_buy': 110,
            'last_buy_price': 100
        }
        # trailing_stop = 110 - (1 * 5) = 105
        # take_profit = 100 * (1 + 0.05 + 0.001) = 105.1
        # Price 120 > 105.1 (Take profit)
        self.assertTrue(self.strategy.get_sell_signal(indicators, current_state))

if __name__ == '__main__':
    unittest.main()
