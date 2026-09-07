import numpy as np
import talib
import logging

class TradingStrategy:
    def __init__(self, config):
        self.config = config
        self.indicators = config['indicators']

    def calculate_indicators(self, ohlcv):
        """Calcule les indicateurs techniques."""
        closes = np.array([c[4] for c in ohlcv], dtype=float)
        highs = np.array([c[2] for c in ohlcv], dtype=float)
        lows = np.array([c[3] for c in ohlcv], dtype=float)

        try:
            ma_short = talib.SMA(closes, timeperiod=self.indicators['ma_short_period'])
            ma_long = talib.SMA(closes, timeperiod=self.indicators['ma_long_period'])
            rsi = talib.RSI(closes, timeperiod=self.indicators['rsi_period'])
            upper_band, _, _ = talib.BBANDS(
                closes,
                timeperiod=self.indicators['bb_period'],
                nbdevup=self.indicators['bb_deviations'],
                nbdevdn=self.indicators['bb_deviations']
            )
            atr = talib.ATR(highs, lows, closes, timeperiod=self.indicators['atr_period'])

            return {
                'ma_short_now': ma_short[-1],
                'ma_short_prev': ma_short[-2],
                'ma_long_now': ma_long[-1],
                'ma_long_prev': ma_long[-2],
                'rsi': rsi[-1],
                'upper_band': upper_band[-1],
                'atr': atr[-1],
                'last_price': closes[-1]
            }
        except Exception as e:
            logging.error(f"Erreur calcul indicateurs: {e}")
            return None

    def get_buy_signal(self, indicators, current_state):
        """Détermine si un signal d'achat est présent."""
        crossover = indicators['ma_short_prev'] <= indicators['ma_long_prev'] and \
                    indicators['ma_short_now'] > indicators['ma_long_now']
        rsi_ok = indicators['rsi'] < 70
        price_ok = indicators['last_price'] < indicators['upper_band']

        return crossover and rsi_ok and price_ok

    def get_sell_signal(self, indicators, current_state):
        """Détermine si un signal de vente est présent."""
        trailing_stop = current_state['max_price_since_buy'] - (self.indicators['atr_multiplier'] * indicators['atr'])
        take_profit = current_state['last_buy_price'] * (1 + self.config['strategy']['take_profit_percentage'] + self.config['strategy']['transaction_fees'])

        if indicators['last_price'] < trailing_stop or indicators['last_price'] > take_profit:
            return True
        return False
