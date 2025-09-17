import ccxt
import time
import talib
import numpy as np
import os
import logging
import json
import csv
from datetime import datetime

# --- CONFIGURATION (À REMPLIR) ---
API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# --- PARAMÈTRES DU BOT ---
SYMBOLS = ['BTC/USDT', 'ETH/USDT'] # Liste des paires à trader
CHECK_INTERVAL_SECONDS = 60
CANDLE_PERIOD = '1h'
MA_SHORT_PERIOD = 20
MA_LONG_PERIOD = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5 # Coefficient pour le stop-loss dynamique
BB_PERIOD = 20
BB_DEVIATIONS = 2
TAKE_PROFIT_PERCENTAGE = 0.03
TRANSACTION_FEES = 0.002
POSITION_SIZE_PERCENTAGE = 0.1
STATE_FILE = 'bot_state.json'
TRADE_HISTORY_FILE = 'trade_history.csv'
PAPER_TRADING_MODE = False # Mettre à True pour simuler les trades sans risques
MIN_NOTIONAL_FALLBACK = 10.0 # Valeur de secours pour le montant minimum de transaction

# --- CONFIGURATION DE LA JOURNALISATION (LOGGING) ---
logging.basicConfig(filename='trading_bot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# --- FONCTIONS UTILES ---
def save_state(state):
    """Sauvegarde l'état du bot dans un fichier JSON."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
    except IOError as e:
        logging.error(f"Erreur lors de la sauvegarde de l'état du bot : {e}")

def load_state():
    """Charge l'état du bot depuis un fichier JSON. Initialise l'état si le fichier n'existe pas."""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Erreur lors du chargement de l'état du bot : {e}")
    
    initial_state = {}
    for symbol in SYMBOLS:
        initial_state[symbol] = {
            'last_transaction_type': 'SELL',
            'last_buy_price': 0.0,
            'max_price_since_buy': 0.0,
            'consecutive_api_failures': 0
        }
    return initial_state

def get_ohlcv(exchange, symbol, timeframe, limit):
    """Récupère les données OHLCV pour un symbole donné."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return ohlcv
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logging.error(f"Erreur lors de la récupération des données pour {symbol}: {e}")
        return None

def get_account_balance(exchange):
    """Récupère le solde du compte."""
    try:
        balance = exchange.fetch_balance()
        return balance
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logging.error(f"Erreur lors de la récupération du solde du compte : {e}")
        return None

def get_balance_for_currency(balance, currency):
    """Retourne un dictionnaire avec le solde total, libre et utilisé pour une devise donnée."""
    return {
        'free': balance['free'].get(currency, 0),
        'used': balance['used'].get(currency, 0),
        'total': balance['total'].get(currency, 0),
    }

def log_trade_to_csv(trade_data):
    """Enregistre les détails d'une transaction dans un fichier CSV."""
    header_exists = os.path.exists(TRADE_HISTORY_FILE)
    with open(TRADE_HISTORY_FILE, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'symbol', 'type', 'price', 'amount', 'total', 'fees']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not header_exists:
            writer.writeheader()
        
        writer.writerow(trade_data)

def log_current_balances(balance, symbols):
    """Affiche le solde des devises pertinentes."""
    logging.info("--- SOLDE ACTUEL ---")
    if balance:
        currencies_to_check = set()
        for s in symbols:
            currencies_to_check.add(s.split('/')[0])
            currencies_to_check.add(s.split('/')[1])
        
        for currency in currencies_to_check:
            bal_info = get_balance_for_currency(balance, currency)
            if bal_info['total'] > 0.001:
                logging.info(f"{currency}: Total: {bal_info['total']:.6f} | Disponible: {bal_info['free']:.6f}")
    logging.info("--------------------")

def quantize_amount(exchange, symbol, amount):
    """Arrondit la quantité à la précision requise par l'échange."""
    return exchange.amount_to_precision(symbol, amount)

def quantize_price(exchange, symbol, price):
    """Arrondit le prix à la précision requise par l'échange."""
    return exchange.price_to_precision(symbol, price)

def handle_buy_signal(exchange, symbol, current_state, last_price, balance):
    """Gère la logique d'un ordre d'achat."""
    
    quote_currency = symbol.split('/')[1]
    usdt_balance = get_balance_for_currency(balance, quote_currency)
    
    if usdt_balance['free'] > MIN_NOTIONAL_FALLBACK:
        logging.info(f"Signal d'achat détecté pour {symbol}. Exécution de l'ordre...")
        try:
            market = exchange.markets[symbol]
            min_notional = market['limits']['cost']['min'] if 'cost' in market['limits'] and market['limits']['cost']['min'] else MIN_NOTIONAL_FALLBACK
            
            amount_to_buy_usdt = usdt_balance['free'] * POSITION_SIZE_PERCENTAGE
            
            if amount_to_buy_usdt < min_notional:
                logging.warning(f"Quantité d'achat trop faible ({amount_to_buy_usdt:.2f} {quote_currency}). Min. requis: {min_notional:.2f} {quote_currency}. Ordre annulé.")
                return

            amount_to_buy = amount_to_buy_usdt / last_price
            amount_to_buy = quantize_amount(exchange, symbol, amount_to_buy)

            order = None
            if PAPER_TRADING_MODE:
                order = {
                    'status': 'closed',
                    'filled': float(amount_to_buy),
                    'price': last_price,
                    'fee': {'cost': float(amount_to_buy) * TRANSACTION_FEES}
                }
                logging.info(f"[MODE SIMULATION] Ordre d'achat simulé pour {symbol}. Montant: {amount_to_buy:.4f}")
            else:
                order = exchange.create_market_buy_order(symbol, amount_to_buy)
                time.sleep(1) # Délai pour que l'ordre soit traité
                order = exchange.fetch_order(order['id'], symbol)
            
            if order and order.get('filled', 0) > 0:
                current_state['last_transaction_type'] = 'BUY'
                current_state['last_buy_price'] = order.get('price', last_price)
                current_state['max_price_since_buy'] = order.get('price', last_price)
                
                trade_data = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'type': 'BUY',
                    'price': order.get('price', last_price),
                    'amount': order['filled'],
                    'total': order['filled'] * order.get('price', last_price),
                    'fees': json.dumps(order.get('fee', {}))
                }
                log_trade_to_csv(trade_data)
                logging.info(f"Ordre d'achat exécuté sur {symbol}. Montant acheté: {order['filled']:.4f}, Prix: {order.get('price', last_price):.2f}")
            else:
                logging.warning(f"L'ordre d'achat sur {symbol} n'a pas été exécuté correctement. Le bot reste en mode 'SELL'.")
        except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout) as e:
            logging.error(f"Erreur d'échange lors de l'achat sur {symbol}: {e}")
    else:
        logging.warning(f"Signal d'achat sur {symbol} mais solde {quote_currency} insuffisant. Le bot ne fait rien.")

def handle_sell_signal(exchange, symbol, current_state, last_price, atr, balance):
    """Gère la logique d'un ordre de vente."""
    
    base_currency = symbol.split('/')[0]
    base_balance = get_balance_for_currency(balance, base_currency)
    
    trailing_stop_price = current_state['max_price_since_buy'] - (ATR_MULTIPLIER * atr)
    take_profit_price = current_state['last_buy_price'] * (1 + TAKE_PROFIT_PERCENTAGE + TRANSACTION_FEES)
    
    logging.info(f"Position ouverte sur {symbol}. Achat: ${current_state['last_buy_price']:.2f}, Max: ${current_state['max_price_since_buy']:.2f}, SL: ${trailing_stop_price:.2f}, TP: ${take_profit_price:.2f}")

    if last_price <= trailing_stop_price or last_price >= take_profit_price:
        if base_balance['free'] > 0: # Vérifier si le solde est suffisant
            logging.info(f"Signal de vente pour {symbol}! Exécution de l'ordre...")
            try:
                amount_to_sell = quantize_amount(exchange, symbol, base_balance['free'])

                order = None
                if PAPER_TRADING_MODE:
                    order = {
                        'status': 'closed',
                        'filled': float(amount_to_sell),
                        'price': last_price,
                        'fee': {'cost': float(amount_to_sell) * TRANSACTION_FEES}
                    }
                    logging.info(f"[MODE SIMULATION] Ordre de vente simulé pour {symbol}. Montant: {amount_to_sell:.4f}")
                else:
                    order = exchange.create_market_sell_order(symbol, amount_to_sell)
                    time.sleep(1) # Délai pour que l'ordre soit traité
                    order = exchange.fetch_order(order['id'], symbol)

                if order and order.get('filled', 0) > 0:
                    current_state['last_transaction_type'] = 'SELL'
                    trade_data = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'type': 'SELL',
                        'price': order.get('price', last_price),
                        'amount': order['filled'],
                        'total': order['filled'] * order.get('price', last_price),
                        'fees': json.dumps(order.get('fee', {}))
                    }
                    log_trade_to_csv(trade_data)
                    logging.info(f"Ordre de vente exécuté sur {symbol}. Montant vendu: {order['filled']:.4f}")
                else:
                    logging.warning(f"L'ordre de vente sur {symbol} n'a pas été exécuté correctement. Le bot reste en mode 'BUY'.")
            except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout) as e:
                logging.error(f"Erreur d'échange lors de la vente sur {symbol}: {e}")
        else:
            logging.warning(f"Signal de vente sur {symbol} mais pas de solde {base_currency} suffisant. Le bot ne fait rien.")

def run_bot_logic(exchange, symbol, state, balance):
    """Contient la logique principale du bot pour un symbole donné."""
    current_state = state[symbol]
    
    ohlcv = get_ohlcv(exchange, symbol, CANDLE_PERIOD, 200)
    if ohlcv is None or len(ohlcv) < max(MA_LONG_PERIOD, RSI_PERIOD, BB_PERIOD, ATR_PERIOD):
        current_state['consecutive_api_failures'] += 1
        if current_state['consecutive_api_failures'] >= 5:
            logging.critical(f"Trop d'échecs API consécutifs pour {symbol}. Le bot se met en pause.")
            time.sleep(3600)
            current_state['consecutive_api_failures'] = 0
        logging.warning(f"Pas assez de données historiques pour l'analyse de {symbol}. En attente.")
        return
    else:
        current_state['consecutive_api_failures'] = 0

    closes = np.array([candle[4] for candle in ohlcv], dtype=float)
    highs = np.array([candle[2] for candle in ohlcv], dtype=float)
    lows = np.array([candle[3] for candle in ohlcv], dtype=float)
    
    try:
        ma_short = talib.SMA(closes, timeperiod=MA_SHORT_PERIOD)[-1]
        ma_long = talib.SMA(closes, timeperiod=MA_LONG_PERIOD)[-1]
        rsi = talib.RSI(closes, timeperiod=RSI_PERIOD)[-1]
        upper_band, _, _ = talib.BBANDS(closes, timeperiod=BB_PERIOD, nbdevup=BB_DEVIATIONS, nbdevdn=BB_DEVIATIONS, matype=0)
        upper_band = upper_band[-1]
        atr = talib.ATR(highs, lows, closes, timeperiod=ATR_PERIOD)[-1]
    except Exception as e:
        logging.error(f"Erreur dans le calcul des indicateurs pour {symbol}: {e}")
        return

    if np.isnan(ma_short) or np.isnan(ma_long) or np.isnan(rsi) or np.isnan(upper_band) or np.isnan(atr):
        logging.warning(f"Indicateurs invalides (NaN) pour {symbol}, passage au cycle suivant.")
        return

    last_price = closes[-1]
    
    # Étape 3 : Logique de la stratégie
    if current_state['last_transaction_type'] == 'BUY':
        current_state['max_price_since_buy'] = max(current_state['max_price_since_buy'], last_price)
        handle_sell_signal(exchange, symbol, current_state, last_price, atr, balance)
    elif current_state['last_transaction_type'] == 'SELL':
        if ma_short > ma_long and rsi < 70 and last_price < upper_band:
            handle_buy_signal(exchange, symbol, current_state, last_price, balance)

def main():
    if not API_KEY or not SECRET_KEY:
        logging.critical("Les clés API n'ont pas été trouvées. Veuillez les définir comme variables d'environnement.")
        return

    state = load_state()

    try:
        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'enableRateLimit': True,
        })
        exchange.load_markets()
        logging.info("Connexion à Binance réussie. Lancement du bot de trading...")
        
        while True:
            balance = get_account_balance(exchange)
            if not balance:
                logging.error("Impossible de récupérer le solde. Nouvel essai au prochain cycle.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue
            
            log_current_balances(balance, SYMBOLS)

            for symbol in SYMBOLS:
                run_bot_logic(exchange, symbol, state, balance)
                time.sleep(1) # Ajoute un petit délai entre les requêtes pour chaque symbole
            
            save_state(state)
            logging.info(f"Prochaine analyse dans {CHECK_INTERVAL_SECONDS} secondes...")
            time.sleep(CHECK_INTERVAL_SECONDS)

    except Exception as e:
        logging.critical(f"Une erreur fatale est survenue : {e}. Le bot s'arrête.")
        save_state(state)
        return

if __name__ == "__main__":
    main()