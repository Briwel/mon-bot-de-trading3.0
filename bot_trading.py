import ccxt
import time
import talib
import numpy as np
import os
import logging
import json
import csv
from datetime import datetime

from api_connector import create_exchange

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
POSITION_SIZE_PERCENTAGE = 0.1  # Plafond : max 10 % du capital par trade
RISK_PER_TRADE_PERCENTAGE = 0.01  # 1 % du capital total risqué par trade (position sizing ATR)
EXCHANGE_TIMEOUT = 30000  # Timeout CCXT en millisecondes (30 s)
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

def _default_symbol_state():
    """Retourne l'état initial par défaut pour un symbole."""
    return {
        'last_transaction_type': 'SELL',
        'last_buy_price': 0.0,
        'max_price_since_buy': 0.0,
        'consecutive_api_failures': 0
    }

def load_state():
    """Charge l'état du bot depuis un fichier JSON. Initialise l'état si le fichier n'existe pas."""
    state = {}
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Erreur lors du chargement de l'état du bot : {e}")

    # Supprimer les clés orphelines de premier niveau qui ne correspondent à aucun symbole actif
    # (p. ex. restes d'une ancienne version du fichier d'état)
    orphan_keys = [k for k in list(state.keys()) if k not in SYMBOLS]
    for k in orphan_keys:
        logging.info(f"Nettoyage de la clé d'état orpheline : '{k}'")
        del state[k]

    # S'assurer que chaque symbole actif a bien son entrée (p. ex. si un nouveau symbole est ajouté)
    for symbol in SYMBOLS:
        if symbol not in state:
            state[symbol] = _default_symbol_state()
    return state

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
    try:
        header_exists = os.path.exists(TRADE_HISTORY_FILE)
        with open(TRADE_HISTORY_FILE, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'symbol', 'type', 'price', 'amount', 'total', 'fees']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not header_exists:
                writer.writeheader()
            
            writer.writerow(trade_data)
    except IOError as e:
        logging.error(f"Erreur lors de l'enregistrement du trade dans le CSV : {e}")

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

def calculate_position_size_usdt(balance, symbol, last_price, atr):
    """Calcule le montant en devise de cotation à investir basé sur l'ATR (risque constant).

    La taille est déterminée pour que la distance de stop-loss (ATR_MULTIPLIER × ATR)
    représente exactement RISK_PER_TRADE_PERCENTAGE du capital total.
    Le résultat est plafonné à POSITION_SIZE_PERCENTAGE du capital pour éviter
    les positions surdimensionnées en cas de faible volatilité.
    """
    quote_currency = symbol.split('/')[1]
    total_balance = get_balance_for_currency(balance, quote_currency)['total']

    if total_balance <= 0 or atr <= 0 or last_price <= 0:
        return 0.0

    risk_amount = total_balance * RISK_PER_TRADE_PERCENTAGE
    stop_distance_units = ATR_MULTIPLIER * atr          # distance stop-loss en unités de base
    units_to_buy = risk_amount / stop_distance_units    # nb d'unités de base
    notional = units_to_buy * last_price                # montant en quote currency

    max_notional = total_balance * POSITION_SIZE_PERCENTAGE
    return min(notional, max_notional)


def quantize_amount(exchange, symbol, amount):
    """Arrondit la quantité à la précision requise par l'échange."""
    return exchange.amount_to_precision(symbol, amount)

def quantize_price(exchange, symbol, price):
    """Arrondit le prix à la précision requise par l'échange."""
    return exchange.price_to_precision(symbol, price)

def handle_buy_signal(exchange, symbol, current_state, last_price, atr, balance):
    """Gère la logique d'un ordre d'achat."""
    
    quote_currency = symbol.split('/')[1]
    usdt_balance = get_balance_for_currency(balance, quote_currency)
    
    if usdt_balance['free'] > MIN_NOTIONAL_FALLBACK:
        logging.info(f"Signal d'achat détecté pour {symbol}. Exécution de l'ordre...")
        try:
            market = exchange.markets[symbol]
            min_notional = market['limits']['cost']['min'] if 'cost' in market['limits'] and market['limits']['cost']['min'] else MIN_NOTIONAL_FALLBACK
            
            amount_to_buy_usdt = calculate_position_size_usdt(balance, symbol, last_price, atr)
            logging.info(f"Position sizing ATR pour {symbol} : {amount_to_buy_usdt:.2f} {quote_currency}")
            
            if amount_to_buy_usdt < min_notional:
                logging.warning(f"Quantité d'achat trop faible ({amount_to_buy_usdt:.2f} {quote_currency}). Min. requis: {min_notional:.2f} {quote_currency}. Ordre annulé.")
                return

            amount_to_buy = amount_to_buy_usdt / last_price
            amount_to_buy = quantize_amount(exchange, symbol, amount_to_buy)

            order = None
            if PAPER_TRADING_MODE:
                filled_amount = float(amount_to_buy)
                order = {
                    'status': 'closed',
                    'filled': filled_amount,
                    'price': last_price,
                    'fee': {'cost': filled_amount * TRANSACTION_FEES}
                }
                logging.info(f"[MODE SIMULATION] Ordre d'achat simulé pour {symbol}. Montant: {filled_amount:.4f}")
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
                    filled_amount = float(amount_to_sell)
                    order = {
                        'status': 'closed',
                        'filled': filled_amount,
                        'price': last_price,
                        'fee': {'cost': filled_amount * TRANSACTION_FEES}
                    }
                    logging.info(f"[MODE SIMULATION] Ordre de vente simulé pour {symbol}. Montant: {filled_amount:.4f}")
                else:
                    order = exchange.create_market_sell_order(symbol, amount_to_sell)
                    time.sleep(1) # Délai pour que l'ordre soit traité
                    order = exchange.fetch_order(order['id'], symbol)

                if order and order.get('filled', 0) > 0:
                    sell_price = order.get('price', last_price)
                    pnl = (sell_price - current_state['last_buy_price']) * order['filled']
                    pnl_pct = ((sell_price / current_state['last_buy_price']) - 1) * 100 if current_state['last_buy_price'] > 0 else 0.0

                    current_state['last_transaction_type'] = 'SELL'
                    current_state['last_buy_price'] = 0.0
                    current_state['max_price_since_buy'] = 0.0
                    trade_data = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'type': 'SELL',
                        'price': sell_price,
                        'amount': order['filled'],
                        'total': order['filled'] * sell_price,
                        'fees': json.dumps(order.get('fee', {}))
                    }
                    log_trade_to_csv(trade_data)
                    logging.info(f"Ordre de vente exécuté sur {symbol}. Montant vendu: {order['filled']:.4f}, Prix: {sell_price:.2f}")
                    logging.info(f"P&L réalisé sur {symbol} : {pnl:+.2f} {symbol.split('/')[1]} ({pnl_pct:+.2f}%)")
                else:
                    logging.warning(f"L'ordre de vente sur {symbol} n'a pas été exécuté correctement. Le bot reste en mode 'BUY'.")
            except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout) as e:
                logging.error(f"Erreur d'échange lors de la vente sur {symbol}: {e}")
        else:
            logging.warning(f"Signal de vente sur {symbol} mais pas de solde {base_currency} suffisant. Le bot ne fait rien.")

def run_bot_logic(exchange, symbol, state, balance):
    """Contient la logique principale du bot pour un symbole donné."""
    current_state = state[symbol]

    # Vérifier si ce symbole est en pause temporaire après trop d'échecs API
    pause_until = current_state.get('pause_until', 0)
    if time.time() < pause_until:
        remaining = int(pause_until - time.time())
        logging.info(f"{symbol} est en pause suite à des erreurs répétées. Reprise dans {remaining}s.")
        return
    elif pause_until:
        current_state.pop('pause_until', None)
    
    ohlcv = get_ohlcv(exchange, symbol, CANDLE_PERIOD, 200)
    if ohlcv is None or len(ohlcv) < max(MA_LONG_PERIOD, RSI_PERIOD, BB_PERIOD, ATR_PERIOD):
        current_state['consecutive_api_failures'] += 1
        if current_state['consecutive_api_failures'] >= 5:
            logging.critical(f"Trop d'échecs API consécutifs pour {symbol}. Ce symbole sera ignoré pendant 1 heure.")
            current_state['pause_until'] = time.time() + 3600
            current_state['consecutive_api_failures'] = 0
        logging.warning(f"Pas assez de données historiques pour l'analyse de {symbol}. En attente.")
        return
    else:
        current_state['consecutive_api_failures'] = 0
        current_state.pop('pause_until', None)

    closes = np.array([candle[4] for candle in ohlcv], dtype=float)
    highs = np.array([candle[2] for candle in ohlcv], dtype=float)
    lows = np.array([candle[3] for candle in ohlcv], dtype=float)

    # Exclure la bougie en cours (non fermée) pour éviter les faux signaux
    closes_c = closes[:-1]
    highs_c = highs[:-1]
    lows_c = lows[:-1]
    
    try:
        ma_short = talib.SMA(closes_c, timeperiod=MA_SHORT_PERIOD)[-1]
        ma_long = talib.SMA(closes_c, timeperiod=MA_LONG_PERIOD)[-1]
        rsi = talib.RSI(closes_c, timeperiod=RSI_PERIOD)[-1]
        upper_band, _, _ = talib.BBANDS(closes_c, timeperiod=BB_PERIOD, nbdevup=BB_DEVIATIONS, nbdevdn=BB_DEVIATIONS, matype=0)
        upper_band = upper_band[-1]
        atr = talib.ATR(highs_c, lows_c, closes_c, timeperiod=ATR_PERIOD)[-1]
    except Exception as e:
        logging.error(f"Erreur dans le calcul des indicateurs pour {symbol}: {e}")
        return

    if np.isnan(ma_short) or np.isnan(ma_long) or np.isnan(rsi) or np.isnan(upper_band) or np.isnan(atr):
        logging.warning(f"Indicateurs invalides (NaN) pour {symbol}, passage au cycle suivant.")
        return

    last_price = closes_c[-1]  # dernier prix de bougie fermée
    
    # Étape 3 : Logique de la stratégie
    if current_state['last_transaction_type'] == 'BUY':
        current_state['max_price_since_buy'] = max(current_state['max_price_since_buy'], last_price)
        handle_sell_signal(exchange, symbol, current_state, last_price, atr, balance)
    elif current_state['last_transaction_type'] == 'SELL':
        if ma_short > ma_long and rsi < 70 and last_price < upper_band:
            handle_buy_signal(exchange, symbol, current_state, last_price, atr, balance)

def main():
    if not API_KEY or not SECRET_KEY:
        logging.critical("Les clés API n'ont pas été trouvées. Veuillez les définir comme variables d'environnement.")
        return

    state = load_state()

    try:
        try:
            exchange = create_exchange(API_KEY, SECRET_KEY, EXCHANGE_TIMEOUT)
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout):
            logging.critical("Le bot s'arrête faute de connexion à Binance.")
            return
        logging.info("Lancement du bot de trading...")

        while True:
            try:
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
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                logging.error(f"Erreur réseau temporaire dans la boucle principale : {e}. Nouvel essai dans {CHECK_INTERVAL_SECONDS}s.")
                save_state(state)
                time.sleep(CHECK_INTERVAL_SECONDS)

    except Exception as e:
        logging.critical(f"Une erreur fatale est survenue : {e}. Le bot s'arrête.")
        save_state(state)
        return

if __name__ == "__main__":
    main()