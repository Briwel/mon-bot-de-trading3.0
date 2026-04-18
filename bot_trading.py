import ccxt
import time
import talib
import numpy as np
import os
import logging
import json
import csv
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
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
CIRCUIT_BREAKER_DAILY_LOSS_PCT = 0.05   # Arrêt automatique si perte ≥ 5 % du capital en 24 h
CIRCUIT_BREAKER_WINDOW_SECONDS = 86400  # Durée de la fenêtre de référence : 24 h
CIRCUIT_BREAKER_SUSPENSION_SECONDS = 3600  # Durée de la suspension lors du déclenchement (1 h)
STATE_FILE = 'bot_state.json'
TRADE_HISTORY_FILE = 'trade_history.csv'
PAPER_TRADING_MODE = False # Mettre à True pour simuler les trades sans risques
MIN_NOTIONAL_FALLBACK = 10.0 # Valeur de secours pour le montant minimum de transaction
DUST_BALANCE_THRESHOLD = 0.0001  # Seuil en dessous duquel un solde est considéré nul ("poussière")

# --- ALERTES TELEGRAM ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_TIMEOUT_SECONDS = 5  # Délai maximum d'attente pour l'envoi d'une alerte Telegram

# Seuil maximum de perte journalière avant déclenchement du disjoncteur (alias lisible)
MAX_DAILY_LOSS_PERCENTAGE = CIRCUIT_BREAKER_DAILY_LOSS_PCT

# --- CONFIGURATION DE LA JOURNALISATION (LOGGING) ---
_LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format=_LOG_FORMAT)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(_LOG_FORMAT)
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# --- VERROUS POUR LA CONCURRENCE (ThreadPoolExecutor) ---
_exchange_lock = threading.Lock()  # Protège les appels réseau à l'échange (non thread-safe)
_state_lock = threading.Lock()     # Protège les écritures dans bot_state.json


def send_telegram(message: str) -> None:
    """Envoie une alerte Telegram si les variables d'environnement sont définies.

    Silencieux en cas d'erreur pour ne pas interrompre le trading.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': message}, timeout=TELEGRAM_TIMEOUT_SECONDS)
    except Exception:
        pass

# --- FONCTIONS UTILES ---
def save_state(state):
    """Sauvegarde l'état du bot dans un fichier JSON (thread-safe via _state_lock)."""
    with _state_lock:
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

def _default_circuit_breaker_state():
    """Retourne l'état initial du disjoncteur de perte journalière."""
    return {
        'daily_start_value': 0.0,   # Valeur du portefeuille au début du jour UTC
        'reference_time': 0.0,      # Timestamp UNIX du début de la fenêtre de 24 h
        'triggered_until': 0.0,     # Timestamp UNIX de fin de suspension (0 = inactif)
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
    # (p. ex. restes d'une ancienne version du fichier d'état).
    # La clé "circuit_breaker" est préservée intentionnellement.
    KNOWN_TOP_LEVEL_KEYS = {'circuit_breaker'}
    orphan_keys = [k for k in list(state.keys()) if k not in SYMBOLS and k not in KNOWN_TOP_LEVEL_KEYS]
    for k in orphan_keys:
        logging.info(f"Nettoyage de la clé d'état orpheline : '{k}'")
        del state[k]

    # S'assurer que chaque symbole actif a bien son entrée (p. ex. si un nouveau symbole est ajouté)
    for symbol in SYMBOLS:
        if symbol not in state:
            state[symbol] = _default_symbol_state()

    # Initialiser l'état du disjoncteur s'il est absent
    if 'circuit_breaker' not in state:
        state['circuit_breaker'] = _default_circuit_breaker_state()

    # Migrer l'ancienne clé reference_balance → daily_start_value si nécessaire
    cb = state['circuit_breaker']
    if 'reference_balance' in cb and 'daily_start_value' not in cb:
        cb['daily_start_value'] = cb.pop('reference_balance')
        logging.info("Migration circuit_breaker : reference_balance → daily_start_value appliquée.")

    return state


def reconcile_state(exchange, state):
    """Réconcilie l'état persisté avec les soldes réels de l'exchange au démarrage.

    Pour chaque symbole en état 'BUY', vérifie que le solde base_currency est réel.
    Si ≈ 0, corrige l'état vers 'SELL' et journalise un WARNING.
    Appelée une seule fois dans main() après load_state().
    """
    try:
        with _exchange_lock:
            balance = exchange.fetch_balance()
        if not balance:
            logging.warning("reconcile_state : impossible de récupérer le solde — réconciliation ignorée.")
            return
        for symbol in SYMBOLS:
            sym_state = state.get(symbol, {})
            if sym_state.get('last_transaction_type') == 'BUY':
                base_currency = symbol.split('/')[0]
                base_total = get_balance_for_currency(balance, base_currency)['total']
                if base_total < DUST_BALANCE_THRESHOLD:
                    logging.warning(
                        f"[Réconciliation] {symbol} : état persisté='BUY' "
                        f"mais solde {base_currency}={base_total:.8f} ≈ 0. "
                        f"Correction automatique → 'SELL'."
                    )
                    sym_state['last_transaction_type'] = 'SELL'
                    sym_state['last_buy_price'] = 0.0
                    sym_state['max_price_since_buy'] = 0.0
    except Exception as e:
        logging.error(f"reconcile_state : erreur lors de la réconciliation : {e}")

def check_circuit_breaker(state, total_usdt):
    """Vérifie le disjoncteur de perte journalière (drawdown).

    Retourne True si le trading doit être suspendu, False sinon.
    La fenêtre de référence est glissante sur CIRCUIT_BREAKER_WINDOW_SECONDS (24 h).
    Si (daily_start_value - valeur_actuelle) / daily_start_value > MAX_DAILY_LOSS_PERCENTAGE,
    le bot suspend tous les nouveaux ordres pendant 3 600 secondes.
    """
    cb = state.setdefault('circuit_breaker', _default_circuit_breaker_state())
    now = time.time()

    # Le disjoncteur est déjà déclenché
    if cb['triggered_until'] > now:
        remaining = int(cb['triggered_until'] - now)
        logging.warning(
            f"DISJONCTEUR ACTIF : trading suspendu pendant encore {remaining}s "
            f"({remaining // 3600}h {(remaining % 3600) // 60}min)."
        )
        return True
    elif cb['triggered_until'] > 0:
        # La période de suspension vient d'expirer : réinitialiser
        logging.info("Disjoncteur réinitialisé. Le trading reprend avec un nouveau solde de référence.")
        cb['triggered_until'] = 0.0
        cb['daily_start_value'] = total_usdt
        cb['reference_time'] = now
        return False

    # Nouvelle fenêtre de 24 h ou premier démarrage : enregistrer la valeur de référence
    if cb['daily_start_value'] <= 0 or (now - cb['reference_time']) >= CIRCUIT_BREAKER_WINDOW_SECONDS:
        if cb['daily_start_value'] > 0:
            logging.info(
                f"Nouvelle fenêtre de 24 h. Valeur journalière de référence mise à jour : {total_usdt:.2f} USDT"
            )
        cb['daily_start_value'] = total_usdt
        cb['reference_time'] = now
        return False

    # Calculer le drawdown journalier par rapport à daily_start_value
    loss_pct = (cb['daily_start_value'] - total_usdt) / cb['daily_start_value']
    if loss_pct >= MAX_DAILY_LOSS_PERCENTAGE:
        cb['triggered_until'] = now + CIRCUIT_BREAKER_SUSPENSION_SECONDS
        msg = (
            f"DISJONCTEUR DÉCLENCHÉ : perte de {loss_pct * 100:.2f}% sur 24 h "
            f"(référence : {cb['daily_start_value']:.2f} USDT → actuel : {total_usdt:.2f} USDT). "
            f"Seuil autorisé : {MAX_DAILY_LOSS_PERCENTAGE * 100:.0f}%. "
            f"Trading suspendu pendant {CIRCUIT_BREAKER_SUSPENSION_SECONDS // 60} minutes."
        )
        logging.critical(msg)
        send_telegram(f"🚨 {msg}")
        return True

    logging.info(
        f"Disjoncteur OK — perte 24 h : {loss_pct * 100:.2f}% "
        f"(seuil : {MAX_DAILY_LOSS_PERCENTAGE * 100:.0f}%)"
    )
    return False


def get_ohlcv(exchange, symbol, timeframe, limit):
    """Récupère les données OHLCV pour un symbole donné (thread-safe via _exchange_lock)."""
    try:
        with _exchange_lock:
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
                # Acquisition du verrou pour les appels réseau à l'échange (thread-safe)
                with _exchange_lock:
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
                msg = (
                    f"✅ BUY {symbol} — Prix: {order.get('price', last_price):.2f} "
                    f"| Montant: {order['filled']:.4f} | Total: {trade_data['total']:.2f} USDT"
                )
                logging.info(msg)
                send_telegram(msg)
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
                    # Acquisition du verrou pour les appels réseau à l'échange (thread-safe)
                    with _exchange_lock:
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
                    msg = (
                        f"🔴 SELL {symbol} — Prix: {sell_price:.2f} "
                        f"| Montant: {order['filled']:.4f} | P&L: {pnl:+.2f} USDT ({pnl_pct:+.2f}%)"
                    )
                    logging.info(msg)
                    send_telegram(msg)
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
        # Calcul des tableaux complets pour détecter le croisement sur les 2 dernières bougies
        ma_short_arr = talib.SMA(closes_c, timeperiod=MA_SHORT_PERIOD)
        ma_long_arr = talib.SMA(closes_c, timeperiod=MA_LONG_PERIOD)
        ma_short = ma_short_arr[-1]
        ma_long = ma_long_arr[-1]
        rsi = talib.RSI(closes_c, timeperiod=RSI_PERIOD)[-1]
        upper_band, _, _ = talib.BBANDS(closes_c, timeperiod=BB_PERIOD, nbdevup=BB_DEVIATIONS, nbdevdn=BB_DEVIATIONS, matype=0)
        upper_band = upper_band[-1]
        atr = talib.ATR(highs_c, lows_c, closes_c, timeperiod=ATR_PERIOD)[-1]
    except Exception as e:
        logging.error(f"Erreur dans le calcul des indicateurs pour {symbol}: {e}")
        return

    if any(np.isnan(v) for v in [ma_short, ma_long, ma_short_arr[-2], ma_long_arr[-2], rsi, upper_band, atr]):
        logging.warning(f"Indicateurs invalides (NaN) pour {symbol}, passage au cycle suivant.")
        return

    # Garde sur la longueur minimale des tableaux MA pour la détection du croisement
    if len(ma_short_arr) < 2 or len(ma_long_arr) < 2:
        logging.warning(f"Tableaux MA trop courts pour {symbol} (len={len(ma_short_arr)}), passage au cycle suivant.")
        return

    last_price = closes_c[-1]  # dernier prix de bougie fermée

    # Croisement haussier récent : MA courte était ≤ MA longue, maintenant > MA longue
    buy_crossover = (ma_short_arr[-2] <= ma_long_arr[-2]) and (ma_short_arr[-1] > ma_long_arr[-1])

    # Logique de la stratégie
    if current_state['last_transaction_type'] == 'BUY':
        current_state['max_price_since_buy'] = max(current_state['max_price_since_buy'], last_price)
        handle_sell_signal(exchange, symbol, current_state, last_price, atr, balance)
    elif current_state['last_transaction_type'] == 'SELL':
        # Vérifier le solde disponible avant d'envoyer un signal d'achat
        quote_currency = symbol.split('/')[1]
        free_quote = get_balance_for_currency(balance, quote_currency)['free']
        if free_quote <= MIN_NOTIONAL_FALLBACK:
            logging.warning(
                f"{symbol} : solde {quote_currency} libre insuffisant "
                f"({free_quote:.2f} ≤ {MIN_NOTIONAL_FALLBACK}). Signal d'achat ignoré."
            )
        elif buy_crossover and rsi < 70 and last_price < upper_band:
            handle_buy_signal(exchange, symbol, current_state, last_price, atr, balance)

def main():
    if not API_KEY or not SECRET_KEY:
        logging.critical("Les clés API n'ont pas été trouvées. Veuillez les définir comme variables d'environnement.")
        return

    state = load_state()

    try:
        try:
            exchange = create_exchange(API_KEY, SECRET_KEY, EXCHANGE_TIMEOUT)
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            logging.critical(f"Le bot s'arrête faute de connexion à Binance : {e}")
            return
        logging.info("Lancement du bot de trading...")

        # Réconciliation unique de l'état persisté avec les soldes réels au démarrage
        reconcile_state(exchange, state)
        save_state(state)

        while True:
            try:
                balance = get_account_balance(exchange)
                if not balance:
                    logging.error("Impossible de récupérer le solde. Nouvel essai au prochain cycle.")
                    time.sleep(CHECK_INTERVAL_SECONDS)
                    continue
                
                log_current_balances(balance, SYMBOLS)

                # --- DISJONCTEUR DE PERTE JOURNALIÈRE (drawdown 24 h) ---
                usdt_total = get_balance_for_currency(balance, 'USDT')['total']
                if check_circuit_breaker(state, usdt_total):
                    save_state(state)
                    time.sleep(CIRCUIT_BREAKER_SUSPENSION_SECONDS)
                    continue

                # Traitement parallèle de toutes les paires via ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=min(len(SYMBOLS), 4)) as executor:
                    futures = {
                        executor.submit(run_bot_logic, exchange, symbol, state, balance): symbol
                        for symbol in SYMBOLS
                    }
                    for future, sym in futures.items():
                        try:
                            future.result()
                        except Exception as e:
                            logging.error(f"Erreur non gérée dans le thread pour {sym} : {e}")
                
                save_state(state)
                logging.info(f"Prochaine analyse dans {CHECK_INTERVAL_SECONDS} secondes...")
                time.sleep(CHECK_INTERVAL_SECONDS)
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                logging.error(f"Erreur réseau temporaire dans la boucle principale : {e}. Nouvel essai dans {CHECK_INTERVAL_SECONDS}s.")
                save_state(state)
                time.sleep(CHECK_INTERVAL_SECONDS)

    except Exception as e:
        msg = f"Une erreur fatale est survenue : {e}. Le bot s'arrête."
        logging.critical(msg)
        send_telegram(f"🚨 CRITIQUE: {msg}")
        save_state(state)
        return

if __name__ == "__main__":
    main()