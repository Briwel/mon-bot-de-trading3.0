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
import yaml
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

# ─── CONFIGURATION LOADING ────────────────────────────────────────────────────
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

API_KEY    = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# Alertes Telegram (optionnel — laisser vide pour désactiver)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

# ─── PARAMÈTRES DU BOT ────────────────────────────────────────────────────────
SYMBOLS                  = CONFIG['trading']['symbols']
CHECK_INTERVAL_SECONDS   = CONFIG['trading']['check_interval_seconds']
CANDLE_PERIOD            = CONFIG['indicators']['candle_period']
MA_SHORT_PERIOD          = CONFIG['indicators']['ma_short_period']
MA_LONG_PERIOD           = CONFIG['indicators']['ma_long_period']
RSI_PERIOD               = CONFIG['indicators']['rsi_period']
ATR_PERIOD               = CONFIG['indicators']['atr_period']
ATR_MULTIPLIER           = CONFIG['indicators']['atr_multiplier']
BB_PERIOD                = CONFIG['indicators']['bb_period']
BB_DEVIATIONS            = CONFIG['indicators']['bb_deviations']
TAKE_PROFIT_PERCENTAGE   = CONFIG['strategy']['take_profit_percentage']
TRANSACTION_FEES         = CONFIG['strategy']['transaction_fees']
POSITION_SIZE_PERCENTAGE = CONFIG['trading']['position_size_percentage']
MAX_DAILY_LOSS_PCT       = CONFIG['trading']['max_daily_loss_pct']
STATE_FILE               = 'bot_state.json'
TRADE_HISTORY_FILE       = 'trade_history.csv'
PAPER_TRADING_MODE       = CONFIG['trading']['paper_trading_mode']
MIN_NOTIONAL_FALLBACK    = CONFIG['trading']['min_notional_fallback']

# ─── LOGGING ──────────────────────────────────────────────────────────────────
# Format enrichi : on sait maintenant dans quelle fonction chaque log est émis
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
))
logging.getLogger().addHandler(console_handler)

# Lock global pour protéger les ressources partagées en mode multi-thread
_state_lock = threading.Lock()

# ─── ALERTES TELEGRAM ─────────────────────────────────────────────────────────
def send_telegram(message: str) -> None:
    """Envoie un message Telegram. Silencieux si le token n'est pas configuré."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=5)
    except Exception:
        pass  # Ne jamais bloquer le bot pour une alerte ratée

# ─── PERSISTANCE DE L'ÉTAT ────────────────────────────────────────────────────
def save_state(state: dict) -> None:
    """Sauvegarde l'état du bot dans bot_state.json (thread-safe)."""
    with _state_lock:
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=4)
        except IOError as e:
            logging.error(f"Erreur sauvegarde état : {e}")

def load_state() -> dict:
    """Charge l'état depuis bot_state.json. Crée un état initial si absent."""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Erreur chargement état : {e}")

    # État initial propre pour chaque paire
    state = {'daily_start_value': 0.0, 'daily_start_date': ''}
    for symbol in SYMBOLS:
        state[symbol] = {
            'last_transaction_type': 'SELL',
            'last_buy_price': 0.0,
            'max_price_since_buy': 0.0,
            'consecutive_api_failures': 0,
        }
    return state

# ─── RÉCONCILIATION D'ÉTAT ────────────────────────────────────────────────────
def reconcile_state(exchange, state: dict) -> None:
    """
    Compare bot_state.json avec le solde réel de l'exchange au démarrage.
    Corrige les incohérences (ex : état BUY mais plus de crypto sur le compte).
    """
    logging.info("Réconciliation de l'état avec les soldes réels...")
    try:
        balance = exchange.fetch_balance()
    except Exception as e:
        logging.warning(f"Impossible de réconcilier l'état : {e}")
        return

    for symbol in SYMBOLS:
        if symbol not in state:
            continue
        sym_state = state[symbol]
        base_currency = symbol.split('/')[0]
        base_free = balance['free'].get(base_currency, 0)

        if sym_state.get('last_transaction_type') == 'BUY' and base_free < 0.0001:
            logging.warning(
                f"[RÉCONCILIATION] {symbol} : état=BUY mais solde {base_currency}={base_free:.8f}. "
                f"Réinitialisation en SELL."
            )
            send_telegram(
                f"⚠️ Réconciliation {symbol} : état corrigé BUY→SELL (solde réel insuffisant)."
            )
            sym_state['last_transaction_type'] = 'SELL'
            sym_state['last_buy_price']         = 0.0
            sym_state['max_price_since_buy']    = 0.0

    save_state(state)
    logging.info("Réconciliation terminée.")

# ─── CIRCUIT BREAKER ──────────────────────────────────────────────────────────
def check_circuit_breaker(state: dict, balance: dict) -> bool:
    """
    Calcule la valeur totale du portefeuille en USDT et vérifie le drawdown journalier.
    Retourne True si le circuit breaker est déclenché (le bot doit se mettre en pause).
    """
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # Calculer la valeur totale du portefeuille en USDT
    total_usdt = balance['total'].get('USDT', 0)
    for symbol in SYMBOLS:
        base = symbol.split('/')[0]
        base_total = balance['total'].get(base, 0)
        # Estimation grossière : on ne refait pas d'appel API pour le prix ici
        # La valeur sera approximative mais suffisante pour le circuit breaker
        total_usdt += base_total  # sera affiné si nécessaire

    # Réinitialiser la valeur de référence chaque nouveau jour UTC
    if state.get('daily_start_date') != today:
        state['daily_start_date']  = today
        state['daily_start_value'] = total_usdt
        logging.info(f"Nouveau jour UTC. Valeur de référence du portefeuille : {total_usdt:.2f} USDT")
        return False

    ref = state.get('daily_start_value', 0)
    if ref <= 0:
        return False

    loss_pct = (ref - total_usdt) / ref
    if loss_pct >= MAX_DAILY_LOSS_PCT:
        msg = (
            f"🚨 CIRCUIT BREAKER : perte journalière de {loss_pct*100:.1f}% "
            f"(référence={ref:.2f}, actuel={total_usdt:.2f} USDT). "
            f"Pause d'1 heure."
        )
        logging.critical(msg)
        send_telegram(msg)
        return True

    return False

# ─── DONNÉES DE MARCHÉ ────────────────────────────────────────────────────────
def get_ohlcv(exchange, symbol: str, timeframe: str, limit: int):
    """Récupère les bougies OHLCV pour un symbole."""
    try:
        return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logging.error(f"Erreur récupération OHLCV {symbol} : {e}")
        return None

def get_account_balance(exchange):
    """Récupère le solde complet du compte."""
    try:
        return exchange.fetch_balance()
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logging.error(f"Erreur récupération solde : {e}")
        return None

def get_balance_for_currency(balance: dict, currency: str) -> dict:
    """Extrait le solde (free/used/total) pour une devise donnée."""
    return {
        'free':  balance['free'].get(currency, 0),
        'used':  balance['used'].get(currency, 0),
        'total': balance['total'].get(currency, 0),
    }

def log_current_balances(balance: dict, symbols: list) -> None:
    """Affiche en log le solde de toutes les devises concernées."""
    logging.info("--- SOLDE ACTUEL ---")
    if balance:
        currencies = set()
        for s in symbols:
            currencies.add(s.split('/')[0])
            currencies.add(s.split('/')[1])
        for currency in sorted(currencies):
            bal = get_balance_for_currency(balance, currency)
            if bal['total'] > 0.001:
                logging.info(
                    f"{currency}: Total={bal['total']:.6f} | Disponible={bal['free']:.6f}"
                )
    logging.info("--------------------")

# ─── PRÉCISION DES ORDRES ─────────────────────────────────────────────────────
def quantize_amount(exchange, symbol: str, amount: float) -> str:
    return exchange.amount_to_precision(symbol, amount)

def quantize_price(exchange, symbol: str, price: float) -> str:
    return exchange.price_to_precision(symbol, price)

# ─── HISTORIQUE CSV ───────────────────────────────────────────────────────────
def log_trade_to_csv(trade_data: dict) -> None:
    """Enregistre un trade dans trade_history.csv."""
    header_exists = os.path.exists(TRADE_HISTORY_FILE)
    with open(TRADE_HISTORY_FILE, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'symbol', 'type', 'price', 'amount', 'total', 'fees']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not header_exists:
            writer.writeheader()
        writer.writerow(trade_data)

# ─── GESTION DES ORDRES D'ACHAT ───────────────────────────────────────────────
def handle_buy_signal(exchange, symbol: str, current_state: dict, last_price: float, balance: dict) -> None:
    """Passe un ordre d'achat si les conditions de solde sont remplies."""
    quote_currency = symbol.split('/')[1]
    usdt_balance   = get_balance_for_currency(balance, quote_currency)

    # Vérification du solde AVANT tout appel API
    if usdt_balance['free'] <= MIN_NOTIONAL_FALLBACK:
        logging.warning(f"Signal d'achat {symbol} ignoré : solde {quote_currency} insuffisant ({usdt_balance['free']:.2f}).")
        return

    logging.info(f"Signal d'achat détecté pour {symbol}. Exécution de l'ordre...")
    try:
        market       = exchange.markets[symbol]
        min_notional = (
            market['limits']['cost']['min']
            if 'cost' in market['limits'] and market['limits']['cost']['min']
            else MIN_NOTIONAL_FALLBACK
        )

        amount_to_buy_usdt = usdt_balance['free'] * POSITION_SIZE_PERCENTAGE
        if amount_to_buy_usdt < min_notional:
            logging.warning(
                f"Montant d'achat trop faible ({amount_to_buy_usdt:.2f} {quote_currency}). "
                f"Min requis : {min_notional:.2f}. Ordre annulé."
            )
            return

        amount_to_buy = quantize_amount(exchange, symbol, amount_to_buy_usdt / last_price)

        if PAPER_TRADING_MODE:
            order = {
                'status': 'closed',
                'filled': float(amount_to_buy),
                'price':  last_price,
                'fee':    {'cost': float(amount_to_buy) * TRANSACTION_FEES},
            }
            logging.info(f"[SIMULATION] Achat {symbol} : {amount_to_buy} @ {last_price:.2f}")
        else:
            order = exchange.create_market_buy_order(symbol, amount_to_buy)
            time.sleep(1)
            order = exchange.fetch_order(order['id'], symbol)

        if order and order.get('filled', 0) > 0:
            current_state['last_transaction_type'] = 'BUY'
            current_state['last_buy_price']        = order.get('price', last_price)
            current_state['max_price_since_buy']   = order.get('price', last_price)

            log_trade_to_csv({
                'timestamp': datetime.now().isoformat(),
                'symbol':    symbol,
                'type':      'BUY',
                'price':     order.get('price', last_price),
                'amount':    order['filled'],
                'total':     order['filled'] * order.get('price', last_price),
                'fees':      json.dumps(order.get('fee', {})),
            })
            msg = (
                f"✅ ACHAT {symbol} | Prix: {order.get('price', last_price):.2f} | "
                f"Qtté: {order['filled']:.6f}"
            )
            logging.info(msg)
            send_telegram(msg)
        else:
            logging.warning(f"Ordre d'achat {symbol} non rempli. Bot reste en SELL.")

    except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout) as e:
        logging.error(f"Erreur échange lors de l'achat {symbol} : {e}")

# ─── GESTION DES ORDRES DE VENTE ──────────────────────────────────────────────
def handle_sell_signal(exchange, symbol: str, current_state: dict, last_price: float, atr: float, balance: dict) -> None:
    """Passe un ordre de vente si le trailing stop-loss ou le take-profit est atteint."""
    base_currency = symbol.split('/')[0]
    base_balance  = get_balance_for_currency(balance, base_currency)

    trailing_stop_price = current_state['max_price_since_buy'] - (ATR_MULTIPLIER * atr)
    take_profit_price   = current_state['last_buy_price'] * (1 + TAKE_PROFIT_PERCENTAGE + TRANSACTION_FEES)

    logging.info(
        f"Position {symbol} | Achat: {current_state['last_buy_price']:.2f} | "
        f"Max: {current_state['max_price_since_buy']:.2f} | "
        f"SL: {trailing_stop_price:.2f} | TP: {take_profit_price:.2f} | "
        f"Prix actuel: {last_price:.2f}"
    )

    if last_price > trailing_stop_price and last_price < take_profit_price:
        return  # Pas encore le moment de vendre

    if base_balance['free'] <= 0:
        logging.warning(f"Signal de vente {symbol} mais solde {base_currency} nul. Rien à vendre.")
        return

    logging.info(f"Signal de vente déclenché pour {symbol}. Exécution...")
    try:
        amount_to_sell = quantize_amount(exchange, symbol, base_balance['free'])

        if PAPER_TRADING_MODE:
            order = {
                'status': 'closed',
                'filled': float(amount_to_sell),
                'price':  last_price,
                'fee':    {'cost': float(amount_to_sell) * TRANSACTION_FEES},
            }
            logging.info(f"[SIMULATION] Vente {symbol} : {amount_to_sell} @ {last_price:.2f}")
        else:
            order = exchange.create_market_sell_order(symbol, amount_to_sell)
            time.sleep(1)
            order = exchange.fetch_order(order['id'], symbol)

        if order and order.get('filled', 0) > 0:
            current_state['last_transaction_type'] = 'SELL'
            profit = (order.get('price', last_price) - current_state['last_buy_price']) * order['filled']

            log_trade_to_csv({
                'timestamp': datetime.now().isoformat(),
                'symbol':    symbol,
                'type':      'SELL',
                'price':     order.get('price', last_price),
                'amount':    order['filled'],
                'total':     order['filled'] * order.get('price', last_price),
                'fees':      json.dumps(order.get('fee', {})),
            })
            msg = (
                f"🔴 VENTE {symbol} | Prix: {order.get('price', last_price):.2f} | "
                f"Qtté: {order['filled']:.6f} | P&L estimé: {profit:+.2f} USDT"
            )
            logging.info(msg)
            send_telegram(msg)
        else:
            logging.warning(f"Ordre de vente {symbol} non rempli. Bot reste en BUY.")

    except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout) as e:
        logging.error(f"Erreur échange lors de la vente {symbol} : {e}")

from strategy import TradingStrategy
# ... (rest of imports)

# ─── CONFIGURATION LOADING ────────────────────────────────────────────────────
# ... (load_config, CONFIG, etc.)

# Initialiser la stratégie
STRATEGY = TradingStrategy(CONFIG)

# ... (rest of the file until run_bot_logic)

# ─── LOGIQUE PRINCIPALE PAR SYMBOLE ──────────────────────────────────────────
def run_bot_logic(exchange, symbol: str, state: dict, balance: dict) -> None:
    """
    Analyse un symbole et exécute la stratégie via le module TradingStrategy.
    """
    current_state = state[symbol]

    # Récupération des bougies
    ohlcv = get_ohlcv(exchange, symbol, CANDLE_PERIOD, 200)
    min_candles = max(MA_LONG_PERIOD, RSI_PERIOD, BB_PERIOD, ATR_PERIOD) + 2

    if ohlcv is None or len(ohlcv) < min_candles:
        current_state['consecutive_api_failures'] += 1
        if current_state['consecutive_api_failures'] >= 5:
            msg = f"🚨 {symbol} : 5 échecs API consécutifs. Pause 1h."
            logging.critical(msg)
            send_telegram(msg)
            time.sleep(3600)
            current_state['consecutive_api_failures'] = 0
        logging.warning(f"Données insuffisantes pour {symbol}. En attente du prochain cycle.")
        return

    current_state['consecutive_api_failures'] = 0

    indicators = STRATEGY.calculate_indicators(ohlcv)
    if not indicators or any(np.isnan(v) for v in indicators.values() if isinstance(v, float)):
        logging.warning(f"Indicateurs invalides pour {symbol}, cycle ignoré.")
        return

    # ── Stratégie de vente ───────────────────────────────────────────────────
    if current_state['last_transaction_type'] == 'BUY':
        current_state['max_price_since_buy'] = max(current_state['max_price_since_buy'], indicators['last_price'])
        if STRATEGY.get_sell_signal(indicators, current_state):
             handle_sell_signal(exchange, symbol, current_state, indicators['last_price'], indicators['atr'], balance)

    # ── Stratégie d'achat ────────────────────────────────────────────────────
    elif current_state['last_transaction_type'] == 'SELL':
        # Vérification préalable du solde (évite les appels API inutiles)
        quote_currency = symbol.split('/')[1]
        usdt_free = get_balance_for_currency(balance, quote_currency)['free']
        if usdt_free <= MIN_NOTIONAL_FALLBACK:
            logging.warning(f"Solde {quote_currency} insuffisant pour {symbol} ({usdt_free:.2f}). Signal ignoré.")
            return

        if STRATEGY.get_buy_signal(indicators, current_state):
            logging.info(
                f"Signal d'achat détecté sur {symbol} | MA_court={indicators['ma_short_now']:.2f} > MA_long={indicators['ma_long_now']:.2f} | "
                f"RSI={indicators['rsi']:.1f} | Prix={indicators['last_price']:.2f} < BB_haute={indicators['upper_band']:.2f}"
            )
            handle_buy_signal(exchange, symbol, current_state, indicators['last_price'], balance)
        else:
            logging.info(f"{symbol} : pas de signal.")

# ─── BOUCLE PRINCIPALE ────────────────────────────────────────────────────────
def main() -> None:
    if not API_KEY or not SECRET_KEY:
        logging.critical("Clés API introuvables. Définissez BINANCE_API_KEY et BINANCE_SECRET_KEY.")
        return

    state = load_state()

    try:
        # Correction 1 : synchronisation horloge Binance
        exchange = ccxt.binance({
            'apiKey':          API_KEY,
            'secret':          SECRET_KEY,
            'enableRateLimit': True,
            'options':         {'adjustForTimeDifference': True},
        })
        exchange.load_markets()
        logging.info("Connexion à Binance réussie. Lancement du bot...")
        send_telegram("🟢 Bot de trading démarré.")

        # Correction 2 : réconciliation état vs soldes réels
        reconcile_state(exchange, state)

        while True:
            balance = get_account_balance(exchange)
            if not balance:
                logging.error("Solde inaccessible. Nouvel essai au prochain cycle.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            log_current_balances(balance, SYMBOLS)

            # Circuit breaker : vérifie le drawdown journalier
            if check_circuit_breaker(state, balance):
                time.sleep(3600)
                continue

            # Traitement parallèle des paires (une thread par symbole)
            def process_symbol(symbol):
                try:
                    run_bot_logic(exchange, symbol, state, balance)
                except Exception as e:
                    logging.error(f"Erreur inattendue sur {symbol} : {e}")

            with ThreadPoolExecutor(max_workers=len(SYMBOLS)) as executor:
                executor.map(process_symbol, SYMBOLS)

            with _state_lock:
                save_state(state)

            logging.info(f"Prochain cycle dans {CHECK_INTERVAL_SECONDS}s.")
            time.sleep(CHECK_INTERVAL_SECONDS)

    except Exception as e:
        msg = f"🔴 Erreur fatale : {e}. Le bot s'arrête."
        logging.critical(msg)
        send_telegram(msg)
        save_state(state)

if __name__ == "__main__":
    main()