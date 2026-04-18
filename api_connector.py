"""
api_connector.py — Factory de connexion à l'échange via ccxt.

Fournit une fonction unique ``create_exchange`` pour instancier et initialiser
la connexion à Binance (ou tout autre échange supporté par ccxt). En centralisant
la création de l'échange ici, les paramètres de connexion (clés API, timeout,
rate-limit…) sont définis en un seul endroit, ce qui facilite les tests et les
changements de configuration.
"""

import logging
import ccxt


def create_exchange(api_key: str, secret_key: str, timeout_ms: int = 30000) -> ccxt.Exchange:
    """Crée et retourne une instance ccxt.binance prête à l'emploi.

    Args:
        api_key:    Clé API Binance.
        secret_key: Clé secrète Binance.
        timeout_ms: Délai maximal d'attente pour les requêtes HTTP, en millisecondes.
                    Par défaut : 30 000 ms (30 s).

    Returns:
        Une instance ``ccxt.binance`` avec les marchés chargés.

    Raises:
        ccxt.NetworkError: Si la connexion réseau échoue lors du chargement des marchés.
        ccxt.ExchangeError: En cas d'erreur retournée par l'échange.
    """
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': secret_key,
        'enableRateLimit': True,
        'timeout': timeout_ms,
        # Corrige automatiquement les écarts d'horloge entre le client et Binance
        'options': {'adjustForTimeDifference': True},
    })

    try:
        exchange.load_markets()
        logging.info("Connexion à Binance réussie — marchés chargés.")
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        logging.critical(f"Impossible de charger les marchés depuis Binance : {e}")
        raise

    return exchange
