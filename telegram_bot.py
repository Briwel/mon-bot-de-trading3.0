"""
Bot Telegram pour contrôler et surveiller le bot de trading.

Commandes disponibles :
  /start       — Affiche le menu principal
  /status      — Soldes et état actuel des paires
  /start_bot   — Lance le bot de trading en arrière-plan
  /stop_bot    — Arrête le bot de trading proprement
  /pause       — Met le trading en pause (analyse sans ordre)
  /resume      — Reprend le trading
  /history     — 10 derniers trades depuis trade_history.csv
  /params      — Affiche les paramètres actuels
  /set <param> <valeur> — Modifie un paramètre à chaud
  /paper on|off — Active/désactive le mode simulation
  /reset_state <SYMBOLE> — Réinitialise l'état d'une paire
  /circuit     — Affiche le drawdown journalier
  /help        — Liste des commandes

Installation :
    pip install python-telegram-bot==22.* ccxt

Démarrage :
  export TELEGRAM_BOT_TOKEN="votre_token"
  export TELEGRAM_CHAT_ID="votre_chat_id"  # optionnel : filtre d'accès
  python telegram_bot.py
"""

import os
import json
import csv
import logging
import asyncio
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)


def load_local_env_file(env_path: str = ".env") -> None:
    """Charge des variables depuis un fichier .env sans ecraser l'environnement existant."""
    file_path = Path(env_path)
    if not file_path.exists():
        return

    try:
        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        # Le bot reste operationnel meme si le .env est inaccessible.
        pass

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
load_local_env_file()

TOKEN        = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN") or "").strip()
ALLOWED_ID   = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()   # Laisser vide pour ne pas filtrer
STATE_FILE   = "bot_state.json"
HISTORY_FILE = "trade_history.csv"
BOT_SCRIPT   = "bot_trading.py"

# Paramètres modifiables à chaud (nom → valeur courante)
MUTABLE_PARAMS = {
    "ATR_MULTIPLIER":           2.0,
    "TAKE_PROFIT_PERCENTAGE":   0.03,
    "POSITION_SIZE_PERCENTAGE": 0.1,
    "MAX_DAILY_LOSS_PCT":       0.05,
    "CHECK_INTERVAL_SECONDS":   60,
}

# ─── ÉTAT INTERNE DU BOT TELEGRAM ─────────────────────────────────────────────
_trading_process: subprocess.Popen | None = None
_paused = False
_lock   = threading.Lock()

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── UTILITAIRES ──────────────────────────────────────────────────────────────
def is_authorized(update: Update) -> bool:
    """Vérifie que l'utilisateur est autorisé (si ALLOWED_ID est défini)."""
    if not ALLOWED_ID:
        return True
    return str(update.effective_chat.id) == str(ALLOWED_ID)

def load_state() -> dict:
    """Charge bot_state.json."""
    try:
        if Path(STATE_FILE).exists():
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_state(state: dict) -> None:
    """Sauvegarde bot_state.json."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def load_history(n: int = 10) -> list[dict]:
    """Charge les N derniers trades depuis trade_history.csv."""
    if not Path(HISTORY_FILE).exists():
        return []
    rows = []
    try:
        with open(HISTORY_FILE, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        pass
    return rows[-n:]

def bot_is_running() -> bool:
    """Retourne True si le processus de trading tourne."""
    global _trading_process
    if _trading_process is None:
        return False
    return _trading_process.poll() is None

def fmt_state_symbol(symbol: str, sym_state: dict) -> str:
    """Formate l'état d'une paire pour l'affichage."""
    tx_type = sym_state.get("last_transaction_type", "?")
    icon    = "🟢" if tx_type == "BUY" else "🔴"
    lines   = [f"{icon} *{symbol}* — {tx_type}"]
    if tx_type == "BUY":
        lines.append(f"  Prix d'achat : `{sym_state.get('last_buy_price', 0):.2f}` USDT")
        lines.append(f"  Max atteint  : `{sym_state.get('max_price_since_buy', 0):.2f}` USDT")
    failures = sym_state.get("consecutive_api_failures", 0)
    if failures:
        lines.append(f"  ⚠️ Échecs API : {failures}")
    return "\n".join(lines)

# ─── COMMANDES ────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    keyboard = [
        [InlineKeyboardButton("📊 Statut",    callback_data="status"),
         InlineKeyboardButton("📜 Historique", callback_data="history")],
        [InlineKeyboardButton("▶️ Démarrer",  callback_data="start_bot"),
         InlineKeyboardButton("⏹ Arrêter",   callback_data="stop_bot")],
        [InlineKeyboardButton("⏸ Pause",     callback_data="pause"),
         InlineKeyboardButton("▶️ Reprendre", callback_data="resume")],
        [InlineKeyboardButton("⚙️ Paramètres", callback_data="params"),
         InlineKeyboardButton("❓ Aide",       callback_data="help")],
    ]
    await update.message.reply_text(
        "🤖 *Panneau de contrôle — Bot de Trading*\n\nChoisissez une action :",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown",
    )

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    state   = load_state()
    running = bot_is_running()
    paused  = _paused

    status_icon = "🟢 En cours" if running else "🔴 Arrêté"
    if running and paused:
        status_icon = "⏸ En pause"

    lines = [f"*Statut du bot :* {status_icon}\n"]

    # Drawdown journalier
    ref = state.get("daily_start_value", 0)
    if ref > 0:
        lines.append(f"📅 Référence journalière : `{ref:.2f}` USDT\n")

    # État par paire
    for sym in ["BTC/USDT", "ETH/USDT"]:
        if sym in state:
            lines.append(fmt_state_symbol(sym, state[sym]))
            lines.append("")

    text = "\n".join(lines) if lines else "Aucun état disponible."
    target = update.message or update.callback_query.message
    await target.reply_text(text, parse_mode="Markdown")

async def cmd_start_bot(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    global _trading_process, _paused
    target = update.message or update.callback_query.message

    if bot_is_running():
        await target.reply_text("⚠️ Le bot de trading tourne déjà.")
        return

    try:
        _trading_process = subprocess.Popen(
            ["python", BOT_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _paused = False
        await target.reply_text(
            f"✅ Bot de trading démarré (PID {_trading_process.pid})."
        )
    except Exception as e:
        await target.reply_text(f"❌ Impossible de démarrer le bot : {e}")

async def cmd_stop_bot(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    global _trading_process
    target = update.message or update.callback_query.message

    if not bot_is_running():
        await target.reply_text("⚠️ Le bot de trading n'est pas en cours d'exécution.")
        return

    _trading_process.terminate()
    try:
        _trading_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _trading_process.kill()
    await target.reply_text("⏹ Bot de trading arrêté.")

async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    global _paused
    target = update.message or update.callback_query.message
    _paused = True
    # Écrire un flag que bot_trading.py peut lire (optionnel — voir note bas de fichier)
    Path("PAUSE_FLAG").touch()
    await target.reply_text(
        "⏸ Pause activée.\nLe bot analyse mais ne passe plus d'ordres.\nUtilisez /resume pour reprendre."
    )

async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    global _paused
    target = update.message or update.callback_query.message
    _paused = False
    Path("PAUSE_FLAG").unlink(missing_ok=True)
    await target.reply_text("▶️ Trading repris.")

async def cmd_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    trades = load_history(10)
    target = update.message or update.callback_query.message

    if not trades:
        await target.reply_text("📜 Aucun trade enregistré.")
        return

    lines = ["*📜 10 derniers trades :*\n"]
    for t in reversed(trades):
        icon  = "🟢" if t.get("type") == "BUY" else "🔴"
        ts    = t.get("timestamp", "")[:16].replace("T", " ")
        price = float(t.get("price", 0))
        qty   = float(t.get("amount", 0))
        total = float(t.get("total", 0))
        lines.append(
            f"{icon} *{t.get('type')}* {t.get('symbol')} — {ts}\n"
            f"   Prix: `{price:.2f}` | Qtté: `{qty:.6f}` | Total: `{total:.2f}` USDT"
        )

    await target.reply_text("\n".join(lines), parse_mode="Markdown")

async def cmd_params(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    lines = ["*⚙️ Paramètres actuels :*\n"]
    for k, v in MUTABLE_PARAMS.items():
        lines.append(f"  `{k}` = `{v}`")
    lines.append("\nPour modifier : `/set NOM_PARAM valeur`")
    target = update.message or update.callback_query.message
    await target.reply_text("\n".join(lines), parse_mode="Markdown")

async def cmd_set(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    args = ctx.args
    if len(args) != 2:
        await update.message.reply_text(
            "Usage : `/set NOM_PARAM valeur`\nEx : `/set ATR_MULTIPLIER 2.5`",
            parse_mode="Markdown",
        )
        return

    param, raw_val = args[0], args[1]
    if param not in MUTABLE_PARAMS:
        await update.message.reply_text(
            f"❌ Paramètre inconnu : `{param}`\n"
            f"Paramètres disponibles : {', '.join(MUTABLE_PARAMS.keys())}",
            parse_mode="Markdown",
        )
        return

    try:
        new_val = float(raw_val)
    except ValueError:
        await update.message.reply_text("❌ La valeur doit être un nombre.")
        return

    old_val = MUTABLE_PARAMS[param]
    MUTABLE_PARAMS[param] = new_val
    await update.message.reply_text(
        f"✅ `{param}` : `{old_val}` → `{new_val}`\n"
        f"⚠️ Redémarrez le bot de trading pour appliquer le changement.",
        parse_mode="Markdown",
    )

async def cmd_paper(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    args = ctx.args
    if not args or args[0] not in ("on", "off"):
        await update.message.reply_text("Usage : `/paper on` ou `/paper off`", parse_mode="Markdown")
        return
    mode = args[0] == "on"
    MUTABLE_PARAMS["PAPER_TRADING_MODE"] = mode
    icon = "🧪" if mode else "💰"
    await update.message.reply_text(
        f"{icon} Mode simulation : *{'ACTIVÉ' if mode else 'DÉSACTIVÉ'}*\n"
        f"⚠️ Redémarrez le bot de trading pour appliquer.",
        parse_mode="Markdown",
    )

async def cmd_reset_state(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    args = ctx.args
    if not args:
        await update.message.reply_text(
            "Usage : `/reset_state BTC/USDT` ou `/reset_state ETH/USDT`",
            parse_mode="Markdown",
        )
        return

    symbol = args[0].upper()
    state  = load_state()

    if symbol not in state:
        await update.message.reply_text(f"❌ Symbole `{symbol}` introuvable dans l'état.", parse_mode="Markdown")
        return

    state[symbol] = {
        "last_transaction_type": "SELL",
        "last_buy_price": 0.0,
        "max_price_since_buy": 0.0,
        "consecutive_api_failures": 0,
    }
    save_state(state)
    await update.message.reply_text(
        f"✅ État de `{symbol}` réinitialisé en SELL.", parse_mode="Markdown"
    )

async def cmd_circuit(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    state = load_state()
    ref   = state.get("daily_start_value", 0)
    date  = state.get("daily_start_date", "—")
    lines = [
        f"*🛡️ Circuit breaker*\n",
        f"Date de référence : `{date}`",
        f"Valeur de référence : `{ref:.2f}` USDT",
        f"Seuil de perte : `{MUTABLE_PARAMS.get('MAX_DAILY_LOSS_PCT', 0.05)*100:.0f}%`",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    text = (
        "*❓ Commandes disponibles :*\n\n"
        "/start — Menu principal\n"
        "/status — État des paires et du bot\n"
        "/start\\_bot — Lance le bot de trading\n"
        "/stop\\_bot — Arrête le bot de trading\n"
        "/pause — Pause (analyse sans ordres)\n"
        "/resume — Reprend le trading\n"
        "/history — 10 derniers trades\n"
        "/params — Paramètres actuels\n"
        "/set PARAM valeur — Modifie un paramètre\n"
        "/paper on|off — Mode simulation\n"
        "/reset\\_state SYMBOLE — Réinitialise une paire\n"
        "/circuit — Drawdown journalier\n"
        "/help — Cette aide"
    )
    target = update.message or update.callback_query.message
    await target.reply_text(text, parse_mode="Markdown")

# ─── CALLBACKS DES BOUTONS INLINE ─────────────────────────────────────────────
async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data

    dispatch = {
        "status":    cmd_status,
        "history":   cmd_history,
        "start_bot": cmd_start_bot,
        "stop_bot":  cmd_stop_bot,
        "pause":     cmd_pause,
        "resume":    cmd_resume,
        "params":    cmd_params,
        "help":      cmd_help,
    }
    if data in dispatch:
        await dispatch[data](update, ctx)

# ─── DÉMARRAGE ────────────────────────────────────────────────────────────────
def main() -> None:
    if not TOKEN:
        logging.critical(
            "TELEGRAM_BOT_TOKEN non defini. Configurez la variable d'environnement "
            "ou ajoutez TELEGRAM_BOT_TOKEN=... dans un fichier .env local."
        )
        return

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start",        cmd_start))
    app.add_handler(CommandHandler("status",       cmd_status))
    app.add_handler(CommandHandler("start_bot",    cmd_start_bot))
    app.add_handler(CommandHandler("stop_bot",     cmd_stop_bot))
    app.add_handler(CommandHandler("pause",        cmd_pause))
    app.add_handler(CommandHandler("resume",       cmd_resume))
    app.add_handler(CommandHandler("history",      cmd_history))
    app.add_handler(CommandHandler("params",       cmd_params))
    app.add_handler(CommandHandler("set",          cmd_set))
    app.add_handler(CommandHandler("paper",        cmd_paper))
    app.add_handler(CommandHandler("reset_state",  cmd_reset_state))
    app.add_handler(CommandHandler("circuit",      cmd_circuit))
    app.add_handler(CommandHandler("help",         cmd_help))
    app.add_handler(CallbackQueryHandler(button_handler))

    logging.info("Bot Telegram démarré. En attente de commandes...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()

# ─── NOTE : intégration PAUSE_FLAG dans bot_trading.py ────────────────────────
# Pour que /pause soit respecté par le bot de trading, ajoutez cette vérification
# dans run_bot_logic(), avant de passer les ordres :
#
#   from pathlib import Path
#   if Path("PAUSE_FLAG").exists():
#       logging.info(f"{symbol} : bot en pause, aucun ordre passé.")
#       return