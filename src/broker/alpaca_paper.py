"""
Rex Clemens Capital — Alpaca Paper Trading Bridge
===================================================
Connecte les signaux Rex Clemens a l'API Alpaca en mode PAPER ONLY.
Securites: pas de levier, paper uniquement, validation Clement requise pour le reel.
"""

import os
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
    from alpaca.data.live import StockDataStream, CryptoDataStream
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False

BROKER_LOG_PATH = Path("data/broker_log.jsonl")


def _log(action, detail):
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        **detail,
    }
    with open(BROKER_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"  [BROKER] {action}: {json.dumps(detail)}")


class AlpacaPaperBroker:
    """
    Bridge entre Rex Clemens et Alpaca Paper Trading.
    REGLES NON NEGOCIABLES:
    - Paper trading UNIQUEMENT (base_url pointe sur paper)
    - ZERO levier
    - ZERO argent reel sans validation explicite
    """

    PAPER_URL = "https://paper-api.alpaca.markets"

    def __init__(self, api_key: str = None, secret_key: str = None):
        if not HAS_ALPACA:
            raise ImportError("pip install alpaca-py")

        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Cles API Alpaca manquantes. "
                "Definir ALPACA_API_KEY et ALPACA_SECRET_KEY (paper trading)."
            )

        self.client = TradingClient(
            self.api_key, self.secret_key, paper=True
        )

        self._verify_paper_mode()
        self.active = True
        _log("INIT", {"mode": "PAPER", "status": "connected"})

    def _verify_paper_mode(self):
        account = self.client.get_account()
        if account.status != "ACTIVE":
            raise RuntimeError(f"Compte Alpaca non actif: {account.status}")
        _log("VERIFY", {
            "account_id": str(account.id),
            "cash": str(account.cash),
            "buying_power": str(account.buying_power),
            "paper": True,
        })

    def get_account_info(self):
        account = self.client.get_account()
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "paper": True,
        }

    def _normalize_symbol(self, sym):
        """Convertit les symboles Rex (BTC-USD) vers Alpaca (BTC/USD pour crypto, PLTR pour stocks)."""
        if sym.endswith("-USD"):
            base = sym.replace("-USD", "")
            crypto_bases = {
                "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX",
                "LINK", "DOT", "MATIC", "SHIB", "UNI", "ATOM", "LTC",
                "NEAR", "HBAR", "FIL", "APT", "ARB", "OP", "INJ",
                "RENDER", "FET", "PEPE", "SUI", "SEI",
            }
            if base in crypto_bases:
                return f"{base}/USD", "crypto"
        clean = sym.replace("-USD", "").replace("=F", "").replace("=X", "")
        return clean, "stock"

    def buy(self, sym, montant, prix_actuel, stop_loss=None, take_profit=None):
        """Execute un ACHAT paper."""
        alpaca_sym, asset_type = self._normalize_symbol(sym)
        qte = round(montant / prix_actuel, 6) if asset_type == "crypto" else int(montant / prix_actuel)

        if qte <= 0:
            _log("SKIP_BUY", {"sym": sym, "reason": "quantite_zero", "montant": montant})
            return None

        try:
            order_data = MarketOrderRequest(
                symbol=alpaca_sym,
                qty=qte if asset_type == "stock" else None,
                notional=round(montant, 2) if asset_type == "crypto" else None,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC if asset_type == "crypto" else TimeInForce.DAY,
            )
            order = self.client.submit_order(order_data)

            _log("BUY", {
                "sym": sym,
                "alpaca_sym": alpaca_sym,
                "qty": str(qte),
                "montant": round(montant, 2),
                "order_id": str(order.id),
                "status": str(order.status),
                "prix_actuel": prix_actuel,
                "sl": stop_loss,
                "tp": take_profit,
            })

            return {
                "order_id": str(order.id),
                "status": str(order.status),
                "qty": qte,
                "sym": alpaca_sym,
            }

        except Exception as e:
            _log("BUY_ERROR", {"sym": sym, "error": str(e)})
            return None

    def sell(self, sym, qte=None):
        """Execute une VENTE paper (ferme la position)."""
        alpaca_sym, asset_type = self._normalize_symbol(sym)

        try:
            if qte is None:
                self.client.close_position(alpaca_sym)
                _log("SELL_CLOSE", {"sym": sym, "alpaca_sym": alpaca_sym})
            else:
                order_data = MarketOrderRequest(
                    symbol=alpaca_sym,
                    qty=qte,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC if asset_type == "crypto" else TimeInForce.DAY,
                )
                order = self.client.submit_order(order_data)
                _log("SELL", {
                    "sym": sym,
                    "qty": str(qte),
                    "order_id": str(order.id),
                    "status": str(order.status),
                })

            return True

        except Exception as e:
            _log("SELL_ERROR", {"sym": sym, "error": str(e)})
            return False

    def get_positions(self):
        """Retourne toutes les positions Alpaca."""
        positions = self.client.get_all_positions()
        result = {}
        for pos in positions:
            result[pos.symbol] = {
                "qty": float(pos.qty),
                "avg_entry": float(pos.avg_entry_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "current_price": float(pos.current_price),
            }
        return result

    def get_orders(self, status="open"):
        """Liste les ordres."""
        orders = self.client.get_orders(filter={"status": status})
        return [{
            "id": str(o.id),
            "symbol": o.symbol,
            "side": str(o.side),
            "qty": str(o.qty),
            "status": str(o.status),
            "created_at": str(o.created_at),
        } for o in orders]

    def sync_with_rex(self, rex_positions):
        """
        Synchronise les positions Rex avec Alpaca Paper.
        Rex est le maitre, Alpaca est l'executeur.
        """
        alpaca_positions = self.get_positions()
        actions = []

        for sym, pos in rex_positions.items():
            if not pos.get("en_position"):
                continue
            alpaca_sym, _ = self._normalize_symbol(sym)
            if alpaca_sym.replace("/", "") not in {p.replace("/", "") for p in alpaca_positions}:
                actions.append(("BUY_SYNC", sym, pos.get("montant_entree", 0)))

        for alpaca_sym, apos in alpaca_positions.items():
            found = False
            for sym in rex_positions:
                norm, _ = self._normalize_symbol(sym)
                if norm.replace("/", "") == alpaca_sym.replace("/", ""):
                    if rex_positions[sym].get("en_position"):
                        found = True
                    break
            if not found:
                actions.append(("SELL_SYNC", alpaca_sym, None))

        _log("SYNC", {"actions": len(actions), "rex_active": sum(1 for p in rex_positions.values() if p.get("en_position")), "alpaca_active": len(alpaca_positions)})
        return actions
