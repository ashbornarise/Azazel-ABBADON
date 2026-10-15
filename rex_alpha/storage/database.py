"""
Stockage SQLite — Rex Alpha Engine
Journal d'audit immuable + portefeuille + historique.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RexDB:
    def __init__(self, path: Path | str = "data/rex_alpha/rex_alpha.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        c = self.conn.cursor()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS prix_quotidien (
            symbole TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            adj_close REAL,
            PRIMARY KEY (symbole, date)
        );

        CREATE TABLE IF NOT EXISTS fondamentaux (
            symbole TEXT NOT NULL,
            date_publication TEXT NOT NULL,
            metric TEXT NOT NULL,
            valeur REAL,
            source TEXT,
            PRIMARY KEY (symbole, date_publication, metric)
        );

        CREATE TABLE IF NOT EXISTS signaux (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbole TEXT NOT NULL,
            type_signal TEXT NOT NULL,
            score REAL,
            details TEXT,
            version_modele TEXT
        );

        CREATE TABLE IF NOT EXISTS ordres (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbole TEXT NOT NULL,
            sens TEXT NOT NULL CHECK(sens IN ('ACHAT', 'VENTE')),
            quantite REAL NOT NULL,
            prix_attendu REAL NOT NULL,
            prix_execute REAL,
            frais REAL DEFAULT 0,
            slippage REAL DEFAULT 0,
            raison TEXT,
            statut TEXT DEFAULT 'SIMULE',
            version_strategie TEXT
        );

        CREATE TABLE IF NOT EXISTS portefeuille (
            symbole TEXT PRIMARY KEY,
            quantite REAL NOT NULL,
            prix_moyen REAL NOT NULL,
            date_entree TEXT NOT NULL,
            valeur_actuelle REAL DEFAULT 0,
            pnl_latent REAL DEFAULT 0,
            poids REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS etat_portefeuille (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            capital_total REAL NOT NULL,
            cash REAL NOT NULL,
            valeur_positions REAL NOT NULL,
            nb_positions INTEGER NOT NULL,
            pnl_jour REAL DEFAULT 0,
            pnl_total REAL DEFAULT 0,
            drawdown REAL DEFAULT 0,
            exposition REAL DEFAULT 0,
            benchmark_valeur REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS journal_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            details TEXT NOT NULL,
            version TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_prix_symbole ON prix_quotidien(symbole);
        CREATE INDEX IF NOT EXISTS idx_prix_date ON prix_quotidien(date);
        CREATE INDEX IF NOT EXISTS idx_signaux_symbole ON signaux(symbole);
        CREATE INDEX IF NOT EXISTS idx_ordres_symbole ON ordres(symbole);
        """)
        self.conn.commit()

    # ─── Prix ────────────────────────────────────────────

    def insert_prix(self, symbole: str, date: str, o: float, h: float,
                    l: float, c: float, vol: int, adj: float | None = None):
        self.conn.execute(
            "INSERT OR REPLACE INTO prix_quotidien VALUES (?,?,?,?,?,?,?,?)",
            (symbole, date, o, h, l, c, vol, adj or c),
        )
        self.conn.commit()

    def insert_prix_bulk(self, rows: list[tuple]):
        self.conn.executemany(
            "INSERT OR REPLACE INTO prix_quotidien VALUES (?,?,?,?,?,?,?,?)", rows
        )
        self.conn.commit()

    def get_prix(self, symbole: str, depuis: str | None = None) -> list[dict]:
        q = "SELECT * FROM prix_quotidien WHERE symbole=?"
        params: list[Any] = [symbole]
        if depuis:
            q += " AND date>=?"
            params.append(depuis)
        q += " ORDER BY date"
        return [dict(r) for r in self.conn.execute(q, params).fetchall()]

    def get_symboles(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT symbole FROM prix_quotidien ORDER BY symbole"
        ).fetchall()
        return [r["symbole"] for r in rows]

    def nb_jours(self, symbole: str) -> int:
        r = self.conn.execute(
            "SELECT COUNT(*) as n FROM prix_quotidien WHERE symbole=?", (symbole,)
        ).fetchone()
        return r["n"]

    # ─── Signaux ─────────────────────────────────────────

    def insert_signal(self, symbole: str, type_signal: str, score: float,
                      details: dict | None = None, version: str = "0.1"):
        self.conn.execute(
            "INSERT INTO signaux (timestamp, symbole, type_signal, score, details, version_modele) "
            "VALUES (?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), symbole, type_signal,
             score, json.dumps(details or {}), version),
        )
        self.conn.commit()

    def get_signaux(self, symbole: str | None = None, limit: int = 50) -> list[dict]:
        if symbole:
            rows = self.conn.execute(
                "SELECT * FROM signaux WHERE symbole=? ORDER BY id DESC LIMIT ?",
                (symbole, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM signaux ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ─── Ordres ──────────────────────────────────────────

    def insert_ordre(self, symbole: str, sens: str, quantite: float,
                     prix_attendu: float, prix_execute: float | None = None,
                     frais: float = 0, slippage: float = 0,
                     raison: str = "", version: str = "0.1"):
        self.conn.execute(
            "INSERT INTO ordres (timestamp, symbole, sens, quantite, prix_attendu, "
            "prix_execute, frais, slippage, raison, version_strategie) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), symbole, sens, quantite,
             prix_attendu, prix_execute or prix_attendu, frais, slippage, raison, version),
        )
        self.conn.commit()

    def get_ordres(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM ordres ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ─── Portefeuille ────────────────────────────────────

    def set_position(self, symbole: str, quantite: float, prix_moyen: float,
                     date_entree: str):
        if quantite <= 0:
            self.conn.execute("DELETE FROM portefeuille WHERE symbole=?", (symbole,))
        else:
            self.conn.execute(
                "INSERT OR REPLACE INTO portefeuille VALUES (?,?,?,?,0,0,0)",
                (symbole, quantite, prix_moyen, date_entree),
            )
        self.conn.commit()

    def get_positions(self) -> list[dict]:
        return [dict(r) for r in
                self.conn.execute("SELECT * FROM portefeuille ORDER BY symbole").fetchall()]

    def snapshot_portefeuille(self, capital: float, cash: float, val_pos: float,
                              nb_pos: int, pnl_jour: float, pnl_total: float,
                              dd: float, expo: float, bench: float):
        self.conn.execute(
            "INSERT INTO etat_portefeuille "
            "(timestamp, capital_total, cash, valeur_positions, nb_positions, "
            "pnl_jour, pnl_total, drawdown, exposition, benchmark_valeur) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), capital, cash, val_pos,
             nb_pos, pnl_jour, pnl_total, dd, expo, bench),
        )
        self.conn.commit()

    def get_historique_portefeuille(self, limit: int = 500) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM etat_portefeuille ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ─── Journal d'audit ─────────────────────────────────

    def log(self, type_event: str, details: dict | str, version: str = "0.1"):
        if isinstance(details, dict):
            details = json.dumps(details)
        self.conn.execute(
            "INSERT INTO journal_decisions (timestamp, type, details, version) VALUES (?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), type_event, details, version),
        )
        self.conn.commit()

    def get_journal(self, type_event: str | None = None, limit: int = 100) -> list[dict]:
        if type_event:
            rows = self.conn.execute(
                "SELECT * FROM journal_decisions WHERE type=? ORDER BY id DESC LIMIT ?",
                (type_event, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM journal_decisions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
