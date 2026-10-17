"""
Rex Clemens — MT5 Robot Scanner
================================
Copie les robots .ex5 dans MT5, lance le Strategy Tester pour chacun,
et capture des screenshots pour analyse.

Usage: python rex_alpha/data_engine/scan_mt5_robots.py
"""

import os
import shutil
import subprocess
import time
import configparser
from pathlib import Path
from datetime import datetime

MT5_EXE = r"C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_DATA = r"C:\Users\agbal\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075"
EXPERTS_DIR = os.path.join(MT5_DATA, "MQL5", "Experts")
TESTER_DIR = os.path.join(MT5_DATA, "Tester")
SCREENSHOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "mt5_screenshots")
ROBOTS_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "robots_telegram")

ROBOT_MARKET_MAP = {
    "XAUUSD": ["XAUUSD", "Gold", "GOLD"],
    "EURUSD": ["EURUSD", "EUR"],
    "BTCUSD": ["BTC"],
    "GBPUSD": ["GBP"],
    "USDJPY": ["JPY"],
    "EURCAD": ["EURCAD"],
    "EURJPY": ["EURJPY"],
    "NZDCHF": ["NZDCHF"],
}

TIMEFRAME_MAP = {
    "M1": "1",
    "M5": "5",
    "M15": "15",
    "M30": "30",
    "H1": "60",
    "H4": "240",
    "D1": "1440",
}


def detect_market(robot_name):
    name_upper = robot_name.upper()
    for symbol, keywords in ROBOT_MARKET_MAP.items():
        for kw in keywords:
            if kw in name_upper:
                return symbol
    if any(kw in name_upper for kw in ["SCALP", "GRID", "HEDGE", "EA", "ROBOT", "BOT"]):
        return "XAUUSD"
    return "EURUSD"


def detect_timeframe(robot_name):
    name_upper = robot_name.upper()
    for tf_label in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
        if tf_label in name_upper:
            return tf_label
    if any(kw in name_upper for kw in ["SCALP", "SCALPER"]):
        return "M5"
    if "SWING" in name_upper:
        return "H4"
    return "M15"


def create_tester_ini(ea_name, symbol, timeframe, ini_path):
    tf_code = TIMEFRAME_MAP.get(timeframe, "15")
    config = configparser.ConfigParser()
    config.optionxform = str
    config["Tester"] = {
        "Expert": ea_name,
        "Symbol": symbol,
        "Period": tf_code,
        "FromDate": "2026.06.01",
        "ToDate": "2026.10.15",
        "Model": "1",
        "Optimization": "0",
        "Visual": "1",
        "ShutdownTerminal": "1",
    }
    with open(ini_path, "w") as f:
        config.write(f)


def capture_screenshot(robot_name, idx):
    time.sleep(3)
    screenshot_path = os.path.join(SCREENSHOTS_DIR, f"{idx:03d}_{robot_name}.png")
    try:
        subprocess.run(
            ["powershell", "-Command",
             f'Add-Type -AssemblyName System.Windows.Forms; '
             f'$bmp = New-Object Drawing.Bitmap([System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width, [System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height); '
             f'$g = [Drawing.Graphics]::FromImage($bmp); '
             f'$g.CopyFromScreen(0, 0, 0, 0, $bmp.Size); '
             f'$bmp.Save("{screenshot_path}"); '
             f'$g.Dispose(); $bmp.Dispose()'],
            timeout=15, capture_output=True
        )
        if os.path.exists(screenshot_path):
            print(f"  [SCREENSHOT] {screenshot_path}")
            return screenshot_path
    except Exception as e:
        print(f"  [SCREENSHOT ERR] {e}")
    return None


def scan_robots():
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    robots = sorted([f for f in os.listdir(EXPERTS_DIR) if f.endswith(".ex5")])
    print(f"[MT5 SCANNER] {len(robots)} robots trouves dans {EXPERTS_DIR}")
    print(f"[MT5 SCANNER] Screenshots -> {SCREENSHOTS_DIR}")
    print()

    results = []
    ini_path = os.path.join(TESTER_DIR, "rex_test.ini")
    os.makedirs(TESTER_DIR, exist_ok=True)

    for idx, robot_file in enumerate(robots, 1):
        robot_name = robot_file.replace(".ex5", "")
        symbol = detect_market(robot_name)
        timeframe = detect_timeframe(robot_name)

        print(f"[{idx}/{len(robots)}] {robot_name}")
        print(f"  Marche: {symbol} | TF: {timeframe}")

        create_tester_ini(robot_name, symbol, timeframe, ini_path)

        try:
            proc = subprocess.Popen(
                [MT5_EXE, f"/config:{ini_path}"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            time.sleep(8)
            screenshot = capture_screenshot(robot_name, idx)
            time.sleep(2)

            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"],
                                   capture_output=True, timeout=10)
                except Exception:
                    pass

            time.sleep(2)

            results.append({
                "idx": idx,
                "robot": robot_name,
                "file": robot_file,
                "symbol": symbol,
                "timeframe": timeframe,
                "screenshot": screenshot,
            })

        except Exception as e:
            print(f"  [ERREUR] {e}")
            results.append({
                "idx": idx,
                "robot": robot_name,
                "file": robot_file,
                "symbol": symbol,
                "timeframe": timeframe,
                "screenshot": None,
                "error": str(e),
            })

    report_path = os.path.join(SCREENSHOTS_DIR, "rapport_scan.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Rex Clemens — Rapport Scan MT5 Robots\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Total: {len(robots)} robots\n")
        f.write(f"{'='*60}\n\n")
        for r in results:
            status = "OK" if r.get("screenshot") else "FAIL"
            f.write(f"[{status}] {r['robot']}\n")
            f.write(f"  Marche: {r['symbol']} | TF: {r['timeframe']}\n")
            if r.get("screenshot"):
                f.write(f"  Screenshot: {r['screenshot']}\n")
            if r.get("error"):
                f.write(f"  Erreur: {r['error']}\n")
            f.write("\n")

    ok = sum(1 for r in results if r.get("screenshot"))
    print(f"\n{'='*60}")
    print(f"[TERMINE] {ok}/{len(robots)} robots scannes avec screenshot")
    print(f"[RAPPORT] {report_path}")
    print(f"[SCREENSHOTS] {SCREENSHOTS_DIR}")

    return results


if __name__ == "__main__":
    scan_robots()
