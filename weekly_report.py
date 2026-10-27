"""
Rex Clemens Capital — Rapport Hebdomadaire
============================================
Genere un PDF + CSV avec l'etat du trading et les envoie par email.
Usage: python weekly_report.py
Cron: chaque dimanche 20h UTC
"""

import json
import csv
import os
import smtplib
import io
from datetime import datetime, timezone, timedelta
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

DEST_EMAIL = "agbalenyoclementkokou@gmail.com"
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")

DATA_DIR = Path("data")
ETAT_PATH = DATA_DIR / "etat_realtime.json"
METRICS_PATH = DATA_DIR / "metrics_daily.jsonl"
EQUITY_PATH = DATA_DIR / "equity_history.jsonl"
EVENTS_PATH = DATA_DIR / "events.jsonl"
TRADES_PATH = DATA_DIR / "trade_history.json"
REPORT_DIR = Path("data/reports")


def charger_json(path):
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def charger_jsonl(path, max_days=7):
    if not path.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_days)
    lines = []
    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            lines.append(entry)
        except json.JSONDecodeError:
            continue
    return lines


def generer_csv_trades(trades):
    output = io.StringIO()
    if not trades:
        return ""
    keys = ["date", "sym", "nom", "action", "prix", "montant", "pnl", "capital", "raison"]
    writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for t in trades:
        writer.writerow(t)
    return output.getvalue()


def generer_rapport_texte(etat, metrics_list, trades_semaine):
    now = datetime.now(timezone.utc)
    semaine = now.strftime("Semaine du %d/%m/%Y")

    capital = etat.get("capital", 0)
    capital_init = etat.get("capital_initial", 500)
    pnl = capital - capital_init
    pnl_pct = (pnl / capital_init * 100) if capital_init > 0 else 0
    nb_trades = len(trades_semaine)
    wins = sum(1 for t in trades_semaine if t.get("pnl", 0) > 0)
    losses = sum(1 for t in trades_semaine if t.get("pnl", 0) < 0)
    win_rate = (wins / nb_trades * 100) if nb_trades > 0 else 0
    total_pnl = sum(t.get("pnl", 0) for t in trades_semaine if "pnl" in t)

    last_metrics = metrics_list[-1] if metrics_list else {}
    sharpe = last_metrics.get("sharpe_annualized", "N/A")
    expectancy = last_metrics.get("expectancy", "N/A")
    profit_factor = last_metrics.get("profit_factor", "N/A")

    best_trade = max(trades_semaine, key=lambda t: t.get("pnl", 0)) if trades_semaine else {}
    worst_trade = min(trades_semaine, key=lambda t: t.get("pnl", 0)) if trades_semaine else {}

    rapport = f"""
{'='*60}
  REX CLEMENS CAPITAL — RAPPORT HEBDOMADAIRE
  {semaine}
{'='*60}

1. ETAT DES LIEUX
{'─'*40}
  Capital actuel:     ${capital:.2f}
  Capital initial:    ${capital_init:.2f}
  PnL total:          {pnl:+.2f}$ ({pnl_pct:+.1f}%)
  Plus haut:          ${etat.get('plus_haut', capital):.2f}
  Drawdown actuel:    {etat.get('drawdown', 0):.1f}%
  Phase:              {etat.get('phase_nom', 'N/A')}
  Module survie:      ACTIF

2. PERFORMANCE SEMAINE
{'─'*40}
  Trades executes:    {nb_trades}
  Gagnants:           {wins} ({win_rate:.0f}%)
  Perdants:           {losses}
  PnL semaine:        {total_pnl:+.2f}$
  Sharpe annualise:   {sharpe}
  Expectancy:         {expectancy}
  Profit Factor:      {profit_factor}

3. MEILLEURS / PIRES TRADES
{'─'*40}
  Meilleur:  {best_trade.get('nom', 'N/A')} — {best_trade.get('pnl', 0):+.2f}$
  Pire:      {worst_trade.get('nom', 'N/A')} — {worst_trade.get('pnl', 0):+.2f}$

4. AMELIORATIONS RECOMMANDEES
{'─'*40}"""

    if win_rate < 50:
        rapport += "\n  [!] Win rate < 50% — filtrer les signaux faibles, augmenter seuils ML"
    if sharpe != "N/A" and float(sharpe) < 0:
        rapport += "\n  [!] Sharpe negatif — le systeme ne bat pas le risque, revoir strategie"
    if total_pnl < 0:
        rapport += "\n  [!] Semaine en perte — reduire le sizing, passer en mode observation"
    if nb_trades > 100:
        rapport += "\n  [!] Trop de trades — suractivite detectee, augmenter cooldown"
    if nb_trades < 5:
        rapport += "\n  [!] Peu de trades — verifier les seuils ML, elargir l'univers"
    if not any([win_rate < 50, total_pnl < 0, nb_trades > 100, nb_trades < 5]):
        rapport += "\n  [OK] Performance stable — maintenir les parametres actuels"

    rapport += f"""

5. OPTIMISATION TOKENS IA
{'─'*40}
  DeepSeek API:       ~$0.001/requete (deepseek-chat)
  Budget mensuel:     ~$3-5 pour 3000-5000 requetes
  Frequence duels:    25% des scans (optimise)
  Estimation mois:    ~{nb_trades * 4 * 0.25:.0f} appels DeepSeek = ~${nb_trades * 4 * 0.25 * 0.001:.2f}
  Recommandation:     Garder le rate a 25%, suffisant pour validation

{'='*60}
  Rapport genere le {now.strftime('%d/%m/%Y a %H:%M UTC')}
  Rex Clemens Capital V3.1 — Module de Survie ACTIF
{'='*60}
"""
    return rapport


def generer_pdf(rapport_texte, output_path):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_LEFT

        doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                                topMargin=2*cm, bottomMargin=2*cm,
                                leftMargin=2*cm, rightMargin=2*cm)
        styles = getSampleStyleSheet()
        mono = ParagraphStyle("Mono", parent=styles["Normal"],
                              fontName="Courier", fontSize=8, leading=10,
                              alignment=TA_LEFT)
        story = []
        for line in rapport_texte.split("\n"):
            line = line.replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")
            story.append(Paragraph(line or "&nbsp;", mono))
        doc.build(story)
        return True
    except ImportError:
        with open(output_path.with_suffix(".txt"), "w") as f:
            f.write(rapport_texte)
        print("[REPORT] reportlab non disponible — rapport TXT genere")
        return False


def envoyer_email(rapport_texte, pdf_path, csv_content, semaine_label):
    if not SMTP_USER or not SMTP_PASS:
        print("[EMAIL] SMTP non configure — rapport sauvegarde localement")
        return False

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = DEST_EMAIL
    msg["Subject"] = f"Rex Clemens — Rapport {semaine_label}"

    msg.attach(MIMEText(rapport_texte, "plain"))

    if pdf_path and pdf_path.exists():
        with open(pdf_path, "rb") as f:
            part = MIMEBase("application", "pdf")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={pdf_path.name}")
            msg.attach(part)

    if csv_content:
        part = MIMEBase("text", "csv")
        part.set_payload(csv_content.encode("utf-8"))
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f"attachment; filename=rex_trades_{semaine_label.replace('/', '-')}.csv")
        msg.attach(part)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, DEST_EMAIL, msg.as_string())
        print(f"[EMAIL] Rapport envoye a {DEST_EMAIL}")
        return True
    except Exception as e:
        print(f"[EMAIL] Erreur envoi: {e}")
        return False


def main():
    now = datetime.now(timezone.utc)
    semaine_label = now.strftime("%Y-W%W")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    etat = charger_json(ETAT_PATH)
    metrics_list = charger_jsonl(METRICS_PATH, max_days=7)

    all_trades = []
    if TRADES_PATH.exists():
        try:
            all_trades = json.loads(TRADES_PATH.read_text())
        except Exception:
            pass

    cutoff = now - timedelta(days=7)
    trades_semaine = []
    for t in all_trades:
        ts = t.get("ts_utc", "")
        if ts:
            try:
                trade_dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
                if trade_dt >= cutoff:
                    trades_semaine.append(t)
            except ValueError:
                pass

    rapport = generer_rapport_texte(etat, metrics_list, trades_semaine)
    print(rapport)

    pdf_path = REPORT_DIR / f"rex_rapport_{semaine_label}.pdf"
    generer_pdf(rapport, pdf_path)

    csv_content = generer_csv_trades(trades_semaine)
    csv_path = REPORT_DIR / f"rex_trades_{semaine_label}.csv"
    with open(csv_path, "w", newline="") as f:
        f.write(csv_content)

    envoyer_email(rapport, pdf_path, csv_content, semaine_label)

    print(f"\n[REPORT] Rapport sauvegarde: {pdf_path}")
    print(f"[REPORT] CSV sauvegarde: {csv_path}")


if __name__ == "__main__":
    main()
