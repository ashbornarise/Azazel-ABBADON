#!/bin/bash
# Rex Clemens Capital — Weekly Data Archive to GitHub
# Usage: crontab -e → 0 3 * * 0 /opt/rex-clemens/weekly_archive.sh
# Runs every Sunday at 3:00 AM UTC

cd /opt/rex-clemens || exit 1

DATE=$(date -u +"%Y-%m-%d")

# Archive les JSONL dans un dossier par semaine
ARCHIVE_DIR="data/archives/${DATE}"
mkdir -p "$ARCHIVE_DIR"

for f in data/*.jsonl; do
    [ -f "$f" ] || continue
    BASENAME=$(basename "$f")
    cp "$f" "$ARCHIVE_DIR/${BASENAME}"
    LINES=$(wc -l < "$f")
    echo "[ARCHIVE] $BASENAME: $LINES lignes -> $ARCHIVE_DIR/"
done

# Git commit + push
git add data/archives/
git add data/*.jsonl
git commit -m "Weekly data archive ${DATE} — $(wc -l < data/events.jsonl 2>/dev/null || echo 0) events, $(wc -l < data/equity_history.jsonl 2>/dev/null || echo 0) equity points, $(wc -l < data/cortex_logs.jsonl 2>/dev/null || echo 0) cortex logs, $(wc -l < data/signals_history.jsonl 2>/dev/null || echo 0) signals, $(wc -l < data/prices_history.jsonl 2>/dev/null || echo 0) price snapshots"

git push origin main

echo "[ARCHIVE] Done: $DATE"
