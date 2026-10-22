#!/bin/bash
# run the app headless-ish on a video with each engine and summarise the tracker trace
cd "$(dirname "$0")/.."
SRC="$1"; SEC="${2:-30}"
for e in roi tasks; do
  echo "== $e"
  .venv/Scripts/python.exe app/portal_app.py --source "$SRC" --engine $e --seconds $SEC --hud 2>&1 | grep -v "^W0000\|^INFO\|^WARNING" | tail -5
  .venv/Scripts/python.exe tools/trace_summary.py 2>&1 | tail -6
  cp logs/live_frames.jsonl logs/previous/video_$e.jsonl
done
