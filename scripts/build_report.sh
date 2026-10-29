#!/bin/sh
# Build the report PDF from docs/report/report.html with headless Chrome (no extra dependency).
# Usage (repo root): scripts/build_report.sh OUT.pdf [SRC.html]   (default SRC: docs/report/report.html)   -- figures first: .venv/bin/python scripts/make_report_figs.py
# Guarantees: reads the HTML under docs/report/; writes only the given PDF; fails if Chrome is missing or the PDF is empty.
set -eu
out="$1"
src="${2:-docs/report/report.html}"
chrome="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[ -x "$chrome" ] || { echo "Chrome not found: $chrome" >&2; exit 1; }
"$chrome" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$out" "file://$(pwd)/$src" 2>/dev/null
[ -s "$out" ] || { echo "empty PDF" >&2; exit 1; }
echo "wrote $out"
