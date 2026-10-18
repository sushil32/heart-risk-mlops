#!/usr/bin/env bash
# Render REPORT.md -> report/REPORT.docx (re-run after adding new screenshots).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p report
pandoc REPORT.md --resource-path=. --toc --toc-depth=2 -o report/REPORT.docx
echo "wrote report/REPORT.docx"
