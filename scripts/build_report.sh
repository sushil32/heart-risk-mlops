#!/usr/bin/env bash
# Render REPORT.md -> report/REPORT.docx and report/REPORT.pdf (with embedded images).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p report /tmp/report_build

# 1. Word Document (.docx)
pandoc REPORT.md --resource-path=. --toc --toc-depth=2 -o report/REPORT.docx
echo "wrote report/REPORT.docx"

# 2. PDF Document (.pdf) with all images embedded as base64
cat << 'EOF' > /tmp/report_build/style.css
@page {
  size: A4;
  margin: 15mm 15mm 15mm 15mm;
}
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: #1f2937;
  line-height: 1.5;
  font-size: 10.5pt;
}
h1 {
  color: #1e3a8a;
  border-bottom: 2px solid #3b82f6;
  padding-bottom: 4px;
  margin-top: 24px;
  font-size: 16pt;
}
h2 {
  color: #1e40af;
  margin-top: 18px;
  font-size: 13pt;
}
img {
  max-width: 95%;
  max-height: 480px;
  height: auto;
  display: block;
  margin: 12px auto;
  border: 1px solid #e2e8f0;
  border-radius: 4px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 14px 0;
  font-size: 9.5pt;
}
th, td {
  border: 1px solid #cbd5e1;
  padding: 6px 10px;
  text-align: left;
}
th {
  background-color: #f8fafc;
  font-weight: 600;
  color: #0f172a;
}
tr:nth-child(even) {
  background-color: #f8fafc;
}
pre {
  background: #1e293b;
  color: #f8fafc;
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 9pt;
  overflow-x: auto;
}
code {
  font-family: "SFMono-Regular", Consolas, Menlo, monospace;
  background: #f1f5f9;
  padding: 2px 4px;
  border-radius: 3px;
  font-size: 9pt;
}
pre code {
  background: transparent;
  padding: 0;
  color: #a5f3fc;
}
EOF

pandoc REPORT.md --embed-resources --standalone --css=/tmp/report_build/style.css -o /tmp/report_build/report.html
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --print-to-pdf="report/REPORT.pdf" "file:///tmp/report_build/report.html" 2>&1 | tail -2
rm -rf /tmp/report_build
echo "wrote report/REPORT.pdf"
