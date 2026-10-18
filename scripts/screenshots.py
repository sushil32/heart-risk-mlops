"""Capture report screenshots with Playwright (uses the locally installed Chrome).

Usage: uv run --with playwright python scripts/screenshots.py [name=url ...]
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "screenshots"
DEFAULT = {
    "06_grafana_dashboard": "http://localhost:3000/d/heart-risk-api/?orgId=1&kiosk&from=now-10m&to=now",
    "05_swagger_docs": "http://localhost:8000/docs",
}


def main() -> None:
    targets = dict(a.split("=", 1) for a in sys.argv[1:]) or DEFAULT
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
        for name, url in targets.items():
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(4000)
            page.screenshot(path=OUT / f"{name}.png", full_page=True)
            print("saved", name)
        browser.close()


if __name__ == "__main__":
    main()
