"""Capture dashboard screenshots for the README (dev utility).

    streamlit run app/streamlit_app.py --server.port 8599 &
    pip install playwright
    python docs/capture_screenshots.py --port 8599      # uses the locally installed Chrome
"""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = {
    "overview": "",
    "explore": "explore",
    "models": "models",
    "predictor": "predictor",
    "campaign": "campaign",
    "segments": "segments",
}

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8501)
ap.add_argument("--out", default=str(Path(__file__).parent / "screenshots"))
ap.add_argument("--only", nargs="*")
ap.add_argument("--max-height", type=int, default=3200)
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1.5)
    for name, path in PAGES.items():
        if args.only and name not in args.only:
            continue
        page.set_viewport_size({"width": 1500, "height": 1000})
        page.goto(f"http://localhost:{args.port}/{path}")
        page.wait_for_selector("[data-testid='stMetric'], [data-testid='stVegaLiteChart']", timeout=120_000)
        # wait until Streamlit has finished the script run (no running indicator) and charts are drawn
        page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=180_000)
        page.wait_for_timeout(1500)
        # Streamlit scrolls inside its main container, so grow the viewport to the content height instead
        height = page.evaluate("document.querySelector('[data-testid=\"stMain\"]').scrollHeight")
        page.set_viewport_size({"width": 1500, "height": min(int(height) + 40, args.max_height)})
        page.wait_for_timeout(2500)
        page.screenshot(path=str(out / f"{name}.png"))
        print("saved", name)
    browser.close()
