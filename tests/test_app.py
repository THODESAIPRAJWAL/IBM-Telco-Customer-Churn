"""Headless smoke tests: every dashboard page renders without an exception."""

from pathlib import Path

import pytest

from conftest import needs_artifacts

pytestmark = needs_artifacts

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
PAGES = ["overview", "explore", "models", "explain", "predictor", "campaign", "segments", "batch", "report"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=120).run()
    if page != "overview":
        at.switch_page(f"app_pages/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]


def test_campaign_planner_reacts_to_assumptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=120).run()
    at.switch_page("app_pages/campaign.py").run()
    before = at.metric[0].value
    at.slider[4].set_value(0.15).run()  # much lower offer success rate -> lower profit
    assert not at.exception
    assert at.metric[0].value != before
