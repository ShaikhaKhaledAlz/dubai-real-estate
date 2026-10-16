"""
Automatic checks that the whole project runs and the numbers make sense.
GitHub runs these on every change (see .github/workflows/check.yml).
Run them yourself with:  pytest
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analysis  # noqa: E402
import market_overview  # noqa: E402
import model  # noqa: E402


def test_market_overview_runs():
    market_overview.main()
    assert (ROOT / "images" / "market_monthly.png").exists()


def test_analysis_numbers_are_sensible():
    m, _ = analysis.area_metrics(analysis.load_sales(), analysis.load_rents())
    assert len(m) >= 10                                   # enough areas to compare
    assert m["gross_yield_latest"].between(0.02, 0.15).all()   # yields between 2% and 15%
    assert m["volatility"].gt(0).all()


def test_model_runs():
    model.main()
    assert (ROOT / "data" / "processed" / "possible_undervalued_deals.csv").exists()


def test_dashboard_runs():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert len(app.metric) == 3
