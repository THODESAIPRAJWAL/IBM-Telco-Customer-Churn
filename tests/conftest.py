import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from churn import data  # noqa: E402

ARTIFACTS = (ROOT / "models" / "churn_model.joblib").exists() and (ROOT / "models" / "model_metadata.json").exists()
needs_artifacts = pytest.mark.skipif(not ARTIFACTS, reason="run `python run_pipeline.py` first")


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    return data.load_clean()


@pytest.fixture(scope="session")
def sample(df) -> pd.DataFrame:
    return df.sample(300, random_state=0).reset_index(drop=True)
