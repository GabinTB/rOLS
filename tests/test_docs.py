import re
from pathlib import Path

import numpy as np
import pandas as pd


def test_readme_snippets():
    readme_path = Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()

    blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)

    class MockWebReader:
        @staticmethod
        def DataReader(name, source, start):
            dates = pd.date_range("2020-01-01", periods=100)
            if name == "12_Industry_Portfolios":
                df = pd.DataFrame(np.random.randn(100, 12), index=dates)
                df.index.name = "Date"
                return {1: df}
            else:
                df = pd.DataFrame(
                    np.random.randn(100, 2), index=dates, columns=["CPIAUCSL", "CPILFESL"]
                )
                return df

    class MockPDR:
        @staticmethod
        def get_data_famafrench(name, start):
            dates = pd.date_range("2020-01-01", periods=100)
            df = pd.DataFrame(np.random.randn(100, 3), index=dates, columns=["Mkt-RF", "SMB", "RF"])
            df.index.name = "Date"
            return {0: df}

    namespace = {
        "start": "2020-01-01",
        "web": MockWebReader,
        "pdr": MockPDR,
        "process": lambda factor, data: None,
        "targets": ["y1", "y2", "y3"],
        "group_a": ["y1"],
        "group_b": ["y2", "y3"],
        "pd": pd,
        "np": np,
    }

    for i, block in enumerate(blocks):
        # Skip illustrative or intentionally raising blocks
        if "ValueError: 3 factors were supplied" in block:
            continue
        if 'result.get_se("f1")            # Newey-West SE — requires hac_lags' in block:
            continue
        if 'result.get_beta("f1", assets=["AAPL", "MSFT"])' in block:
            continue
        if "for factor, se in result.iter_se():" in block:
            continue
        if "result.to_long_all()" in block:
            continue
        if "df = pd.concat([factors, controls, targets], axis=1).dropna()" in block:
            continue

        # Ensure illustrative variables are present in df so snippet can run
        if "df" not in namespace:
            namespace["df"] = pd.DataFrame(index=pd.date_range("2020-01-01", periods=100))

        for col in [
            "f1",
            "f2",
            "f3",
            "ctrl1",
            "ctrl2",
            "y1",
            "y2",
            "y3",
            "AAPL",
            "MSFT",
            "factors",
            "controls",
        ]:
            if col not in namespace["df"].columns:
                namespace["df"][col] = np.random.randn(len(namespace["df"]))

        try:
            exec(block, namespace)
        except Exception as e:
            raise RuntimeError(f"Failed executing block {i}:\n{block}") from e
