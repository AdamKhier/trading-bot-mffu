import warnings

import pandas as pd
import pytest

pd.set_option("future.no_silent_downcasting", True)


@pytest.fixture(autouse=True)
def _filter_deprecations():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*generic' unit for NumPy timedelta.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Bitwise inversion.*bool.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Downcasting object dtype arrays.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*infer_objects.*",
            category=FutureWarning,
        )
        yield