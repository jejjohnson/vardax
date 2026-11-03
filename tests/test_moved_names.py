"""Names that moved to another library raise an ``ImportError`` naming the new home.

``KalmanSmoother`` moved to filterax, which owns Kalman and ensemble methods.
vardax must not import filterax, so the old name points there instead of
re-exporting it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import vardax as vdx


def test_attribute_access_names_the_new_home():
    with pytest.raises(ImportError, match=r"filterax\.KalmanSmoother"):
        vdx.KalmanSmoother  # noqa: B018


def test_from_import_names_the_new_home():
    with pytest.raises(ImportError, match=r"filterax\.KalmanSmoother"):
        from vardax import KalmanSmoother  # noqa: F401


def test_moved_name_is_not_public():
    assert "KalmanSmoother" not in vdx.__all__


def test_importing_vardax_does_not_import_filterax():
    code = "import sys, vardax; assert 'filterax' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
