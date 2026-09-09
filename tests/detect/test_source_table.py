from __future__ import annotations

import numpy as np
import pytest

from src.detect.segmentation import SourceTable


def make_table(n=3, frame=7):
    return SourceTable(
        frame=frame,
        x=np.array([50.0, 150.0, 200.0][:n]),
        y=np.array([60.0, 40.0, 190.0][:n]),
        flux=np.array([12621.9, 6063.8, 3761.4][:n]),
        peak=np.array([800.0, 400.0, 250.0][:n]),
        elongation=np.array([1.03, 1.45, 2.20][:n]),
        npix=np.array([64, 45, 41][:n]),
    )


def test_len_and_xy_shape_and_column_order():
    t = make_table()
    assert len(t) == 3
    assert t.xy.shape == (3, 2)
    # xy column order must be [x, y]: x=column, y=row.
    assert t.xy[0, 0] == pytest.approx(50.0)
    assert t.xy[0, 1] == pytest.approx(60.0)
    assert t.xy.dtype == np.float64


def test_dtypes_are_coerced_even_from_python_lists():
    """Construction must force types rather than trust the caller."""
    t = SourceTable(frame=1, x=[1, 2], y=[3, 4], flux=[5, 6],
                    peak=[7, 8], elongation=[9, 10], npix=[11, 12])
    for arr in (t.x, t.y, t.flux, t.peak, t.elongation):
        assert isinstance(arr, np.ndarray) and arr.dtype == np.float64
    assert isinstance(t.npix, np.ndarray) and t.npix.dtype == np.int64


def test_astropy_columns_with_units_do_not_leak():
    """Unit-bearing astropy Columns must become plain arrays for cKDTree."""
    from astropy.table import Column
    from astropy import units as u
    from scipy.spatial import cKDTree

    t = SourceTable(
        frame=2,
        x=Column([50.0, 150.0]),
        y=Column([60.0, 40.0]),
        flux=Column([100.0, 50.0]),
        peak=Column([9.0, 4.0]),
        elongation=Column([1.1, 1.2], unit=u.dimensionless_unscaled),
        npix=Column([64, 45], unit=u.Unit("pix2")),
    )
    assert t.npix.dtype == np.int64
    assert not hasattr(t.xy, "unit")
    cKDTree(t.xy).query(np.array([[50.0, 60.0]]))


def test_select_accepts_boolean_and_integer_and_preserves_frame():
    t = make_table()
    b = t.select(np.array([True, False, True]))
    assert len(b) == 2 and b.frame == 7
    assert b.x[0] == pytest.approx(50.0) and b.x[1] == pytest.approx(200.0)
    i = t.select(np.array([2, 0]))
    assert len(i) == 2
    assert i.x[0] == pytest.approx(200.0) and i.x[1] == pytest.approx(50.0)


def test_select_rejects_wrong_length_boolean_mask_in_chinese():
    t = make_table()
    with pytest.raises(ValueError) as exc:
        t.select(np.array([True, False]))
    msg = str(exc.value)
    assert "3" in msg and "2" in msg
    assert any("一" <= ch <= "鿿" for ch in msg)


def test_brightest_orders_by_flux_and_never_aliases():
    t = make_table()
    top = t.brightest(2)
    assert len(top) == 2
    assert top.flux[0] > top.flux[1]
    assert top.x[0] == pytest.approx(50.0)
    # The n >= len early path must also return a new object.
    allof = t.brightest(99)
    assert len(allof) == 3
    assert allof is not t
    allof.x[0] = -999.0
    assert t.x[0] != -999.0


def test_brightest_zero_and_negative():
    t = make_table()
    assert len(t.brightest(0)) == 0
    assert len(t.brightest(-1)) == 0


def test_to_records_keys_and_json_friendly_scalars():
    import json

    t = make_table(n=2)
    recs = t.to_records()
    assert len(recs) == 2
    assert set(recs[0]) == {"frame", "x", "y", "flux", "peak", "elongation", "npix"}
    json.loads(json.dumps(recs))
    assert isinstance(recs[0]["npix"], int) and not isinstance(recs[0]["npix"], np.integer)
    assert isinstance(recs[0]["x"], float)
    assert isinstance(recs[0]["frame"], int)


def test_empty_is_fully_formed():
    t = SourceTable.empty()
    assert len(t) == 0 and t.frame == -1
    assert t.xy.shape == (0, 2)
    for arr in (t.x, t.y, t.flux, t.peak, t.elongation):
        assert arr.shape == (0,) and arr.dtype == np.float64
    assert t.npix.shape == (0,) and t.npix.dtype == np.int64
    assert t.to_records() == []
    assert len(t.select(np.zeros(0, dtype=bool))) == 0
    assert len(t.brightest(3)) == 0
    assert SourceTable.empty(frame=16).frame == 16


def test_ragged_columns_are_rejected():
    """Mismatched column lengths must fail during construction."""
    with pytest.raises(ValueError):
        SourceTable(frame=0, x=np.array([1.0, 2.0]), y=np.array([1.0]),
                    flux=np.array([1.0, 2.0]), peak=np.array([1.0, 2.0]),
                    elongation=np.array([1.0, 2.0]), npix=np.array([1, 2]))
