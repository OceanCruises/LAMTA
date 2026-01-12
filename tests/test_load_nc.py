import numpy as np
import xarray as xr

from lamta.Load_nc import loadSWOTL3uv


def _fake_swot_dataset(ny=6, nx=8):
    """Create a small synthetic SWOT-like xarray.Dataset."""
    lon = np.linspace(2.0, 6.0, nx)
    lat = np.linspace(38.0, 43.0, ny)
    lons2d, lats2d = np.meshgrid(lon, lat)

    # Simple smooth field (no NaNs initially)
    ugos = np.sin(lons2d) * 0.1
    vgos = np.cos(lats2d) * 0.1

    return xr.Dataset(
        data_vars={
            "ugos": (("y", "x"), ugos),
            "vgos": (("y", "x"), vgos),
            "longitude": (("y", "x"), lons2d),
            "latitude": (("y", "x"), lats2d),
        }
    )


def test_loadSWOTL3uv_smoke_linear_interp(monkeypatch):
    # Arrange
    day = "20240101"
    rep = "/fake/rep"
    varn = {
        "longitude": "lon",
        "latitude": "lat",
        "u": "u",
        "v": "v",
    }  # only used for CMEMS path
    area = [1.5, 37.5, 6.5, 43.5]  # [lonmin, latmin, lonmax, latmax]

    fake_ds = _fake_swot_dataset()

    # Mock file discovery and dataset loading
    def fake_glob(pattern):
        return [f"{rep}/anything_{day}_whatever.nc"]

    def fake_open_dataset(path):
        return fake_ds

    import glob as _glob

    monkeypatch.setattr(_glob, "glob", fake_glob)
    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    # Act
    field = loadSWOTL3uv([day], rep=rep, varn=varn, unit="cm/s", area=area)

    # Assert
    assert isinstance(field, dict)
    for k in ["lon", "lat", "u", "v", "dates"]:
        assert k in field

    # lon/lat should be 1D axes of the output grid (500 points)
    assert field["lon"].ndim == 1
    assert field["lat"].ndim == 1
    assert field["lon"].shape[0] == 500
    assert field["lat"].shape[0] == 500

    # u/v should be (time, y, x)
    assert field["u"].shape == (1, 500, 500)
    assert field["v"].shape == (1, 500, 500)

    # dates is (time,)
    assert field["dates"].shape == (1,)

    # With unit="cm/s", outputs should be finite-ish (allowing some NaNs from interpolation edges)
    assert np.isfinite(field["u"]).sum() > 0
    assert np.isfinite(field["v"]).sum() > 0


def test_loadSWOTL3uv_unit_deg_per_day(monkeypatch):
    day = "20240101"
    rep = "/fake/rep"
    varn = {"longitude": "lon", "latitude": "lat", "u": "u", "v": "v"}
    area = [1.5, 37.5, 6.5, 43.5]

    fake_ds = _fake_swot_dataset()

    import glob as _glob

    monkeypatch.setattr(_glob, "glob", lambda pattern: [f"{rep}/anything_{day}_whatever.nc"])
    monkeypatch.setattr(xr, "open_dataset", lambda path: fake_ds)

    field = loadSWOTL3uv([day], rep=rep, varn=varn, unit="deg/d", area=area)

    assert field["u"].shape == (1, 500, 500)
    assert field["v"].shape == (1, 500, 500)
    # deg/day magnitudes should be small; just check not all zeros and not exploding
    finite_u = field["u"][np.isfinite(field["u"])]
    assert finite_u.size > 0
    assert np.nanmax(np.abs(finite_u)) < 1e3
