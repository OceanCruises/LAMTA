import numpy as np
from netCDF4 import Dataset
import xarray as xr
import glob
import datetime as dt
from scipy.interpolate import griddata, RegularGridInterpolator
import warnings
from scipy.ndimage import zoom


def loadSWOTL3uv(all_days, rep, varn, unit=None, **kwargs):
    """load u,v on SWOT swaths.
    Other grid points in the "area" can be filled with Nan
    or with CMEMS velocity field to fill in the gaps.
    Inputs:
        - all_days: list of dates (['YYYYMMDD'])for loading
                    velocity fields
        - rep: data repository path
        - varn: dictionnary with variable names as they are
                stored in netcdf file
                varn = {'longitude':'XXX','latitude':'XXX','u':'XXX','v':'XXX'}
        - unit: unit for velocity outputs ('cm/s' or 'deg/d')

    Outputs:
        - field: dictionnary containing variables lon, lat, u and v

        **kwargs:
            - filled: value (float) to fill data gaps.
                      If filled is a string, loads data as defined
                      by string option (see CMEMSuv for example).
            - rep2: path if background velocities saved in an
                    other repository
    """

    def interp2d_pairs(x1, x2, z, kind="linear"):
        """
        Replacement for your old interp2d_pairs(interp2d+dfitpack).

        Same usage pattern as before:
            fu = interp2d_pairs(lon, lat, u.T, kind='linear')
            utmp = fu(xq, yq)

        Here we interpret:
            x1 -> lon (1D)
            x2 -> lat (1D)
            z  -> array shaped like (len(x2), len(x1))  i.e. (lat, lon)
                  (which is exactly what you were passing via u.T)

        Returns a function f(x, y) evaluating at PAIRS (x[i], y[i]).
        """
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        z = np.asarray(z)

        # Expect z as (lat, lon) = (len(x2), len(x1))
        if z.shape != (x2.size, x1.size):
            raise ValueError(
                f"interp2d_pairs expects z with shape (len(lat), len(lon)) = ({x2.size}, {x1.size}), "
                f"got {z.shape}. (Tip: pass u.T like before.)"
            )

        method = "linear" if kind == "linear" else "nearest"

        rgi = RegularGridInterpolator(
            (x2, x1),  # (lat, lon)
            z,
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )

        def f(x, y):
            x = np.asarray(x)
            y = np.asarray(y)
            x2b, y2b = np.broadcast_arrays(x, y)
            pts = np.column_stack([y2b.ravel(), x2b.ravel()])  # (lat, lon)
            out = rgi(pts)
            return out.reshape(x2b.shape)

        return f

    RT = 6371e3
    u_all, v_all = [], []
    date = []

    for i in range(len(all_days)):
        dayv = all_days[i]

        # SWOTswath uv
        filei2 = glob.glob(rep + "/*_" + dayv + "*_*.nc")[0]
        ds = xr.open_dataset(filei2)

        if "area" in kwargs:
            area = kwargs["area"]
            # crop data
            selection = (ds.longitude > area[0]) & (ds.longitude < area[2]) & (ds.latitude > area[1]) & (ds.latitude < area[3])
            ds = ds.where(selection)

        # crop area if masked
        lons = ds.longitude.to_masked_array()
        lats = ds.latitude.to_masked_array()
        us = ds.ugos.to_masked_array()
        vs = ds.vgos.to_masked_array()
        mask_row = np.all(np.isnan(us), axis=1)
        us = us[~mask_row, :]
        vs = vs[~mask_row, :]
        lons = lons[~mask_row, :]
        lats = lats[~mask_row, :]

        # create meshgrid
        xx = np.linspace(area[0], area[2], 500)
        yy = np.linspace(area[1], area[3], 500)
        xv, yv = np.meshgrid(xx, yy)
        u2 = griddata((lons.flatten(), lats.flatten()), us.flatten(), (xv, yv), method="linear")
        v2 = griddata((lons.flatten(), lats.flatten()), vs.flatten(), (xv, yv), method="linear")

        date.append(dt.date.toordinal(dt.datetime.strptime(dayv, "%Y%m%d")))

        # if NaN values filled with DUACS CMEMS velocity field.
        if "filled" in kwargs:
            filled = kwargs["filled"]
            Na = np.isnan(u2)
            idnan = np.argwhere(Na)

            if isinstance(filled, str):
                if filled == "CMEMSuv":
                    if "rep2" in kwargs:
                        rep2 = kwargs["rep2"]
                    else:
                        rep2 = rep

                    filei = glob.glob(rep2 + "/*_" + dayv + "_*.nc")[0]
                    file = Dataset(filei)
                    lon = file.variables[varn["longitude"]][:]
                    lat = file.variables[varn["latitude"]][:]
                    u = file.variables[varn["u"]][0, :, :].T
                    v = file.variables[varn["v"]][0, :, :].T

                    u[np.isnan(u)], v[np.isnan(v)] = 0, 0

                    # KEEP EXACTLY YOUR ORIGINAL CALL STYLE:
                    fu = interp2d_pairs(lon, lat, u.T, kind="linear")
                    fv = interp2d_pairs(lon, lat, v.T, kind="linear")

                    # Evaluate interpolant on pairs (x, y) like before
                    utmp = fu(xv[0, idnan[:, 1]], yv[idnan[:, 0], 0])
                    vtmp = fv(xv[0, idnan[:, 1]], yv[idnan[:, 0], 0])

                    u2[Na] = utmp
                    v2[Na] = vtmp
                else:
                    warnings.warn(f"Warning: 'filled' string {filled} unrecognized.")

            elif isinstance(filled, float):
                u2[Na] = filled
                v2[Na] = filled
            else:
                warnings.warn("Warning: 'filled' type must be float or str.")

        # convert to deg/d or cm/s
        if unit == "deg/d":
            u_dd = (u2 / (RT * np.cos(yv / 180 * np.pi)) * 180 / np.pi) * 24 * 60 * 60
            v_dd = (v2 * 180 / np.pi / RT) * 24 * 60 * 60
        elif unit == "cm/s":
            u_dd = u2.T * 1e2
            v_dd = v2.T * 1e2

        # if necessary remove potential bad values from interpolation
        # u_dd[u_dd < -10], v_dd[v_dd < -10] = np.nan, np.nan
        print("no bad values")

        u_all.append(u_dd.T)
        v_all.append(v_dd.T)

    u_all, v_all = np.array(u_all), np.array(v_all)
    date = np.array(date)
    field = {"lon": xv[0, :], "lat": yv[:, 0], "u": u_all, "v": v_all, "dates": date}
    return field


def loadCMEMSuv(all_days, rep, varn, unit=None):
    RT = 6371e3
    u_all = []
    v_all = []
    date = []
    for i in range(len(all_days)):
        dayv = all_days[i]
        filei = glob.glob(rep + "/*" + dayv + "*.nc")[0]
        file = Dataset(filei)
        date.append(dt.date.toordinal(dt.datetime.strptime(dayv, "%Y%m%d")))
        lon = file.variables[varn["longitude"]][:]
        lat = file.variables[varn["latitude"]][:]
        u = file.variables[varn["u"]][0, :, :].T
        v = file.variables[varn["v"]][0, :, :].T
        [Y, X] = np.meshgrid(lat, lon)
        if unit == "deg/d":
            u_dd = (u / (RT * np.cos(Y / 180 * np.pi)) * 180 / np.pi) * 24 * 60 * 60
            v_dd = (v * 180 / np.pi / RT) * 24 * 60 * 60
        elif unit == "cm/s":
            u_dd = u * 1e2
            v_dd = v * 1e2
        u_dd = u_dd.filled(np.nan)
        v_dd = v_dd.filled(np.nan)
        u_all.append(u_dd)
        v_all.append(v_dd)

    u_all = np.array(u_all)
    v_all = np.array(v_all)
    date = np.array(date)
    field = {"lon": lon, "lat": lat, "u": u_all, "v": v_all, "dates": date}
    return field


def loadDUACSuv(all_days, filei, varn, unit=None):
    RT = 6371e3
    u_all = []
    v_all = []
    date = []
    for i in range(len(all_days)):
        dayv = all_days[i]
        # filei = glob.glob(rep+"/*_"+dayv+"_*.nc")[0]
        # filei = glob.glob(rep+dayv+"_*.nc")[0]
        file = Dataset(filei)
        date.append(dt.date.toordinal(dt.datetime.strptime(dayv, "%Y%m%d")))
        lon = file.variables[varn["longitude"]][:]
        lat = file.variables[varn["latitude"]][:]
        u = file.variables[varn["u"]][i, :, :]
        v = file.variables[varn["v"]][i, :, :]

        # define longitude on -180;180 grid if not default
        if np.min(lon) == 0.0 and np.max(lon) > 359:
            lon[lon > 180] = lon[lon > 180] - 360
            sort = np.argsort(lon)
            lon = lon[sort]
            u = u[sort, :]
            v = v[sort, :]

        [Y, X] = np.meshgrid(lat, lon)
        if unit == "deg/d":
            u_dd = (u / (RT * np.cos(Y / 180 * np.pi)) * 180 / np.pi) * 24 * 60 * 60  ## convert from m/s to degree/day
            v_dd = (v * 180 / np.pi / RT) * 24 * 60 * 60  ## convert from m/s to degree/day
        elif unit == "cm/s":
            u_dd = u * 1e2
            v_dd = v * 1e2

        u_dd = u_dd.filled(np.nan)
        v_dd = v_dd.filled(np.nan)
        u_all.append(u_dd)
        v_all.append(v_dd)
    # return (u_all, u_dd)
    u_all = np.array(u_all)
    v_all = np.array(v_all)
    date = np.array(date)
    field = {"lon": lon, "lat": lat, "u": u_all, "v": v_all, "dates": date}
    return field


def loadAISuv(all_days, filei, varn, unit=None):
    RT = 6371e3
    u_all = []
    v_all = []
    date = []
    for i in range(len(all_days)):
        dayv = all_days[i]
        file = xr.open_dataset(filei)
        date.append(dt.date.toordinal(dt.datetime.strptime(dayv, "%Y%m%d")))
        dayv = dt.datetime.strptime(all_days[i], "%Y%m%d").strftime("%Y-%m-%d")
        lon = file[varn["longitude"]].values
        lat = file[varn["latitude"]].values

        u = file[varn["u"]].sel(time=dayv).transpose()
        v = file[varn["v"]].sel(time=dayv).transpose()

        # define longitude on -180;180 grid if not default
        if np.min(lon) == 0.0 and np.max(lon) > 359:
            lon[lon > 180] = lon[lon > 180] - 360
            sort = np.argsort(lon)
            lon = lon[sort]
            u = u[sort, :]
            v = v[sort, :]

        [Y, X] = np.meshgrid(lat, lon)
        # [X,Y] = np.meshgrid(lon,lat)
        if unit == "deg/d":
            u_dd = (u / (RT * np.cos(Y / 180 * np.pi)) * 180 / np.pi) * 24 * 60 * 60  ## convert from m/s to degree/day
            v_dd = (v * 180 / np.pi / RT) * 24 * 60 * 60  ## convert from m/s to degree/day
        elif unit == "cm/s":
            u_dd = u * 1e2
            v_dd = v * 1e2

        u_dd = u_dd
        v_dd = v_dd
        u_all.append(u_dd)
        v_all.append(v_dd)

    u_all = xr.concat(u_all, dim="time")
    v_all = xr.concat(v_all, dim="time")

    u_all = np.array(u_all)
    v_all = np.array(v_all)
    date = np.array(date)

    field = {"lon": lon, "lat": lat, "u": u_all, "v": v_all, "dates": date}
    return field


def loadAISuv_SWOTmask(all_days, filei, varn, unit=None, grey_layer=None):
    RT = 6371e3
    u_all = []
    v_all = []
    date = []
    for i in range(len(all_days)):
        dayv = all_days[i]
        file = xr.open_dataset(filei)
        date.append(dt.date.toordinal(dt.datetime.strptime(dayv, "%Y%m%d")))
        dayv = dt.datetime.strptime(all_days[i], "%Y%m%d").strftime("%Y-%m-%d")
        lon = file[varn["longitude"]].values
        lat = file[varn["latitude"]].values

        u = file[varn["u"]].sel(time=dayv).transpose()
        v = file[varn["v"]].sel(time=dayv).transpose()

        # define longitude on -180;180 grid if not default
        if np.min(lon) == 0.0 and np.max(lon) > 359:
            lon[lon > 180] = lon[lon > 180] - 360
            sort = np.argsort(lon)
            lon = lon[sort]
            u = u[sort, :]
            v = v[sort, :]

        [Y, X] = np.meshgrid(lat, lon)
        # [X,Y] = np.meshgrid(lon,lat)
        if unit == "deg/d":
            u_dd = (u / (RT * np.cos(Y / 180 * np.pi)) * 180 / np.pi) * 24 * 60 * 60  ## convert from m/s to degree/day
            v_dd = (v * 180 / np.pi / RT) * 24 * 60 * 60  ## convert from m/s to degree/day
        elif unit == "cm/s":
            u_dd = u * 1e2
            v_dd = v * 1e2

        u_dd = u_dd
        v_dd = v_dd
        u_all.append(u_dd)
        v_all.append(v_dd)

    u_all = xr.concat(u_all, dim="time")
    v_all = xr.concat(v_all, dim="time")

    u_all = np.array(u_all)
    v_all = np.array(v_all)
    date = np.array(date)
    grey_layer_for_mask = grey_layer.T
    factor_x = u_all.shape[1] / grey_layer_for_mask.shape[0]
    factor_y = u_all.shape[2] / grey_layer_for_mask.shape[1]
    grey_layer_resized = zoom(grey_layer_for_mask, (factor_x, factor_y), order=0)  # shape (x, y)

    # Broadcasting sur l'axe temps : grey_layer_resized[None, :, :]
    u_all_masked = np.where(np.isnan(grey_layer_resized[None, :, :]), np.nan, u_all)
    v_all_masked = np.where(np.isnan(grey_layer_resized[None, :, :]), np.nan, v_all)

    field = {
        "lon": lon,
        "lat": lat,
        "u": u_all_masked,
        "v": v_all_masked,
        "dates": date,
    }
    return field


def loadsshuv(fname, rep, varn):
    filei = glob.glob(rep + fname)[0]
    file = Dataset(filei)
    lon = file.variables[varn["longitude"]][:]
    lat = file.variables[varn["latitude"]][:]
    ssh = file.variables[varn["ssh"]][:]
    u = file.variables[varn["u"]][:]
    v = file.variables[varn["v"]][:]
    field = {"lon": lon, "lat": lat, "ssh": ssh, "u": u, "v": v}
    return field


def loadsst(fname, rep):
    filei = glob.glob(rep + fname)[0]
    file = Dataset(filei)
    lon = file.variables["lon"][:]
    lat = file.variables["lat"][:]
    sst = file.variables["analysed_sst"][:] - 273.15
    field = {"lon": lon, "lat": lat, "sst": sst}
    return field


# ajout de cette fonctionnalité pour cas des fichiers sst CLS
def loadsstCLS(fname, rep):
    filei = glob.glob(rep + fname)[0]
    file = Dataset(filei)
    lon = file.variables["NbLongitudes"][:]
    lat = file.variables["NbLatitudes"][:]
    sst = file.variables["Grid_0001"][:]
    field = {"lon": lon, "lat": lat, "sst": sst}
    return field


def loadsstSEN(fname, rep, area):
    filei = glob.glob(rep + fname)[0]
    file = xr.open_dataset(filei)
    selection = (file.lon > area[0]) & (file.lon < area[2]) & (file.lat > area[1]) & (file.lat < area[3])
    file = file.where(selection, drop=True)
    lons = file.lon.values.ravel()
    lats = file.lat.values.ravel()
    xx = np.linspace(area[0], area[2], len(file.ni))
    yy = np.linspace(area[1], area[3], len(file.nj))
    xv, yv = np.meshgrid(xx, yy)
    tempM = file.sea_surface_temperature.where(file.quality_level == 5)
    tempM = tempM - 273.15
    tempG = griddata((lons, lats), tempM.values.ravel(), (xv, yv))
    tempG[np.isnan(tempG)] = 0
    lon = xv[0, :]
    lat = yv[:, 0]
    field = {"lon": lon, "lat": lat, "sst": tempG}
    return field


def loadchl(fname, rep, varn, **kwargs):
    filei = glob.glob(rep + fname)[0]
    file = Dataset(filei)
    lon = file.variables[varn["longitude"]][:]
    lat = file.variables[varn["latitude"]][:]
    chl = file.variables[varn["chl"]][:]
    field = {"lon": lon, "lat": lat, "chl": chl}
    return field


def loadbathy(fname, repb, **kwargs):
    filei = glob.glob(repb + fname)[0]
    file = Dataset(filei)
    lon = file.variables["lon"][:]
    lat = file.variables["lat"][:]
    if "rlon" in kwargs:
        rlon = kwargs["rlon"]
        if all(num > 180 for num in rlon):
            rlon = [x - 360 for x in rlon]
        indx = [index for index, item in enumerate(lon) if (item >= rlon[0] and item <= rlon[1])]
    else:
        indx = range(0, len(lon))
    if "rlat" in kwargs:
        rlat = kwargs["rlat"]
        indy = [index for index, item in enumerate(lat) if (item >= rlat[0] and item <= rlat[1])]
    else:
        indy = range(0, len(lat))

    z = file.variables["z"][indy, indx]
    lon = lon[indx]
    lat = lat[indy]
    file.close()
    field = {"lon": lon, "lat": lat, "z": z}
    return field
