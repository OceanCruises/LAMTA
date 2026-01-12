from scipy.interpolate import interpn, RegularGridInterpolator

import numpy as np
import datetime as dt
import warnings
import sys
import glob

# Spasso spec ???
# import GlobalVars, Fields, PlotField, Library


class Lagrangian:
    def diag(self, diag=None, method=None, f=None, **kwargs):
        """Initialize and launch the diagnostics requested and method set to
        advect the Lagrangian particles previsouly set by ParticleSet

        :param diag: list of the requested Lagrangian diagnostics.
        diag = ['FTLE'] or ['LLADV'].
        For multiple diagnostics: diag = ['FTLE','LLADV']

        :param method: method for particle advection. Default is set to Rnga-kutta 4 'rk4flat'.

        :param f: function to get particle new position

        :output: outputs are concatenated in list 'out' starting with particle
        trajectories as a first dict. Then outputs are in the same order as
        listed in diag.
        """
        if "Library" in sys.modules.keys():
            Library.tic()
        out = []
        if "numstep" in kwargs:
            Nstep = kwargs["numstep"]
        else:
            Nstep = 4  # default value
            warntxt = "Warning: 'numstep' is not defined -> using default value (4)"
            warnings.warn(warntxt)
            if "Library" in sys.modules.keys():
                Library.Logfile(warntxt)

        if method == "rk1flat":
            trjf = self.rk1flat(f, Nstep, **kwargs)
        elif method == "rk4flat":
            trjf = self.rk4flat(f, Nstep, **kwargs)
        else:
            warntxt = "Warning: 'method' is not defined -> using default value (rk4flat)"
            warnings.warn(warntxt)
            if "Library" in sys.modules.keys():
                Library.Logfile(warntxt)
            trjf = self.rk4flat(f, Nstep, **kwargs)
        out.append(trjf)
        if "Library" in sys.modules.keys():
            Library.toc("Lagrangian trajectories")

        if diag != None:
            for i in diag:
                if i == "LLADV":
                    dd = self.LLADV(trjf, **kwargs)
                if i == "SSTADV":
                    dd = self.SSTADV(trjf, **kwargs)
                if i == "FTLE":
                    dd = self.FTLE(trjf, **kwargs)
                if i == "FTLE_drift":
                    dd = self.FTLE_drift(trjf, **kwargs)
                if i == "OWTRAJ":
                    dd = self.OWTRAJ(trjf, **kwargs)
                if i == "TIMEFROMBATHY":
                    dd = self.TIMEFROMBATHY(trjf, **kwargs)
                out.append(dd)

        return out

    def backonsphere(self, x, y):
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")

        # --- zonal (pole crossing) ---
        pos90p = y > 90
        pos90m = y < -90

        if self.PeriodicBC is False:
            if pos90p.any():
                y[pos90p] = np.nan
            if pos90m.any():
                y[pos90m] = np.nan
        else:
            if pos90p.any():
                y[pos90p] = 180 - y[pos90p]
                x[pos90p] = x[pos90p] + 180
            if pos90m.any():
                if np.all(y[pos90m] < -180):
                    y[pos90m] = np.nan
                else:
                    y[pos90m] = -180 - y[pos90m]  # FIX
                    x[pos90m] = x[pos90m] + 180

        # --- meridional (periodic lon wrap) ---
        lon = np.asarray(self.lon, dtype="float64")
        span = lon[-1] - lon[0]

        xm = x < lon[0]
        xp = x > lon[-1]

        if self.PeriodicBC is False:
            if xm.any():  # FIX: call the method
                x[xm] = np.nan
            if xp.any():  # FIX
                x[xp] = np.nan
        else:
            if xm.any():  # FIX
                x[xm] += span
            if xp.any():  # FIX
                x[xp] -= span

        return x, y

    def interpf_2fields(self, t, x, y, **kwargs):
        udim, vdim = np.asarray(self.us), np.asarray(self.vs)
        if np.size(t) != np.size(x):
            t = np.tile(t, len(x))
        if udim.ndim == 2 and vdim.ndim == 2:
            if "coordinates" in kwargs and kwargs["coordinates"] == "spherical":
                xn, yn = Lagrangian.backonsphere(self, x, y)
                new_grid = list(zip(xn, yn))
            else:
                new_grid = list(zip(x, y))
            if self.lons.ndim == 2:
                lon, lat = self.lons[:, 0], self.lats[0, :]
            else:
                lon, lat = self.lons, self.lats
            us_in = interpn((lon, lat), self.us, new_grid, bounds_error=False, fill_value=np.nan)
            vs_in = interpn((lon, lat), self.vs, new_grid, bounds_error=False, fill_value=np.nan)
        elif udim.ndim == 3 and udim.ndim == 3:
            if "coordinates" in kwargs and kwargs["coordinates"] == "spherical":
                xn, yn = Lagrangian.backonsphere(self, x, y)
                new_grid = list(zip(t, xn, yn))
            else:
                new_grid = list(zip(t, x, y))
            us_in = interpn(
                (self.dates, self.lons[0, :], self.lats[:, 0]),
                self.us,
                new_grid,
                bounds_error=False,
                fill_value=np.nan,
            )
            vs_in = interpn(
                (self.dates, self.lons[0, :], self.lats[:, 0]),
                self.vs,
                new_grid,
                bounds_error=False,
                fill_value=np.nan,
            )
        return us_in, vs_in

    def _unique_sorted_axis(axis, A, B=None, axis_index=0):
        axis = np.asarray(axis).astype(float)

        order = np.argsort(axis)
        axis_s = axis[order]
        A_s = np.take(A, order, axis=axis_index)
        B_s = np.take(B, order, axis=axis_index) if B is not None else None

        axis_u, first_idx = np.unique(axis_s, return_index=True)  # removes duplicates
        A_u = np.take(A_s, first_idx, axis=axis_index)
        B_u = np.take(B_s, first_idx, axis=axis_index) if B_s is not None else None

        return axis_u, A_u, B_u

    def interpf(self, t, x, y, **kwargs):
        udim = np.asarray(self.u_nonan)
        vdim = np.asarray(self.v_nonan)

        if np.size(t) != np.size(x):
            t = np.tile(t, len(x))

        # -------------------------
        # 2D velocity field
        # -------------------------
        if udim.ndim == 2 and vdim.ndim == 2:
            # coordinates handling
            if "coordinates" in kwargs and kwargs["coordinates"] == "spherical":
                xg, yg = Lagrangian.backonsphere(self, x, y)
                new_grid = list(zip(xg, yg))
            else:
                new_grid = list(zip(x, y))

            # get lon / lat axes
            if self.lon.ndim == 2:
                lon = np.asarray(self.lon[:, 0]).astype(float)
                lat = np.asarray(self.lat[0, :]).astype(float)
            else:
                lon = np.asarray(self.lon).astype(float)
                lat = np.asarray(self.lat).astype(float)

            U = np.asarray(self.u_nonan)
            V = np.asarray(self.v_nonan)

            # handle (lat,lon) vs (lon,lat)
            if U.shape == (len(lat), len(lon)):
                U = U.T
                V = V.T

            # ---- enforce strictly monotone axes (interpn requirement) ----
            # longitude
            ix = np.argsort(lon)
            lon = lon[ix]
            U = U[ix, :]
            V = V[ix, :]

            lon, iuniq = np.unique(lon, return_index=True)
            U = U[iuniq, :]
            V = V[iuniq, :]

            # latitude
            iy = np.argsort(lat)
            lat = lat[iy]
            U = U[:, iy]
            V = V[:, iy]

            lat, iuniq = np.unique(lat, return_index=True)
            U = U[:, iuniq]
            V = V[:, iuniq]

            u_in = interpn((lon, lat), U, new_grid, bounds_error=False, fill_value=np.nan)
            v_in = interpn((lon, lat), V, new_grid, bounds_error=False, fill_value=np.nan)

        # -------------------------
        # 3D velocity field (time, lon, lat)
        # -------------------------
        elif udim.ndim == 3 and vdim.ndim == 3:
            if "coordinates" in kwargs and kwargs["coordinates"] == "spherical":
                xg, yg = Lagrangian.backonsphere(self, x, y)
                new_grid = list(zip(t, xg, yg))
            else:
                new_grid = list(zip(t, x, y))

            u_in = interpn(
                (self.dates, self.lon, self.lat),
                self.u_nonan,
                new_grid,
                bounds_error=False,
                fill_value=np.nan,
            )
            v_in = interpn(
                (self.dates, self.lon, self.lat),
                self.v_nonan,
                new_grid,
                bounds_error=False,
                fill_value=np.nan,
            )

        else:
            raise ValueError("Unsupported dimensions for u/v fields")

        return u_in, v_in

    def interp2d_pairs(*args, **kwargs):
        """Same interface as interp2d but the returned interpolant evaluates inputs as pairs (x[i], y[i])."""
        lon, lat, var = args[0], args[1], args[2]
        kind = kwargs.get("kind", "linear")

        lon = np.asarray(lon).copy()
        lat = np.asarray(lat).copy()
        var = np.asarray(var)

        if lon.ndim != 1 or lat.ndim != 1 or var.ndim != 2:
            raise ValueError("interp2d_pairs expects lon, lat as 1D arrays and var as a 2D array.")

        # Handle var orientation (accept either (nlat,nlon) or (nlon,nlat))
        if var.shape == (lon.size, lat.size):
            var = var.T  # -> (nlat, nlon)

        # Ensure increasing axes
        if np.any(np.diff(lon) < 0):
            ix = np.argsort(lon)
            lon = lon[ix]
            var = var[:, ix]
        if np.any(np.diff(lat) < 0):
            iy = np.argsort(lat)
            lat = lat[iy]
            var = var[iy, :]

        method = "linear" if kind == "linear" else "nearest"

        rgi = RegularGridInterpolator(
            (lat, lon),
            var,
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )

        def f(x, y):
            x = np.asarray(x)
            y = np.asarray(y)
            pts = np.column_stack([y.ravel(), x.ravel()])  # (lat, lon)
            out = rgi(pts)
            return out.reshape(x.shape)

        return f

    def rk1flatstep(self, t, x, y, f, h):
        xp, yp = f(self, t, x, y)  # with mathematical formalism, this is d(pts)/dt, or pts', that is, ptsp
        x_n = x + xp * h
        y_n = y + yp * h
        return x_n, y_n

    def rk4flatstep(self, t, x, y, f, h, **kwargs):
        k1 = h * np.asarray(f(self, t, x, y, **kwargs))
        k2 = h * np.asarray(f(self, (t + h / 2), (x + k1[0] / 2), (y + k1[1] / 2), **kwargs))
        k3 = h * np.asarray(f(self, (t + h / 2), (x + k2[0] / 2), (y + k2[1] / 2), **kwargs))
        k4 = h * np.asarray(f(self, (t + h), (x + k3[0]), (y + k3[1]), **kwargs))
        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x_n = x + k[0]
        y_n = y + k[1]
        return x_n, y_n

    def rk1flat(self, f, Nstep, **kwargs):
        t_v = self.pt
        x = self.px
        y = self.py
        trjx = []
        trjy = []
        trjx.append(x)
        trjy.append(y)
        t = t_v[0]
        h = (t_v[1] - t_v[0]) / (Nstep * self.numdays)
        xn = x
        yn = y
        for i in range(Nstep * self.numdays):
            t = t + h
            xn, yn = Lagrangian.rk1flatstep(self, t, xn, yn, f, h)
            if "coordinates" in kwargs and kwargs["coordinates"] == "spherical":
                xn, yn = Lagrangian.backonsphere(self, xn, yn)
            trjx.append(xn)
            trjy.append(yn)
        trjf = {
            "lons": self.lons,
            "lats": self.lats,
            "trjx": trjx,
            "trjy": trjy,
            "lonf": trjx[-1:],
            "latf": trjy[-1:],
        }
        return trjf

    def rk4flat(self, f, Nstep, **kwargs):
        t_v = self.pt
        x = self.px
        y = self.py
        trjx, trjy, trjt = ([] for i in range(3))
        #            if (key == 'noise'):
        #                noise=value    #Not yet implemented; see field.h
        if np.size(np.shape(t_v)) == 1:
            h = (t_v[1] - t_v[0]) / (Nstep * self.numdays)
        else:
            h = (t_v[1, 0] - t_v[0, 0]) / (Nstep * self.numdays)
        xn = x
        yn = y
        trjx.append(x)
        trjy.append(y)
        trjt.append(t_v[0])
        t = t_v[0] - h
        for i in range(Nstep * self.numdays):
            t = t + h
            xn, yn = Lagrangian.rk4flatstep(self, t, xn, yn, f, h, **kwargs)
            if "coordinates" in kwargs and kwargs["coordinates"] == "spherical":
                xn, yn = Lagrangian.backonsphere(self, xn, yn)
            trjx.append(xn)
            trjy.append(yn)
            trjt.append(t)

        # deal with small domain to find last valid value for trajectory
        if self.PeriodicBC == False:
            lonf, latf = [], []
            tmptrjx = np.asarray(trjx)
            tmptrjy = np.asarray(trjy)
            for i in range(0, np.shape(tmptrjx)[1]):
                xx = tmptrjx[:, i].tolist()
                yy = tmptrjy[:, i].tolist()
                if np.isnan(xx[1:]).all():
                    lonf.append(xx[0])
                else:
                    while xx and xx[-1] is np.nan:
                        xx.pop()
                    lonf.append(xx[len(xx) - 1])
                if np.isnan(yy[1:]).all():
                    latf.append(yy[0])
                else:
                    while yy and yy[-1] is np.nan:
                        yy.pop()
                    latf.append(yy[len(yy) - 1])
        else:
            lonf = trjx[-1:]
            latf = trjy[-1:]

        trjf = {
            "lons": self.lons,
            "lats": self.lats,
            "trjx": trjx,
            "trjy": trjy,
            "trjt": trjt,
            "lonf": lonf,
            "latf": latf,
        }
        return trjf

    def LLADV(self, trjf, **kwargs):
        """Compute Lon/Lat advections
        :param trj: particle trajectories from advection (returned from 'method')

        :output lladv: lons/lats are longitudes and latitudes for mapping; lonf_map and latf_map are
        longitude and latitude advections respectively formatted for mapping.
        """
        if "Library" in sys.modules.keys():
            Library.tic()

        lonf = trjf["lonf"]
        latf = trjf["latf"]
        lons = trjf["lons"]
        lats = trjf["lats"]
        if "dayv" in kwargs:
            dayv = kwargs["dayv"]
        else:
            print("Missing 'dayv' argument", file=sys.stderr)

        if "PeriodicBC" in kwargs and kwargs["PeriodicBC"] == True:
            if self.loni[1] < self.loni[0]:
                lons0 = lons
                lons0[lons0 < 0] += 360
                lonf[0][lonf[0] < 0] += 360
        else:
            lons0 = lons

        [Xs, Ys] = np.meshgrid(lons0, lats)
        lonf_map = np.reshape(lonf, (np.shape(Xs)[0], np.shape(Xs)[1]))
        latf_map = np.reshape(latf, (np.shape(Xs)[0], np.shape(Xs)[1]))
        lonf_map = Xs - lonf_map
        latf_map = Ys - latf_map

        if "PeriodicBC" in kwargs and kwargs["PeriodicBC"] == True:
            if self.loni[1] < self.loni[0]:
                lonf_map[lonf_map > 180] -= 360

        lladv = {"lons": lons, "lats": lats, "lonf_map": lonf_map, "latf_map": latf_map}

        ### saving ###
        if "output" in kwargs and kwargs["output"] == "netcdf":
            if "GlobalVars" in sys.modules.keys():
                prod = kwargs["product"]
                date = dt.datetime.strftime(dt.datetime.strptime(dayv, "%Y-%m-%d"), "%Y%m%d")
                fname = GlobalVars.Dir["dir_wrk"] + date + "_" + prod + "_LLADV.nc"
                title = prod + "LON/LAT ADVECTION " + date
                Fields.LLADV(fname).createnc(lons, lats, lonf_map, title, vvar2=latf_map)
            else:
                warnings.warn("Warning: Use Save.py to save your data")

        return lladv

    def SSTADV(self, trjf, **kwargs):
        """Compute SST advection from lon/lat advection
        :param lladv: lon/lat advection (returned from 'LLADV')

        :output sstadv: lons/lats are longitudes and latitudes for mapping;
        lonf_map and latf_map are longitude and latitude advections respectively
        formatted for mapping.
        """
        trjx = trjf["trjx"]
        trjy = trjf["trjy"]
        lons = trjf["lons"]
        lats = trjf["lats"]

        if "dayv" in kwargs:
            dayv = kwargs["dayv"]
        else:
            print("Missing 'dayv' argument", file=sys.stderr)

        if "PeriodicBC" in kwargs and kwargs["PeriodicBC"] == True:
            if self.loni[1] < self.loni[0]:
                lons0 = lons
                lons0[lons0 < 0] += 360
        else:
            lons0 = lons

        # compute lon/lat adv at (t0 - n)
        if "daysst" in kwargs:
            day = kwargs["daysst"]
        else:
            if "GlobalVars" in sys.modules.keys():
                day = GlobalVars.Lag["sstadvd"]
            else:
                day = 3
                warnings.warn("Warning: 'daysst' is not defined -> using default value (3)")

        iday = (day * kwargs["numstep"]) + 1
        [Xs, Ys] = np.meshgrid(lons0, lats)

        lonf = np.asarray(trjx)
        latf = np.asarray(trjy)
        lonf, latf = lonf[iday, :], latf[iday, :]

        lonf_map = np.reshape(lonf, (np.shape(Xs)[0], np.shape(Xs)[1]))
        latf_map = np.reshape(latf, (np.shape(Xs)[0], np.shape(Xs)[1]))
        lonf_map = lonf_map.flatten()
        latf_map = latf_map.flatten()

        if "PeriodicBC" in kwargs and kwargs["PeriodicBC"] == True:
            if self.loni[1] < self.loni[0]:
                lonf_map[lonf_map > 180] -= 360

        # load sst map at (t0 - n)
        lon, lat, var = None, None, None
        if "sstfield" in kwargs:
            field = kwargs["sstfield"]
            lon = np.asarray(field["lon"])
            lat = np.asarray(field["lat"])
            var = np.asarray(field["sst"])
        elif "GlobalVars" in sys.modules.keys():
            nprod = GlobalVars.config.get("products", GlobalVars.Lag["sstprod"] + "prod")
            fname = glob.glob(GlobalVars.Dir["dir_wrk"] + "/*" + nprod + "*.nc")
            if fname:
                field = eval("Fields." + nprod + "(fname[0]).loadnc()")
                lon = np.asarray(field["lon"])
                lat = np.asarray(field["lat"])
                var = np.asarray(field["var"])
            else:
                warntxt = "Warning: No SST file, sst advection is empty."
                warnings.warn(warntxt)
                if "Library" in sys.modules.keys():
                    Library.Logfile(warntxt)
        else:
            warnings.warn("Missing SST field.")

        # Compute SST advection
        if (var is not None) and np.size(var) > 0 and np.any(np.isfinite(var)):
            # Wrap advected longitudes to match SST grid convention (0..360)
            if np.nanmin(lon) >= 0 and np.nanmax(lon) > 180:
                lonf_map = (lonf_map + 360) % 360

            # ---- normalise lon/lat/var for interp2d_pairs (expects lon,lat 1D; var 2D) ----
            lon = np.asarray(lon)
            lat = np.asarray(lat)
            var = np.asarray(var)

            # If lon/lat are 2D meshgrids, reduce to 1D axes
            if lon.ndim == 2 and lat.ndim == 2:
                lon = lon[0, :]
                lat = lat[:, 0]

            # Drop singleton dims (common: var is (1, nlat, nlon))
            var = np.squeeze(var)

            # If still 3D (e.g. multiple times), take first time slice
            if var.ndim == 3:
                var = var[0, :, :]

            # Create the interpolant (pairs of points)
            f = Lagrangian.interp2d_pairs(lon, lat, var, kind="linear")

            # Evaluate the interpolant on each pairs of x and y values
            sst = f(lonf_map, latf_map)
            sst_map = np.reshape(sst, (np.shape(Xs)[0], np.shape(Xs)[1]))
        else:
            sst_map = np.zeros((np.shape(Xs)[0], np.shape(Xs)[1]))

        sstadv = {"lons": lons, "lats": lats, "sstadv": sst_map}

        ### saving ###
        if "output" in kwargs and kwargs["output"] == "netcdf":
            if "GlobalVars" in sys.modules.keys():
                prod = kwargs["product"]
                date = dt.datetime.strftime(dt.datetime.strptime(dayv, "%Y-%m-%d"), "%Y%m%d")
                fname = GlobalVars.Dir["dir_wrk"] + date + "_" + prod + "_SSTADV.nc"
                title = prod + "SST ADVECTION " + date
                Fields.SSTADV(fname).createnc(lons, lats, sst_map, title)
            else:
                warnings.warn("Warning: Use Save.py to save your data")

        return sstadv

    def FTLE(self, trjf, **kwargs):
        lons = trjf["lons"]
        lats = trjf["lats"]
        lonf = trjf["lonf"]
        latf = trjf["latf"]
        if "numdays" in kwargs:
            numdays = int(kwargs["numdays"])
        else:
            print("Missing 'numdays' argument to compute FTLE", file=sys.stderr)
        if "dayv" in kwargs:
            dayv = kwargs["dayv"]
        else:
            print("Missing 'dayv' argument", file=sys.stderr)

        if "PeriodicBC" in kwargs and kwargs["PeriodicBC"] == True:
            if self.loni[1] < self.loni[0]:
                lons0 = lons
                lons0[lons0 < 0] += 360
                lonf[0][lonf[0] < 0] += 360
            else:
                lons0 = lons
        else:
            lons0 = lons

        [Xs, Ys] = np.meshgrid(lons0, lats)
        lon0_map = Xs
        lat0_map = Ys
        lonf_map = np.reshape(lonf, (np.shape(Xs)[0], np.shape(Xs)[1]))
        latf_map = np.reshape(latf, (np.shape(Xs)[0], np.shape(Xs)[1]))
        ### Gradients of final positions ###
        [d_lonyf, d_lonxf] = np.gradient(lonf_map)
        [d_latyf, d_latxf] = np.gradient(latf_map)
        d_lonxf = d_lonxf * np.cos(Ys / 180 * np.pi)
        ### Gradients of initial positions ###
        [d_lony0, d_lonx0] = np.gradient(lon0_map)
        [d_laty0, d_latx0] = np.gradient(lat0_map)
        d_lonx0 = d_lonx0 * np.cos(Ys / 180 * np.pi)
        ### Final separation ###
        Xgradf = (d_lonxf**2) + (d_latxf**2)
        Ygradf = (d_lonyf**2) + (d_latyf**2)
        XYgradf = []
        XYgradf.append(Xgradf)
        XYgradf.append(Ygradf)
        final_separation = np.max(XYgradf, 0)
        ### Initial separation ###
        initial_separation = (d_lonx0**2) + (d_latx0**2)
        ### FTLE ###
        ftle_lyap = (np.log(final_separation / initial_separation) / (numdays)) / 2
        ftle = {"lons": lons, "lats": lats, "ftle": ftle_lyap}

        ### saving ###
        if "output" in kwargs and kwargs["output"] == "netcdf":
            if "GlobalVars" in sys.modules.keys():
                prod = kwargs["product"]
                date = dt.datetime.strftime(dt.datetime.strptime(dayv, "%Y-%m-%d"), "%Y%m%d")
                fname = GlobalVars.Dir["dir_wrk"] + date + "_" + prod + "_FTLE.nc"
                title = prod + "FTLE " + date
                Fields.FTLE(fname).createnc(lons, lats, ftle_lyap, title)
            else:
                warnings.warn("Warning: Use Save.py to save your data")
        return ftle

    def FTLE_drift(self, trjf, **kwargs):
        """
        Function to predict FTLE drift (advection)

        Parameters
        ----------
        trjf : dictionnary including particle trajectories
        **kwargs :
            dayd : number of days for drifted prediction

        Returns
        -------
        FTLE matrix with original and drifted coordinates

        """
        trjx = trjf["trjx"]
        trjy = trjf["trjy"]
        trjt = trjf["trjt"]
        lons = trjf["lons"]
        lats = trjf["lats"]
        lonf = trjf["lonf"]
        latf = trjf["latf"]

        if "PeriodicBC" in kwargs and kwargs["PeriodicBC"] == True:
            if self.loni[1] < self.loni[0]:
                lons0 = lons
                lons0[lons0 < 0] += 360
        else:
            lons0 = lons

        # compute lon/lat adv at (t0 - n)
        if "dayd" in kwargs:
            day = kwargs["dayd"]
        else:
            if "GlobalVars" in sys.modules.keys():
                day = GlobalVars.Lag["dayd"]
            else:
                day = 2
                warnings.warn("Warning: 'dayd' is not defined -> using default value (2 days)")
        iday = (kwargs["numstep"]) + 1
        print(trjf["trjt"][iday])
        [Xs, Ys] = np.meshgrid(lons0, lats)
        # lonf = np.asarray(trjx)
        # latf = np.asarray(trjy)
        # lonf,latf = lonf[iday,:],latf[iday,:]
        lonf_map = np.reshape(lonf, (np.shape(Xs)[0], np.shape(Xs)[1]))
        latf_map = np.reshape(latf, (np.shape(Xs)[0], np.shape(Xs)[1]))

        lona = (Xs - lonf_map) * np.cos(Ys / 180 * np.pi)
        lata = Ys - latf_map

        # get u,v
        U = self.u_nonan[-1, :, :]
        V = self.v_nonan[-1, :, :]

        lonuv = self.lon
        latuv = self.lat
        u2 = interpn((lonuv, latuv), U, (Xs, Ys), bounds_error=False, fill_value=np.nan)
        v2 = interpn((lonuv, latuv), V, (Xs, Ys), bounds_error=False, fill_value=np.nan)

        [d_lony, d_lonx] = np.gradient(Xs - lonf_map)
        [d_laty, d_latx] = np.gradient(Ys - latf_map)
        d_lonx = d_lonx * np.cos(Ys / 180 * np.pi)
        theta = np.arctan(d_laty, d_lonx)
        coef = (u2 * np.cos(theta + np.pi / 2)) + (v2 * np.sin(theta + np.pi / 2))
        ud = coef * np.cos(theta + np.pi / 2)
        vd = coef * np.sin(theta + np.pi / 2)
        lonsd = Xs + (ud * day)
        latsd = Ys + (vd * day)

        # theta = np.arctan2((Ys-latf_map),(Xs-lonf_map))
        # coef = ((Xs-lonf_map)*np.cos(theta)) + ((Ys-latf_map)*np.sin(theta))
        # lonsd = Xs + coef * np.cos(theta)
        # latsd = Ys + coef * np.sin(theta)

        # lonsd = Xs + (Xs - lonf_map)
        # latsd = Ys + (Ys - latf_map)

        if "ftlef" in kwargs:
            ftle = kwargs["ftlef"]
        else:  # compute FTLE
            ftle = self.FTLE(trjf, **kwargs)

        ftle = {
            "lons": lons,
            "lats": lats,
            "ftle": ftle["ftle"],
            "lonsd": lonsd,
            "latsd": latsd,
            "u": u2,
            "v": v2,
            "ud": ud,
            "vd": vd,
            "lona": lona,
            "lata": lata,
        }

        return ftle

    def OWTRAJ(self, trjf, **kwargs):
        if "Library" in sys.modules.keys():
            Library.tic()

        t = np.ravel(trjf["trjt"])
        tv = np.array(trjf["trjt"])[:, 0]
        x = np.ravel(trjf["trjx"])
        y = np.ravel(trjf["trjy"])
        sz = np.shape(trjf["trjx"])
        lons = trjf["lons"]
        lats = trjf["lats"]

        if "dayv" in kwargs:
            dayv = kwargs["dayv"]
        else:
            print(
                "Missing 'dayv' argument (format: %Y-%m-%d) to compute OWTRAJ",
                file=sys.stderr,
            )

        if "ds" in kwargs:
            ds = kwargs["ds"]
        else:
            print("Missing 'ds' argument to compute OWTRAJ", file=sys.stderr)

        if "PeriodicBC" in kwargs and kwargs["PeriodicBC"] == True:
            if self.loni[1] < self.loni[0]:
                lons0 = lons
                lons0[lons0 < 0] += 360
                x[x < 0] += 360
        else:
            lons0 = lons

        [Xs, Ys] = np.meshgrid(lons0, lats)
        RT = 6371e5
        dUdx, dUdy, dVdx, dVdy, U, V = ([] for i in range(6))

        # convert t in days since beginning of integration
        dayv0 = dt.datetime.toordinal(dt.datetime.strptime(dayv, "%Y-%m-%d").date())
        tv -= dayv0
        convx = np.pi / 180.0 * (RT * np.cos(y / 180 * np.pi))
        convy = 1 / 180 * np.pi * RT
        velx = Lagrangian.interpf(self, t, x + ds, y, **kwargs)
        vely = Lagrangian.interpf(self, t, x, y + ds, **kwargs)
        velmx = Lagrangian.interpf(self, t, x - ds, y, **kwargs)
        velmy = Lagrangian.interpf(self, t, x, y - ds, **kwargs)
        dsx = 2 * ds * convx
        dsy = 2 * ds * convy
        dUdx = (velx[0] - velmx[0]) / dsx
        dUdy = (vely[0] - velmy[0]) / dsy
        dVdx = (velx[1] - velmx[1]) / dsx
        dVdy = (vely[1] - velmy[1]) / dsy

        sn = dUdx - dVdy
        ss = dUdy + dVdx
        vor = -dUdy + dVdx
        ow = (sn**2) + (ss**2) - (vor**2)
        owm = np.reshape(ow, sz)

        posexit = []
        for ct in range(sz[1]):
            tmp = np.sign(owm[:, ct])
            maxv = np.nanmax(tmp)
            exit1 = np.argmax(tmp)
            exit = tv[exit1]
            if maxv < 0:
                exit = tv[-1]
            posexit.append(exit)

        owd = np.reshape(posexit, (np.shape(Xs)[0], np.shape(Xs)[1]))
        owdisp = {"lons": lons, "lats": lats, "owdisp": owd}

        ### saving ###
        if "output" in kwargs and kwargs["output"] == "netcdf":
            if "GlobalVars" in sys.modules.keys():
                prod = kwargs["product"]
                date = dt.datetime.strftime(dt.datetime.strptime(dayv, "%Y-%m-%d"), "%Y%m%d")
                fname = GlobalVars.Dir["dir_wrk"] + date + "_" + prod + "_OWTRAJ.nc"
                title = prod + "OWTRAJ " + date
                Fields.OWTRAJ(fname).createnc(lons, lats, owd, title)
            else:
                warnings.warn("Warning: Use Save.py to save your data")

        if "Library" in sys.modules.keys():
            Library.toc("OWTRAJ")
        return owdisp

    def TIMEFROMBATHY(self, trjf, **kwargs):
        lons = trjf["lons"]
        lats = trjf["lats"]
        trjx = np.asarray(trjf["trjx"])
        trjy = np.asarray(trjf["trjy"])

        if "dayv" in kwargs:
            dayv = kwargs["dayv"]
        else:
            print(
                "Missing 'dayv' argument (format: %Y-%m-%d) to compute OWTRAJ",
                file=sys.stderr,
            )

        if "PeriodicBC" in kwargs and kwargs["PeriodicBC"] == True:
            if self.loni[1] < self.loni[0]:
                trjx = trjx.copy()
                trjx[trjx < 0] += 360

        # extract grid from bathy file in subdomain
        if "bathyfield" in kwargs:
            field = kwargs["bathyfield"]
            lon = np.asarray(field["lon"])
            lat = np.asarray(field["lat"])
            z = np.asarray(field["z"])
            trjx = trjx[1 :: kwargs["numstep"]]
            trjy = trjy[1 :: kwargs["numstep"]]
            bathylvl = kwargs["bathylvl"]
        elif "GlobalVars" in sys.modules.keys():
            trjx = trjx[1 :: GlobalVars.Lag["numstep"]]
            trjy = trjy[1 :: GlobalVars.Lag["numstep"]]
            bathylvl = GlobalVars.Lag["bathylvl"]
            file = GlobalVars.Dir["dir_bathy"] + GlobalVars.Lag["bathyfile"]
            field = Fields.ETOPO.loadnc(file, rlon=GlobalVars.Lag["loni"], rlat=GlobalVars.Lag["lati"])
            lon = np.asarray(field["lon"])
            lat = np.asarray(field["lat"])
            z = np.asarray(field["z"])
        else:
            print("Missing bathymetry field: bathyfield = field", file=sys.stderr)
            lon, lat, z = None, None, None

        if (lon is not None) and (np.size(lon) > 0) and (np.nanmin(lon) < 0) and (np.nanmin(trjx) > 0):
            lon = lon.copy()
            lon[lon < 0] += 360

        # Compute bathy along trajectories
        if (z is not None) and (np.size(z) > 0) and np.any(np.isfinite(z)):
            # Wrap traj longitudes to match bathy lon convention (0..360)
            if np.nanmin(lon) >= 0 and np.nanmax(lon) > 180:
                trjx = (trjx + 360) % 360

            f = Lagrangian.interp2d_pairs(lon, lat, z, kind="linear")
            trjd = f(trjx, trjy)
        else:
            trjd = np.full_like(trjx, np.nan, dtype=float)

        touched = []
        nottouched = []
        for ct in range(0, np.shape(trjx)[1]):
            touch = [i for i, v in enumerate(trjd[:, ct]) if v > bathylvl]
            if not touch:
                touch = 0  # not touched
                nottouched.append(ct)
            else:
                touch = min(touch)
            touched.append(touch)

        touchedlat = []
        touchedlon = []
        for ct in range(0, len(touched)):
            touchedlat.append(trjy[touched[ct], ct])
            touchedlon.append(trjx[touched[ct], ct])

        touched = np.asarray(touched, dtype=float)
        touchedlat = np.asarray(touchedlat, dtype=float)
        touchedlon = np.asarray(touchedlon, dtype=float)

        touched[nottouched] = np.nan
        touchedlat[nottouched] = np.nan
        touchedlon[nottouched] = np.nan

        touched = touched.reshape((len(lats), len(lons)))
        touchedlat = touchedlat.reshape((len(lats), len(lons)))
        touchedlon = touchedlon.reshape((len(lats), len(lons)))

        timfbathy = {
            "lons": lons,
            "lats": lats,
            "timfb": touched,
            "latfb": touchedlat,
            "lonfb": touchedlon,
        }
        return timfbathy


class ParticleSet(Lagrangian):
    def __init__(
        self,
        pt=None,
        px=None,
        py=None,
        lons=None,
        lats=None,
        numdays=None,
        loni=None,
        lati=None,
        delta0=None,
        dayv=None,
        **kwargs,
    ):
        self.pt = pt
        self.px = px
        self.py = py
        self.lons = lons
        self.lats = lats
        self.loni = loni
        self.lati = lati
        if "fieldset" in kwargs:
            ff = kwargs.get("fieldset")
            self.lon = ff["lon"]
            self.lat = ff["lat"]
            self.u = ff["u"]
            self.v = ff["v"]
            self.dates = ff["dates"]
            self.u_nonan = np.where(np.isnan(self.u), 0, self.u)
            self.v_nonan = np.where(np.isnan(self.v), 0, self.v)
            ParticleSet.check_dimensions(self, **kwargs)
            if "us" in ff:
                self.lons = ff["lons"]
                self.lats = ff["lats"]
                self.us = ff["us"]
                self.vs = ff["us"]

        if "PeriodicBC" in kwargs:
            self.PeriodicBC = kwargs["PeriodicBC"]
            if self.PeriodicBC == True:
                dlon = np.array(np.diff(self.lon))
                self.lon = np.hstack([self.lon[0] - dlon[0] / 2, self.lon, self.lon[-1] + dlon[-1] / 2])
                if self.u.ndim == 2 and self.u.ndim == 2:
                    self.u = np.vstack([self.u[0, :], self.u, self.u[-1, :]])
                    self.v = np.vstack([self.v[0, :], self.v, self.v[-1, :]])
                    self.u_nonan = np.vstack([self.u_nonan[0, :], self.u_nonan, self.u_nonan[-1, :]])
                    self.v_nonan = np.vstack([self.v_nonan[0, :], self.v_nonan, self.v_nonan[-1, :]])
                elif self.u.ndim == 3 and self.v.ndim == 3:
                    self.u = [np.vstack([self.u[i, 0, :], self.u[i, :, :], self.u[i, -1, :]]) for i in range(self.u.shape[0])]
                    self.v = [np.vstack([self.v[i, 0, :], self.v[i, :, :], self.v[i, -1, :]]) for i in range(self.v.shape[0])]
                    self.u_nonan = [
                        np.vstack(
                            [
                                self.u_nonan[i, 0, :],
                                self.u_nonan[i, :, :],
                                self.u_nonan[i, -1, :],
                            ]
                        )
                        for i in range(self.u_nonan.shape[0])
                    ]
                    self.v_nonan = [
                        np.vstack(
                            [
                                self.v_nonan[i, 0, :],
                                self.v_nonan[i, :, :],
                                self.v_nonan[i, -1, :],
                            ]
                        )
                        for i in range(self.v_nonan.shape[0])
                    ]
        else:
            self.PeriodicBC = []

        if numdays is None:
            self.numdays = 1
        else:
            self.numdays = numdays

        return

    @classmethod
    def from_input(cls, pt, px, py, **kwargs):
        px = px
        py = py
        pt = pt
        lons, lats = px, py
        return cls(pt, px, py, lons, lats, **kwargs)

    @classmethod
    def from_grid(cls, numdays, loni, lati, delta0, dayv, **kwargs):
        if "PeriodicBC" in kwargs and kwargs["PeriodicBC"] == True:
            if loni[0] > loni[1]:
                bnd = ParticleSet.get_boundary(**kwargs)
                xi = loni[1] + (bnd[1] - bnd[0])
                px = np.arange(loni[0], xi, delta0)
                py = np.arange(lati[0], lati[1], delta0)
            else:
                px = np.arange(loni[0], loni[1], delta0)
                py = np.arange(lati[0], lati[1], delta0)
        elif "PeriodicBC" in kwargs and kwargs["PeriodicBC"] == False:
            if ((loni[0]) == (loni[1])) & ((lati[0]) != (lati[1])):
                px = loni[0]
                py = np.arange(lati[0], lati[1], delta0)
            elif ((loni[0]) != (loni[1])) & ((lati[0]) == (lati[1])):
                py = lati[0]
                px = np.arange(loni[0], loni[1], delta0)
            elif ((loni[0]) == (loni[1])) & ((lati[0]) == (lati[1])):
                px = loni[0]
                py = lati[0]
            else:
                px = np.arange(loni[0], loni[1], delta0)
                py = np.arange(lati[0], lati[1], delta0)
        else:
            if ((loni[0]) == (loni[1])) & ((lati[0]) != (lati[1])):
                px = loni[0]
                py = np.arange(lati[0], lati[1], delta0)
            elif ((loni[0]) != (loni[1])) & ((lati[0]) == (lati[1])):
                py = lati[0]
                px = np.arange(loni[0], loni[1], delta0)
            elif ((loni[0]) == (loni[1])) & ((lati[0]) == (lati[1])):
                px = loni[0]
                py = lati[0]
            else:
                px = np.arange(loni[0], loni[1], delta0)
                py = np.arange(lati[0], lati[1], delta0)

        lons, lats = px, py
        [X, Y] = np.meshgrid(px, py)
        px = np.ravel(X)
        py = np.ravel(Y)
        if dayv == None:
            pt0 = np.array([numdays] * len(px))
            ptf = np.array([0] * len(px))
        else:
            day2 = dt.datetime.strptime(dayv, "%Y-%m-%d").date()
            day2j = dt.datetime.toordinal(day2)
            if "mode" in kwargs:
                if kwargs["mode"] == "backward":
                    day1j = day2j - numdays
                elif kwargs["mode"] == "forward":
                    day1j = day2j + numdays
            else:
                # default is backward
                day1j = day2j - numdays
            pt0 = np.array([day2j] * len(px))
            ptf = np.array([day1j] * len(px))
        pt = np.array([pt0, ptf])
        return cls(pt, px, py, lons, lats, numdays, loni, lati, **kwargs)

    @classmethod
    def from_disk(cls, numdays, pxc, pyc, rad, dayv, sample, **kwargs):
        """Seed particle on an horizontal disk.
        -------
        Inputs:
            pxc = x center position
            pyc = y center position
            rad = maximum radius of disk (in degree of lon/lat)
            numdays = number of days for seeding
            dayv = starting date (%Y-%m-%d) for seeding
            method = seeding method inside disk:
                circles = parcels are sed along multiple circles inside disk
                random = parcels are randomly sed inside disk
        ----------
        Arguments:
            npoints = number of parcels to be sed (default=1000)
        """
        if sample == "circles":
            if "deltar" in kwargs:
                dr = kwargs["deltar"]
            else:
                dr = 10
                warnings.warn("Default number of circles = 10.")
            rads = np.linspace(0.1, rad, dr)  # start with 0.1 otherwise 0 is on the center point
            x, y = [], []
            for rr in rads:
                if "npoints" in kwargs:
                    n = kwargs["npoints"]
                else:
                    n = 100
                    warnings.warn("Default number of particles on circle = 100.")
                theta = np.linspace(0, 2 * np.pi, n)
                xx = pxc + (rr * np.cos(theta))
                yy = pyc + (rr * np.sin(theta))
                x.append(xx)
                y.append(yy)
            x, y = np.array(x).flatten(), np.array(y).flatten()

        elif sample == "random":
            if "npoints" in kwargs:
                n = kwargs["npoints"]
            else:
                n = 1000
                warnings.warn("Default number of points = 1000.")
            r = np.random.uniform(low=0, high=1, size=n)  # random radius
            theta = np.random.uniform(low=0, high=2 * np.pi, size=n)  # random angle
            x = pxc + ((r * rad) * np.cos(theta))
            y = pyc + ((r * rad) * np.sin(theta))
        else:
            print(
                "Unvalid method for seeding from_disk (circles or random)",
                file=sys.stderr,
            )

        px = x
        py = y
        lons, lats = px, py
        # seed for multiple days
        if dayv == None:
            pt0 = np.array([numdays] * len(px))
            ptf = np.array([0] * len(px))
        else:
            day2 = dt.datetime.strptime(dayv, "%Y-%m-%d").date()
            day2j = dt.datetime.toordinal(day2)
            if "mode" in kwargs:
                if kwargs["mode"] == "backward":
                    day1j = day2j - numdays
                elif kwargs["mode"] == "forward":
                    day1j = day2j + numdays
            else:
                # default is backward
                day1j = day2j - numdays
            pt0 = np.array([day2j] * len(px))
            ptf = np.array([day1j] * len(px))
        pt = np.array([pt0, ptf])
        return cls(pt, px, py, lons, lats, **kwargs)

    def get_boundary(**kwargs):
        try:
            if "xmin" and "xmax" in kwargs:
                xmin, xmax = kwargs["xmin"], kwargs["xmax"]
            elif "ymin" and "ymax" in kwargs:
                ymin, ymax = kwargs["ymin"], kwargs["ymax"]
            elif "fieldset" in kwargs:
                ff = kwargs.get("fieldset")
                xmin, xmax = ff["lon"][0], ff["lon"][-1]
                ymin, ymax = ff["lat"][0], ff["lat"][-1]
        except Exception as e:
            print(e)

        return [xmin, xmax, ymin, ymax]

    def check_dimensions(self, **kwargs):
        # lon,lat,dates should be one dimensional
        if self.lon.ndim > 1:
            if self.lon.ndim == 2 and self.lon.ndim == self.lat.ndim:
                self.lon, self.lat = self.lon[:, 0], self.lat[0, :]
                if np.all(np.diff(self.lon)) == 0 and np.all(np.diff(self.lat)) == 0:
                    self.lon, self.lat = self.lon[0, :], self.lat[:, 0]
            elif self.lon.ndim == 3:
                self.lon, self.lat = self.lon[0, :, 0], self.lat[0, 0, :]
                if np.all(np.diff(self.lon)) == 0 and np.all(np.diff(self.lat)) == 0:
                    self.lon, self.lat = self.lon[:, 0, 0], self.lat[0, :, 0]
                    if np.all(np.diff(self.lon)) == 0 and np.all(np.diff(self.lat)) == 0:
                        self.lon, self.lat = self.lon[0, 0, :], self.lat[:, 0, 0]
        # check if longitudes are in -180;180 grid
        if "xy" in kwargs and kwargs["xy"] == "xy":
            warnings.warn("Warning: x and y data are not lon/lat")
        else:
            if np.min(self.lon) == 0.0 and np.max(self.lon) > 181:
                self.lon[self.lon > 180] = self.lon[self.lon > 180] - 360
                sort = np.argsort(self.lon)
                self.lon = self.lon[sort]
                if self.u.ndim == 2:
                    self.u = self.u[sort, :]
                    self.v = self.v[sort, :]
                    self.u_nonan = self.u_nonan[sort, :]
                    self.v_nonan = self.v_nonan[sort, :]
                elif self.u.ndim == 3:
                    self.u = self.u[:, sort, :]
                    self.v = self.v[:, sort, :]
                    self.u_nonan = self.u_nonan[:, sort, :]
                    self.v_nonan = self.v_nonan[:, sort, :]
        return


class Eulerian:
    def __init__(self, fieldset=None, dayv=None):
        self.RT = 6371e5
        if fieldset != None:
            self.lon = fieldset["lon"]
            self.lat = fieldset["lat"]
            self.u = fieldset["u"]
            self.v = fieldset["v"]
            if isinstance(self.u, np.ma.MaskedArray):
                self.u = self.u.filled(np.nan)
                self.v = self.v.filled(np.nan)
            if "dates" in fieldset:
                self.dates = fieldset["dates"]
        else:
            print(
                "Missing 'fieldset' argument. Cannot compute Eulerian diags.",
                file=sys.stderr,
            )

        if dayv != None:
            self.dayv = dayv
        else:
            self.dayv = self.dates[-1]
            print(
                "Missing 'dayv' argument to compute Eulerian diag. Default value is used (i.e. last date of field)",
                file=sys.stderr,
            )
        return

    @staticmethod
    def _interp2d_rgi(lon0, lat0, field2d, lon_new, lat_new, method="linear"):
        lon0 = np.asarray(lon0).copy()
        lat0 = np.asarray(lat0).copy()
        field2d = np.asarray(field2d)

        # Accept either (nlat, nlon) or (nlon, nlat)
        if field2d.shape == (lon0.size, lat0.size):
            field2d = field2d.T  # -> (nlat, nlon)

        # Ensure increasing axes
        if np.any(np.diff(lon0) < 0):
            ix = np.argsort(lon0)
            lon0 = lon0[ix]
            field2d = field2d[:, ix]
        if np.any(np.diff(lat0) < 0):
            iy = np.argsort(lat0)
            lat0 = lat0[iy]
            field2d = field2d[iy, :]

        rgi = RegularGridInterpolator((lat0, lon0), field2d, method=method, bounds_error=False, fill_value=np.nan)

        Lon, Lat = np.meshgrid(lon_new, lat_new)  # (nlat_new, nlon_new)
        pts = np.column_stack([Lat.ravel(), Lon.ravel()])
        out = rgi(pts).reshape(Lat.shape)
        return out

    def diag(self, diag=None, **kwargs):
        out = []
        if diag != None:
            for i in diag:
                if i == "KE":
                    dd = self.KE(**kwargs)
                if i == "OW":
                    dd = self.OW(**kwargs)
                out.append(dd)
        return out

    def KE(self, **kwargs):
        if self.u.ndim == 2:
            U = self.u
            V = self.v
        elif self.u.ndim == 3:
            day = dt.datetime.toordinal(dt.datetime.strptime(self.dayv, "%Y-%m-%d").date())
            idd = np.where(self.dates == day)
            U = np.squeeze(self.u[idd, :, :])
            V = np.squeeze(self.v[idd, :, :])

        if "UVunit" in kwargs:
            if kwargs["UVunit"] == "m/s":
                Ucms, Vcms = U * 1e2, V * 1e2
            if kwargs["UVunit"] == "cm/s":
                Ucms, Vcms = U, V

        if ("delta" in kwargs) and ("lon" in kwargs) and ("lat" in kwargs):
            delta0 = kwargs["delta"]
            loni = kwargs["lon"]
            lati = kwargs["lat"]
            lon = np.arange(loni[0], loni[1], delta0 / 2)
            lat = np.arange(lati[0], lati[1], delta0 / 2)
            [X, Y] = np.meshgrid(lon, lat)

            lon0 = np.array(self.lon)
            lat0 = np.array(self.lat)
            u_nonan = np.where(np.isnan(Ucms), 0, Ucms)
            v_nonan = np.where(np.isnan(Vcms), 0, Vcms)

            Ucms = self._interp2d_rgi(lon0, lat0, u_nonan, lon, lat, method="linear")
            Vcms = self._interp2d_rgi(lon0, lat0, v_nonan, lon, lat, method="linear")
        else:
            print(
                "Missing 'delta', 'lon' and 'lat' arguments to compute Eulerian diag.",
                file=sys.stderr,
            )

        E = (Ucms**2) + (Vcms**2)
        KE = {"lon": X, "lat": Y, "KE": E}
        return KE

    def OW(self, **kwargs):
        if self.u.ndim == 2:
            U = self.u
            V = self.v
        elif self.u.ndim == 3:
            day = dt.datetime.toordinal(dt.datetime.strptime(self.dayv, "%Y-%m-%d").date())
            idd = np.where(self.dates == day)
            U = np.squeeze(self.u[idd, :, :])
            V = np.squeeze(self.v[idd, :, :])

        [X, Y] = np.meshgrid(self.lon, self.lat)
        if np.shape(X) != np.shape(U):
            [Y, X] = np.meshgrid(self.lat, self.lon)

        if "UVunit" in kwargs:
            if kwargs["UVunit"] == "m/s":
                Ucms, Vcms = U * 1e2, V * 1e2
            if kwargs["UVunit"] == "cm/s":
                Ucms, Vcms = U, V

        if ("delta" in kwargs) and ("lon" in kwargs) and ("lat" in kwargs):
            delta0 = kwargs["delta"]
            loni = kwargs["lon"]
            lati = kwargs["lat"]
            lon = np.arange(loni[0], loni[1], delta0 / 2)
            lat = np.arange(lati[0], lati[1], delta0 / 2)
            [X, Y] = np.meshgrid(lon, lat)

            lon0 = np.array(self.lon)
            lat0 = np.array(self.lat)
            u_nonan = np.where(np.isnan(Ucms), 0, Ucms)
            v_nonan = np.where(np.isnan(Vcms), 0, Vcms)

            Ucms = self._interp2d_rgi(lon0, lat0, u_nonan, lon, lat, method="linear")
            Vcms = self._interp2d_rgi(lon0, lat0, v_nonan, lon, lat, method="linear")
        else:
            print(
                "Missing 'delta', 'lon' and 'lat' arguments to compute Eulerian diag.",
                file=sys.stderr,
            )

        # ... keep the rest of OW exactly as you have it ...
        dUdy, dUdx = np.gradient(Ucms)
        dVdy, dVdx = np.gradient(Vcms)

        tmp, Dx = np.gradient(X / 180 * np.pi * self.RT * np.cos(Y * np.pi / 180))
        Dy, tmp = np.gradient(Y / 180 * np.pi * self.RT)

        dUdx = dUdx / Dx
        dVdx = dVdx / Dx
        dUdy = dUdy / Dy
        dVdy = dVdy / Dy

        sn = dUdx - dVdy
        ss = dUdy + dVdx
        vor = -dUdy + dVdx
        ow = sn**2 + ss**2 - vor**2
        ow = ow * (60 * 60 * 24) ** 2  # daily

        OW = {"lon": X, "lat": Y, "sn": sn, "ss": ss, "vor": vor, "ow": ow}
        return OW


# Spasso spec
# def Launch(cruise,approach):
#    opt = [str(x) for x in GlobalVars.config.get('plot_options','options').split(',')]
#    if approach == 'eulerian':
#        for pr in GlobalVars.Eul['products']:
#            if pr:
#                nprod = GlobalVars.config.get('products',pr+'prod')
#                var = Library.GetVars(GlobalVars.config.get('products',pr+'_data'))
#                if GlobalVars.Eul['dayv']=='default':
#                    dayv = var['datef']
#                    tmp = var['date']
#                else:
#                    dayv = [str(x) for x in GlobalVars.Eul['dayv'].split(',')]
#                    tmp = []
#                    for nf in dayv:
#                        tmp.append(var['date'][var['datef'].index(nf)])
#                for nf in range(len(dayv)):
#                    Library.printMessage("Computing Eulerian for "+nprod+" "+dayv[nf])
#                    path = GlobalVars.Dir['dir_wrk']+'/'+tmp[nf]+'*'+nprod+'*.nc'
#                    exf,ff = Library.ExistingFile(path,tmp[nf])
#                    if exf == True:
#                        fname = glob.glob(path)[0]
#                    elif exf == False:
#                        #download data
#                        eval("Fields."+nprod+".download(date=tmp[nf],cp='yes')")
#                        fname = glob.glob(path)[0]
#                    field = eval('Fields.'+nprod+'(fname).loadnc()')
#                    out = Eulerian(field,dayv[nf]).diag(diag=GlobalVars.Eul['diag'],
#                                                              UVunit=GlobalVars.Eul['UVunit'],delta=GlobalVars.Eul['delta0'],
#                                                              lon=GlobalVars.Eul['loni'],lat=GlobalVars.Eul['lati'],
#                                                              output='netcdf',product=nprod)
#                    #plot field
#                    if out:
#                        for di in range(0,len(out)):
#                            Library.printMessage("Ploting "+GlobalVars.Eul['diag'][di]+" for "+nprod)
#                            PlotField.PlotField.Plot(cruise,GlobalVars.Eul['diag'][di],opt,type='Eulerian')
#                    else:
#                        Library.Done('None')
#    elif approach == 'lagrangian':
#        for pr in GlobalVars.Lag['products']:
#            if pr:
#                nprod = GlobalVars.config.get('products',pr+'prod')
#                var = Library.GetVars(GlobalVars.config.get('products',pr+'_data'))
#                if GlobalVars.Lag['dayv']=='default':
#                    dayv = var['datef']
#                    tmp = var['date']
#                else:
#                    dayv = [str(x) for x in GlobalVars.Lag['dayv'].split(',')]
#                    tmp = []
#                    for nf in dayv:
#                        tmp.append(var['date'][var['datef'].index(nf)])
#                for nf in range(len(dayv)):
#                    Library.printMessage("Computing lagrangian for "+nprod+" "+dayv[nf])
#
#                    path = GlobalVars.Dir['dir_wrk']+'/'+tmp[nf]+'*'+nprod+'*.nc'
#                    exf,ff = Library.ExistingFile(path,tmp[nf])
#                    if exf == True:
#                        fname = glob.glob(path)[0]
#                    elif exf == False:
#                        #download data
#                        eval("Fields."+nprod+".download(date=tmp[nf],cp='yes')")
#                        fname = glob.glob(path)[0]
#                    field = eval('Fields.'+nprod+'(fname,dayv=dayv[nf]).LoadLag(GlobalVars.Lag["numdays"],product=nprod)')
#                    pset = ParticleSet.from_grid(GlobalVars.Lag['numdays'],GlobalVars.Lag['loni'],
#                                              GlobalVars.Lag['lati'],GlobalVars.Lag['delta0'],
#                                              dayv[nf],fieldset=field,mode=GlobalVars.Lag['mode'],
#                                              PeriodicBC=GlobalVars.Lag['PeriodicBC'])
#                    out = pset.diag(diag=GlobalVars.Lag['diag'],method=GlobalVars.Lag['method'],
#                                    f=Lagrangian.interpf,numstep=GlobalVars.Lag['numstep'],
#                                    coordinates='spherical',numdays=GlobalVars.Lag['numdays'],
#                                    dayv=dayv[nf],ds=1/6,output='netcdf',product=nprod,PeriodicBC=GlobalVars.Lag['PeriodicBC'])
#
#                    #plot field
#                    if out:
#                        for di in range(0,len(out)-1):
#                            Library.printMessage("Ploting "+GlobalVars.Lag['diag'][di]+" for "+nprod)
#                            PlotField.PlotField.Plot(cruise,GlobalVars.Lag['diag'][di],opt,type='Lagrangian')
#                    else:
#                        Library.Done('None')
#    return
#
