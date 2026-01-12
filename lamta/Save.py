import numpy as np
from netCDF4 import Dataset


class Create:
    @classmethod
    def netcdf(cls, fname, lon, lat, vvar, title, var_name, var_units, **kwargs):
        file = Dataset(fname, mode="w", format="NETCDF4_CLASSIC")
        file.createDimension("lon", len(lon))
        file.createDimension("lat", len(lat))
        file.createDimension("time", 1)
        file.title = title
        latitude = file.createVariable("lat", np.float32, ("lat",))
        latitude.units = "degrees_north"
        longitude = file.createVariable("lon", np.float32, ("lon",))
        longitude.units = "degrees_east"
        longitude[:] = lon
        latitude[:] = lat
        var = file.createVariable(var_name, np.float32, ("time", "lat", "lon"))
        var.units = var_units
        var[:, :, :] = vvar
        if "var2" in kwargs:
            var2 = file.createVariable(kwargs["var2_name"], np.float32, ("time", "lat", "lon"))
            var2.units = kwargs["var2_units"]
            var2[:, :, :] = kwargs["var2"]
        if "var3" in kwargs:
            var3 = file.createVariable(kwargs["var3_name"], np.float32, ("time", "lat", "lon"))
            var3.units = kwargs["var3_units"]
            var3[:, :, :] = kwargs["var3"]
        file.close()
        return
