from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from netCDF4 import Dataset

base_dir = "C:/Users/Jakob/Documents/TUM/Master/4.Semester-QST/Quantum-Challenge/Data/"
map = Basemap(llcrnrlat=32,llcrnrlon=-28,urcrnrlat=62,urcrnrlon=32,projection="merc", resolution="i")
map.drawcoastlines(linewidth=0.25)
map.drawcountries(linewidth=0.25)



