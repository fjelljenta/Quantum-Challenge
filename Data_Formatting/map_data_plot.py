from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

map = Basemap(llcrnrlat=32,llcrnrlon=-28,urcrnrlat=62,urcrnrlon=32,projection="merc", resolution="i")
map.drawcoastlines(linewidth=0.25)
map.drawcountries(linewidth=0.25)

