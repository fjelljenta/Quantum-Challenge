from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from netCDF4 import Dataset

base_dir = "C:/Users/Jakob/Documents/TUM/Master/4.Semester-QST/Quantum-Challenge/Data/"
nc = Dataset(base_dir+"aCCf_0623_p_spec.nc")
lat = nc.variables["LATITUDE"][:]
long = nc.variables["LONGITUDE"][:]
time = nc.variables["TIME"][:].tolist()
fl = nc.variables["LEVEL11_24"][:].tolist()
atmo_data = nc.variables["MERGED"][:]

def get_FL_index(FL):
    """Return index of the given flightlevel

    Args:
        FL (int): Flightlevel in hPa!

    Returns:
        int: Index position
    """
    return fl.index(FL)

def get_time_index(hour):
    """Return index of the given time

    Args:
        hour (int): Hour (6,12,18)

    Returns:
        int: Index position
    """
    if hour == 6:
        return time.index(534)
    elif hour == 12:
        return time.index(540)
    elif hour == 18:
        return time.index(546)

def make_map(FL,time):
    """Create map with a given flightlevel and time

    Args:
        FL (int): Flightlevel
        time (int): Time (6,12,18)
    """
    FL = get_FL_index(FL)
    time = get_time_index(time)
    map = Basemap(llcrnrlon=-30,llcrnrlat=34,urcrnrlon=30,urcrnrlat=60,resolution="i",projection="merc")
    map.drawcoastlines()
    map.drawcountries()
    long_grid, lat_grid = np.meshgrid(long,lat)
    x,y = map(long_grid, lat_grid)
    cs = map.pcolor(x,y,atmo_data[time][FL])
    cbar = map.colorbar(cs, location="bottom", pad="10%")
    cbar.set_label("Delta C")
    plt.title("Atmospheric data map")
    plt.show()


"""
fig, ax = plt.subplots()
m = Basemap(ax=ax, llcrnrlon=-30,llcrnrlat=34,urcrnrlon=30,urcrnrlat=60,resolution="i",projection="merc")
m.drawcoastlines()
m.drawcountries()

def animate_map(time):
    long_grid, lat_grid = np.meshgrid(long,lat)
    x,y = m(long_grid, lat_grid)
    cs = m.pcolor(x,y,atmo_data[0][time]) 

ani = FuncAnimation(fig, animate_map, range(14))
plt.show()
"""
