from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from netCDF4 import Dataset
import datetime
try:
    from quantimize.data_access import get_atmo_raw_data, get_time, get_fl, get_hPa
except ModuleNotFoundError:
    from data_access import get_atmo_raw_data, get_time, get_fl, get_hPa

def get_FL_index(fl_list, FL):
    """Return index of flightlevel in hPa for atmo data for a given flightlevel in FL

    Args:
        fl_list (list): List with available flight levels in hPa
        FL (int/float/str): Current flightlevel in FL

    Returns:
        int: Index position
    """
    hPa_fl = get_hPa(FL)
    return fl_list.index(hPa_fl)

def get_time_index(time):
    """Return index of time_list for a given time

    Args:
        time (datetime.time): Current time

    Returns:
        int: Index position for time
    """
    date = datetime.datetime.strptime(time, "%H:%M:%S")
    if date.hour == 6:
        return 0
    elif date.hour == 12:
        return 1
    elif date.hour == 18:
        return 2

def make_map(FL,time):
    """Create map with a given flightlevel and time

    Args:
        FL (int/float/str): Current flightlevel
        time (datetime.time): Current time
    """
    lat_list, long_list, time_list, fl_list, atmo_data = get_atmo_raw_data()
    available_fl = get_fl(FL)
    FL_index = get_FL_index(fl_list, available_fl)
    available_time = get_time(time)
    time_index = get_time_index(available_time)
    map = Basemap(llcrnrlon=-30,llcrnrlat=34,urcrnrlon=30,urcrnrlat=60,resolution="i",projection="merc")
    map.drawcoastlines()
    map.drawcountries()
    long_grid, lat_grid = np.meshgrid(long_list,lat_list)
    x,y = map(long_grid, lat_grid)
    cs = map.pcolor(x,y,atmo_data[time_index][FL_index])
    cbar = map.colorbar(cs, location="bottom", pad="10%")
    cbar.set_label("Delta C")
    map.drawgreatcircle(-20,40,20,60)
    plt.title("Atmospheric data map. Time:"+available_time+" FL: "+available_fl)
    plt.show()


def make_animated_day_map(FL):
    fig, ax = plt.subplots()
    m = Basemap(ax=ax, llcrnrlon=-30,llcrnrlat=34,urcrnrlon=30,urcrnrlat=60,resolution="i",projection="merc")
    m.drawcoastlines()
    m.drawcountries()
    lat_list, long_list, time_list, fl_list, atmo_data = get_atmo_raw_data()
    available_fl = get_fl(FL)
    FL_index = get_FL_index(fl_list, available_fl)
    print(available_fl)

    def animate_map(time):
        long_grid, lat_grid = np.meshgrid(long_list,lat_list)
        x,y = m(long_grid, lat_grid)
        cs = m.pcolormesh(x,y,atmo_data[time][FL_index]) 

    ani = FuncAnimation(fig, animate_map, range(3))
    plt.show()

def make_animated_FL_map(time):
    fig, ax = plt.subplots()
    m = Basemap(ax=ax, llcrnrlon=-30,llcrnrlat=34,urcrnrlon=30,urcrnrlat=60,resolution="i",projection="merc")
    m.drawcoastlines()
    m.drawcountries()
    lat_list, long_list, time_list, fl_list, atmo_data = get_atmo_raw_data()
    available_time = get_time(time)
    time_index = get_time_index(available_time)
    print(available_time)

    def animate_map(FL_index):
        long_grid, lat_grid = np.meshgrid(long_list,lat_list)
        x,y = m(long_grid, lat_grid)
        cs = m.pcolor(x,y,atmo_data[time_index][FL_index]) 

    ani = FuncAnimation(fig, animate_map, range(14))
    plt.show()

if __name__ == "__main__":
    make_map(115, datetime.time(6,0,0))