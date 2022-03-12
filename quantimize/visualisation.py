from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from netCDF4 import Dataset
import datetime
import quantimize.data_access as da


def get_FL_index(fl_list, FL):
    """Return index of flightlevel in hPa for atmo data for a given flightlevel in FL

    Args:
        fl_list (list): List with available flight levels in hPa
        FL (int/float/str): Current flightlevel in FL

    Returns:
        int: Index position
    """
    hPa_fl = da.get_hPa(FL)
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

def make_atmo_map(FL,time):
    """Create map with a given flightlevel and time

    Args:
        FL (int/float/str): Current flightlevel
        time (datetime.time): Current time
    """
    lat_list, long_list, time_list, fl_list, atmo_data = da.get_atmo_raw_data()
    available_fl = da.get_fl(FL)
    FL_index = get_FL_index(fl_list, available_fl)
    available_time = da.get_time(time)
    time_index = get_time_index(available_time)
    map = Basemap(llcrnrlon=-30,llcrnrlat=34,urcrnrlon=30,urcrnrlat=60,resolution="i",projection="merc")
    map.drawcoastlines()
    map.drawcountries()
    long_grid, lat_grid = np.meshgrid(long_list,lat_list)
    x,y = map(long_grid, lat_grid)
    cs = map.pcolormesh(x,y,atmo_data[time_index][FL_index],vmin=-0.047278133046295995,vmax=0.34942841580572637)
    cbar = map.colorbar(cs, location="bottom", pad="10%")
    cbar.set_label("Delta C")
    #map.drawgreatcircle(-20,40,20,60)
    plt.title("Atmospheric data map. Time:"+available_time+" FL: "+available_fl)
    plt.show()


def make_animated_atmo_day_map(FL):
    fig, ax = plt.subplots()
    m = Basemap(ax=ax, llcrnrlon=-30,llcrnrlat=34,urcrnrlon=30,urcrnrlat=60,resolution="i",projection="merc")
    m.drawcoastlines()
    m.drawcountries()
    lat_list, long_list, time_list, fl_list, atmo_data = da.get_atmo_raw_data()
    available_fl = da.get_fl(FL)
    FL_index = get_FL_index(fl_list, available_fl)
    print(available_fl)

    def animate_map(time):
        long_grid, lat_grid = np.meshgrid(long_list,lat_list)
        x,y = m(long_grid, lat_grid)
        cs = m.pcolormesh(x,y,atmo_data[time][FL_index],vmin=-0.047278133046295995,vmax=0.34942841580572637) 

    ani = FuncAnimation(fig, animate_map, range(3))
    plt.show()

def make_animated_atmo_FL_map(time):
    fig, ax = plt.subplots()
    m = Basemap(ax=ax, llcrnrlon=-30,llcrnrlat=34,urcrnrlon=30,urcrnrlat=60,resolution="i",projection="merc")
    m.drawcoastlines()
    m.drawcountries()
    lat_list, long_list, time_list, fl_list, atmo_data = da.get_atmo_raw_data()
    available_time = da.get_time(time)
    time_index = get_time_index(available_time)
    print(available_time)

    def animate_map(FL_index):
        long_grid, lat_grid = np.meshgrid(long_list,lat_list)
        x,y = m(long_grid, lat_grid)
        cs = m.pcolor(x,y,atmo_data[time_index][FL_index],vmin=-0.047278133046295995,vmax=0.34942841580572637) 

    ani = FuncAnimation(fig, animate_map, range(14))
    plt.show()

def draw_flight_path_on_map(map, trajectories):
    long = []
    lat = []
    fl = []
    time = []
    for point in trajectories:
        long.append(point[0])
        lat.append(point[1])
        fl.append(point[2])
        time.append(point[3])
    return map.scatter(long,lat,c=fl,cmap="viridis",vmin=100,vmax=400,latlon=True,s=1)

def animate_flight_path_on_map(list_of_trajectories, dt):
    time_list, time_grid = da.create_time_grid(dt)
    time_grid = da.map_trajectory_to_time_grid(list_of_trajectories, time_grid)
    fig, ax = plt.subplots()
    m = Basemap(ax=ax, llcrnrlon=-32,llcrnrlat=32,urcrnrlon=32,urcrnrlat=62,resolution="i",projection="merc")
    m.drawcoastlines()
    m.drawcountries()

    def animate_map(time):
        cs = m.scatter(time_grid[time]["LONG"],time_grid[time]["LAT"],c=time_grid[time]["FL"],cmap="viridis",vmin=100,vmax=400,latlon=True,s=1)

    ani = FuncAnimation(fig, animate_map, time_list, interval=20)
    plt.show()

def make_map():
    m = Basemap(llcrnrlon=-35,llcrnrlat=33,urcrnrlon=35,urcrnrlat=61,resolution="i",projection="merc")
    m.drawcoastlines()
    m.drawcountries()
    return m