from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import datetime
import quantimize.data_access as da
import quantimize.converter as cv


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

def make_map():
    """Creates a map
    Returns:
        m : basemap
    """
    m = Basemap(llcrnrlon=-35,llcrnrlat=33,urcrnrlon=35,urcrnrlat=61,resolution="i",projection="merc")
    m.drawcoastlines()
    m.drawcountries()
    return m

def make_3d_map():
    ax = plt.figure().add_subplot(projection='3d')
    m = Basemap(ax=ax, llcrnrlon=-35,llcrnrlat=33,urcrnrlon=35,urcrnrlat=61,resolution="i",projection="merc",fix_aspect=False)
    ax.add_collection3d(m.drawcoastlines())
    ax.add_collection3d(m.drawcountries())
    return ax, m

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

############# Atmo animation ################
def make_animated_atmo_day_map(FL):
    """Creation of an animated atmo data map over the day for one flight level

    Args:
        FL (int): flight level
    """
    fig, ax = plt.subplots()
    m = Basemap(ax=ax, llcrnrlon=-30,llcrnrlat=34,urcrnrlon=30,urcrnrlat=60,resolution="i",projection="merc")
    m.drawcoastlines()
    m.drawcountries()
    lat_list, long_list, time_list, fl_list, atmo_data = da.get_atmo_raw_data()
    available_fl = da.get_fl(FL)
    FL_index = get_FL_index(fl_list, available_fl)
    #print(available_fl)

    def animate_map(time):
        long_grid, lat_grid = np.meshgrid(long_list,lat_list)
        x,y = m(long_grid, lat_grid)
        cs = m.pcolormesh(x,y,atmo_data[time][FL_index],vmin=-0.047278133046295995,vmax=0.34942841580572637)

    ani = FuncAnimation(fig, animate_map, range(3))
    plt.show()

def make_animated_atmo_FL_map(time):
    """Creation of an animated atmo data map for different flight level at one time point

    Args:
        time (time): time point for the animation
    """
    fig, ax = plt.subplots()
    m = Basemap(ax=ax, llcrnrlon=-30,llcrnrlat=34,urcrnrlon=30,urcrnrlat=60,resolution="i",projection="merc")
    m.drawcoastlines()
    m.drawcountries()
    lat_list, long_list, time_list, fl_list, atmo_data = da.get_atmo_raw_data()
    available_time = da.get_time(time)
    time_index = get_time_index(available_time)
    print(fl_list)

    def animate_map(FL_index):
        print(FL_index)
        long_grid, lat_grid = np.meshgrid(long_list,lat_list)
        x,y = m(long_grid, lat_grid)
        cs = m.pcolor(x,y,atmo_data[time_index][FL_index],vmin=-0.047278133046295995,vmax=0.34942841580572637)

    ani = FuncAnimation(fig, animate_map, range(14))
    plt.show()


################## Normal plots of flight trajectories #################
def scatter_flight_path_on_map(map, trajectory):
    """makes a scatter plot of a flight path on a map

    Args:
        map:
        trajectory (list): list of trajectory elements

    Returns:
        map with flight path
    """
    flight_path = cv.check_trajectory_dict(trajectory)
    long, lat, fl, time = cv.trajectory_elements_to_list(flight_path)
    map.scatter(long, lat, c=fl, cmap="viridis", vmin=100, vmax=400, latlon=True, s=1)
    return map

def plot_flight_path_on_map(map, trajectory):
    """makes a normal line plot of a flight path on a map

    Args:
        map:
        trajectory (list): list of trajectory elements

    Returns:
        map with flight path
    """
    flight_path = cv.check_trajectory_dict(trajectory)
    long, lat, fl, time = cv.trajectory_elements_to_list(flight_path)
    map.plot(long, lat, latlon=True)
    return map

############# 3D Plots ########################
def scatter_flight_path_on_map_3d(ax, map, trajectory):
    """Makes a scatterplot on a 3d map of the flight trajectory

    Args:
        ax (matplotlib.Axes): Matplotlib axes
        map (matplotlib.Figure): The actual map
        trajectory (list/dict): List or dict with the trajectory points

    Returns:
        ax, map: Returns the ax and map (might not be needed)
    """
    flight_path = cv.check_trajectory_dict(trajectory)
    long, lat, fl, time = cv.trajectory_elements_to_list(flight_path)
    x,y = map(long, lat)
    ax.scatter(x,y,fl)
    ax.scatter(x,y,0,"k")
    ax.set_zlim(0,400)
    return ax, map

def plot_flight_path_on_map_3d(ax, map, trajectory):
    """Makes a lineplot on a 3d map of the flight trajectory

    Args:
        ax (matplotlib.Axes): Matplotlib axes
        map (matplotlib.Figure): The actual map
        trajectory (list/dict): List or dict with the trajectory points

    Returns:
        ax, map: Returns the ax and map (might not be needed)
    """
    flight_path = cv.check_trajectory_dict(trajectory)
    long, lat, fl, time = cv.trajectory_elements_to_list(flight_path)
    x,y = map(long, lat)
    ax.plot(x,y,fl)
    ax.plot(x,y,0,"k")
    ax.set_zlim(0,400)
    return ax, map

def plot_flight_path_on_map_3d_with_atmo_as_points(ax, map, trajectory):
    """Makes a scatterplot on a 3d map of the flight trajectory and adds colors to the points according to the atmospheric data values for that point

    Args:
        ax (matplotlib.Axes): Matplotlib axes
        map (matplotlib.Figure): The actual map
        trajectory (list/dict): List or dict with the trajectory points

    Returns:
        ax, map: Returns the ax and map (might not be needed)
    """
    atmo = []
    flight_path = cv.check_trajectory_dict(trajectory)
    for point in flight_path:
        atmo.append(da.get_merged_atmo_data(point[0],point[1],point[2],point[3]))
    long, lat, fl, time = cv.trajectory_elements_to_list(flight_path)
    x,y = map(long, lat)
    ax.scatter(x,y,fl,c=atmo,vmin=-0.047278133046295995,vmax=0.34942841580572637,s=1)
    ax.plot(x,y,0,"k")
    ax.set_zlim(0,400)
    return ax, map

def plot_flight_path_on_map_3d_with_atmo_as_slices(ax,map,trajectory):
    """Makes a lineplot on a 3d map of the flight trajectory and adds colored slices to the points according to the atmospheric data values for that point

    Args:
        ax (matplotlib.Axes): Matplotlib axes
        map (matplotlib.Figure): The actual map
        trajectory (list/dict): List or dict with the trajectory points

    Returns:
        ax, map: Returns the ax and map (might not be needed)
    """
    flight_path = cv.check_trajectory_dict(trajectory)
    for point in flight_path:
        xbounds = (point[0]-1,point[0]+1)
        ybounds = (point[1]-0.5,point[1]+0.5)
        #xbounds, ybounds = cv.trajectory_point_boundaries(point)
        xb, yb = np.meshgrid(xbounds, ybounds)
        c1 = da.get_merged_atmo_data(xbounds[0],ybounds[0],point[2],point[3])
        c2 = da.get_merged_atmo_data(xbounds[0],ybounds[1],point[2],point[3])
        c3 = da.get_merged_atmo_data(xbounds[1],ybounds[0],point[2],point[3])
        c4 = da.get_merged_atmo_data(xbounds[1],ybounds[1],point[2],point[3])
        ax.add_collection3d(map.pcolor(xb,yb,[[c1,c3],[c2,c4]],latlon=True,vmin=-0.047278133046295995,vmax=0.34942841580572637,alpha=0.3),zs=point[2])
    long, lat, fl, time = cv.trajectory_elements_to_list(flight_path)
    x,y = map(long, lat)
    ax.plot(x,y,fl,"k")
    ax.plot(x,y,0,"k")
    ax.set_zlim(0,400)
    return ax, map
    
############# Animation ##################
def animate_flight_path_on_map(list_of_trajectories, dt):
    """annimates fight pathes on a map

    Args:
        list_of_trajectories (list of trajectory dicts): list of trajectory dicts
        dt (int): time step of the animation

    """
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
