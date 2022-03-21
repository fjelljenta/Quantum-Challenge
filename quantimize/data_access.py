from dataclasses import dataclass
import json
import datetime
import copy
from netCDF4 import Dataset
import os
import quantimize.converter as cv

base_dir = os.getcwd()
with open(base_dir+"/quantimize/data/atmo.json", "rb") as f:
    atmo_data = json.loads(f.read().decode("utf-8"))
with open(base_dir+"/quantimize/data/bada_data.json", "rb") as f:
    flight_level_data = json.loads(f.read().decode("utf-8"))
with open(base_dir+"/quantimize/data/flights.json","rb") as f:
    flight_data = json.loads(f.read().decode("utf-8"))
nc = Dataset(base_dir+"/quantimize/data/aCCf_0623_p_spec.nc")


def get_flight_info(flight_nr):
    """Return the flight information for a given flight number

    Args:
        flight_nr (int/str/float): Flight number

    Returns:
        dict: Flight number information with start_time as datetime.time object
                (start_time, start_flight_level, start_long, start_lat, end_long, end_lat)
    """
    start_time = datetime.datetime.strptime(flight_data[str(flight_nr)]["start_time"], "%H:%M:%S")
    start_time = datetime.time(start_time.hour, start_time.minute, start_time.second)
    this_flight = copy.copy(flight_data[str(flight_nr)])
    this_flight["start_time"] = start_time
    return this_flight


def get_fl(arb_fl):
    """Find next available flight level in given flight level grid (steps of 20 flight levels)

    Args:
        arb_fl (int/str/float): Current flight level

    Returns:
        str: Next available flight level in grid
    """
    arb_fl = int(arb_fl)
    diff = arb_fl%20
    if diff == 0:
        pass
    elif diff >= 10:
        arb_fl = arb_fl+20-diff
    elif diff < 10:
        arb_fl = arb_fl-diff
    return str(arb_fl)


def get_flight_level_data(flight_level):
    """Return information to the next available flight level

    Args:
        flight_level (int/float/str): Flight level

    Returns:
        dict: Dictionary containing information about the flight level from the bada_data.json
    """
    flight_level = get_fl(flight_level)
    return flight_level_data[flight_level]


def get_long(arb_long):
    """Convert and get longitude ready for data access

    Args:
        arb_long (int/str/float): Longitude value

    Returns:
        str: Converted longitude value for atmo data
    """
    arb_long = float(arb_long)
    #arb_long = 2 * round(arb_long/2,0)    #Todo: Alternative way, last part (with -0.0 removal) still needed, tests!

    diff = arb_long%2
    if diff == 0:
        return str(arb_long)
    elif diff > 0.5 and diff < 1:
        arb_long = round(arb_long-1,0)
        if arb_long == -0.0:
            return str(abs(arb_long))
        else:
            return str(arb_long)
    elif diff >=1 and diff <= 1.5:
        return str(round(arb_long+1,0))
    else:
        arb_long = round(arb_long,0)
        if arb_long == -0.0:
            return str(abs(arb_long))
        else:
            return str(arb_long)


def get_lat(arb_lat):
    """Convert and get latitude ready for data access

    Args:
        arb_lat (int/str/float): Latitude value

    Returns:
        str: Convert latitude value for atmo data
    """
    arb_lat = float(arb_lat)
    # arb_long = 2 * round(arb_long/2,0)    #Todo: Alternative way, tests!
    diff = arb_lat%2
    if diff == 0:
        return str(arb_lat)
    elif diff > 0.5 and diff < 1:
        return str(round(arb_lat-1,0))
    elif diff >=1 and diff <= 1.5:
        return str(round(arb_lat+1,0))
    else:
        return str(round(arb_lat,0))


def avoid_empty_atmo_data(fl):
    """Avoid flightlevels without data, give back closest data

    Args:
        fl (int): flight level to check

    Returns:
        int: nearest flightlevel with available data
    """
    if fl < 140:
        return 140
    elif fl == 220:
        return 200
    elif fl == 280:
        return 260
    elif fl == 320:
        return 300
    elif fl > 380:
        return 380
    else:
        return fl


def get_fl_atmo(arb_fl):
    """Find next available flightlevel with given atmospheric data to given level

    Args:
        arb_fl (int/str/float): Current flight level

    Returns:
        str: Flighlevel for atmo data
    """
    corrected_fl = int(get_fl(arb_fl))
    corrected_fl = avoid_empty_atmo_data(corrected_fl)
    return str(corrected_fl)





def get_time(arb_time):
    """Get the next available time for the atmo data

    Args:
        arb_time (time): Current time of the airplane

    Returns:
        str: Corresponding time for atmo data
    """
    arb_time = datetime.datetime(2018,6,23,arb_time.hour,arb_time.minute,arb_time.second)
    six_am = datetime.datetime(2018,6,23,6)
    nine_am = datetime.datetime(2018,6,23,9)
    twelve = datetime.datetime(2018,6,23,12)
    three_pm = datetime.datetime(2018,6,23,15)
    six_pm = datetime.datetime(2018,6,23,18)
    if arb_time < nine_am:
        return six_am.strftime("%H:%M:%S")
    elif arb_time >= nine_am and arb_time < three_pm:
        return twelve.strftime("%H:%M:%S")
    elif arb_time >= three_pm:
        return six_pm.strftime("%H:%M:%S")


def get_merged_atmo_data(arb_long, arb_lat, arb_fl, arb_time):
    """Returns the merged climate impact data for a given set of location, flightlevel and time

    Args:
        arb_long (int/str/float): Current longitude
        arb_lat (int/str/float): Current latitude
        arb_fl (int/str/float): Current flightlevel
        arb_time (datetime.time): Current time

    Raises:
        Exception: If time was not a datetime.time object

    Returns:
        float: Merged climate data
    """
    LONG = get_long(arb_long)
    LAT = get_lat(arb_lat)
    FL = get_fl_atmo(arb_fl)
    if type(arb_time) is datetime.time:
        TIME = get_time(arb_time)
    else:
        raise Exception("Time is not a time object")
    return atmo_data[LONG][LAT][FL][TIME]["MERGED"]


def get_atmo_raw_data():
    """Returns the raw data of the nc atmo fil

    Returns:
        quintuple: Latitude, Longitude, Time, FL, Merged data
    """
    lat = nc.variables["LATITUDE"][:]
    long = nc.variables["LONGITUDE"][:]
    time = nc.variables["TIME"][:].tolist()
    fl = nc.variables["LEVEL11_24"][:].tolist()
    atmo_data = nc.variables["MERGED"][:]
    return lat, long, time, fl, atmo_data


def get_hPa(fl):
    """Returns the flightlevel in hPa for a given FL in FL

    Args:
        fl (int/string/float): Flightlevel in FL

    Returns:
        int: Flightlevel in hPa
    """
    fl = int(get_fl(fl))
    if fl <= 140:
        return 600
    elif fl == 160:
        return 550
    elif fl == 180:
        return 500
    elif fl == 200:
        return 450
    elif fl == 220:
        return 450
    elif fl == 240:
        return 400
    elif fl == 260:
        return 350
    elif fl == 280:
        return 350
    elif fl == 300:
        return 300
    elif fl == 320:
        return 300
    elif fl == 340:
        return 250
    elif fl == 360:
        return 225
    elif fl >= 380:
        return 200


def create_time_grid(dt):
    """Creates a time grid for plotting with no flight data yet included

    Args:
        dt (int): Timestep in seconds

    Returns:
        tuple: List of timepoints and time grid as dict
    """
    time_grid = {}
    time_list = []
    start = datetime.time(6)
    end = datetime.time(23)
    time_grid[start] = {"LONG":[], "LAT":[], "FL":[], "FL_NR":[]}
    time_list.append(start)
    count = int((end.hour-start.hour)*3600/dt)
    for i in range(count):
        start = cv.update_time(start,dt)
        time_grid[start] = {"LONG":[], "LAT":[], "FL":[], "FL_NR":[]}
        time_list.append(start)
    return time_list, time_grid


def map_trajectory_to_time_grid(trajectories, time_grid):
    """Maps a list of trajectories to a corresponding timegrid with according time steps

    Args:
        trajectories (list): List of trajectories
        time_grid (dict): Time grid

    Returns:
        dict: Filled time grid
    """
    for trajectory in trajectories:
        for point in trajectory["trajectory"]:
            time_grid[point[3]]["FL_NR"].append(trajectory["flight_nr"])
            time_grid[point[3]]["LONG"].append(point[0])
            time_grid[point[3]]["LAT"].append(point[1])
            time_grid[point[3]]["FL"].append(point[2])
    return time_grid