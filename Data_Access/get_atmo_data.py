import json
import datetime

with open("atmo.json", "rb") as f:
    atmo_data = json.loads(f.read().decode("utf-8"))

def get_long(arb_long):
    """Convert and get longitute ready for data access

    Args:
        arb_long (int/str/float): Longitude value

    Returns:
        str: Converted longitude value for atmo data
    """
    arb_long = float(arb_long)
    diff = arb_long%2
    if diff == 0:
        return str(arb_long)
    elif diff > 0.5 and diff < 1:
        return str(round(arb_long-1,0))
    elif diff >=1 and diff <= 1.5:
        return str(round(arb_long+1,0))
    else:
        return str(round(arb_long,0))

def get_lat(arb_lat):
    """Convert and get latitude ready for data access

    Args:
        arb_lat (int/str/float): Latitude value

    Returns:
        str: Convert latitude value for atmo data
    """
    arb_lat = float(arb_lat)
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
        fl (int): Flightlevel to check

    Returns:
        int: Corrected flighlevel
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

def get_fl(arb_fl):
    """Find next available flightlevel to given level

    Args:
        arb_fl (int/str/float): Current flight level

    Returns:
        str: Flighlevel for atmo data
    """
    arb_fl = int(arb_fl)
    diff = arb_fl%20
    if diff == 0:
        pass
    elif diff >= 10:
        arb_fl = arb_fl+20-diff
    elif diff < 10:
        arb_fl = arb_fl-diff
    arb_fl = avoid_empty_atmo_data(arb_fl)
    return str(arb_fl)

def get_time(arb_time):
    """Get the corresponding time for the atmo data

    Args:
        arb_time (time): Current time of the airplane

    Returns:
        str: Corresponding time for atmo data
    """
    arb_time = datetime.datetime(2018,6,23,arb_time.hour,arb_time.minute,arb_time.second)
    six_am = datetime.datetime(2018,6,23,6)
    twelve = datetime.datetime(2018,6,23,12)
    six_pm = datetime.datetime(2018,6,23,18)
    if arb_time >= six_am and arb_time < twelve:
        return six_am.strftime("%H:%M:%S")
    elif arb_time >= twelve and arb_time < six_pm:
        return twelve.strftime("%H:%M:%S")
    elif arb_time >= six_pm:
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
    FL = get_fl(arb_fl)
    if type(arb_time) is datetime.time:
        TIME = get_time(arb_time)
    else:
        raise Exception("Time is not a time object")
    print(LONG, LAT, FL, TIME)
    return atmo_data[LONG][LAT][FL][TIME]["MERGED"]
