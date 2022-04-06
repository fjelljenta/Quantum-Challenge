import math
import datetime


def kts_to_ms(kts):
    """Convert speed in kts to m/s

    Args:
        kts (int/float): Flight speed in knots

    Returns:
        float: Flight speed in m/s
    """
    return 1.852*kts/3.6


def ms_to_kts(ms):
    """Convert speed in m/s to kts

    Args:
        ms (int/float): Flight speed in m/s

    Returns:
        float: Flight speed in knots
    """
    return ms/1.852*3.6


def ftm_to_fls(ftm):
    """Convert feet per minute to flight level per second

    Args:
        ftm (int/float): Flight speed in feet per minute

    Returns:
        float: Flight speed in flight level per second
    """
    return ftm/60/100


def fls_to_ftm(fls):
    """Convert flight level per second to feet per minute

    Args:
        fls (int/float): Flight speed in flight level per second

    Returns:
        float: Flight speed in feet per minute
    """
    return fls * 60 * 100


def ms_to_fls(ms):
    """Convert meter per second to flight level per second

    Args:
        ms (int/float): Flight speed in meter per second

    Returns:
        float: Flight speed in flight level per second
    """
    return ms * 3.28084 / 100


def fls_to_ms(fls):
    """Convert flight level per second to meter per second

    Args:
        fls (int/float): Flight speed in flight level per second

    Returns:
        float: Flight speed in meter per second
    """
    return fls / 3.28084 * 100


def ms_to_kmm(ms):
    """Convert Flight speed in m/s to km/min

    Args:
        ms (int/float): Flight speed in meter per second

    Returns:
        float: Flight speed in kilometer per minute
    """
    return ms*60/1000


def kmm_to_ms(kmm):
    """Convert Flight speed in km/min to m/s

    Args:
        kmm (int/float): Flight speed in kilometer per minute

    Returns:
        float: Flight speed in meter per second
    """
    return kmm/60*1000


def ms_to_kms(ms):
    """Convert Flight speed in m/s to km/s

    Args:
        ms (int/float): Flight speed in meter per second

    Returns:
        float: Flight speed in kilometer per second
    """
    return ms/1000


def kms_to_ms(kms):
    """Convert Flight speed in km/s to m/s

    Args:
        kms (int/float): Flight speed in kilometer per second

    Returns:
        float: Flight speed in meter per second
    """
    return kms*1000


def fl_to_km(fl):
    """Convert height in flight level to kilometer

    Args:
        fl (int/float): height in flight level

    Returns:
        float: height in kilometer
    """
    return fl*100*0.0003048


def coordinates_to_distance3D(start_long, start_lat, start_FL, stop_long, stop_lat, stop_FL):
    """Convert two 3D coordinates to a distance in km

    Args:
        start_long (int/float): Start longitudinal value
        start_lat (int/float): Start latitudinal value
        start_FL (int/float): Start flightlevel value
        stop_long (int/float): Stop longitudinal value
        stop_lat (int/float): Stop latitudinal value
        stop_FL (int/float): Stop flightlevel value

    Returns:
        float: Distance in km
    """
    diff_long = stop_long-start_long
    diff_lat = stop_lat-start_lat
    diff_FL= stop_FL-start_FL
    diff_long_km = diff_long*85
    diff_lat_km = diff_lat*111
    diff_FL_km=100*diff_FL*0.0003048#1 feet is 0,0003048km and 1FL is 100feet
    distance = math.sqrt(diff_long_km**2+diff_lat_km**2+diff_FL_km**2)
    return round(distance, 2)   #Todo: Check if that is necessary


def coordinates_to_distance(start_long, start_lat, stop_long, stop_lat):
    """Convert two coordinates to a distance in km

    Args:
        start_long (int/float): Start longitudinal value
        start_lat (int/float): Start latitudinal value
        stop_long (int/float): Stop longitudinal value
        stop_lat (int/float): Stop latitudinal value

    Returns:
        float: Distance in km
    """
    diff_long = stop_long-start_long
    diff_lat = stop_lat-start_lat
    diff_long_km = diff_long*85
    diff_lat_km = diff_lat*111
    distance = math.sqrt(diff_long_km**2+diff_lat_km**2)
    return round(distance, 2)   #Todo: Check if that is necessary


def calculate_min_radius(TAS):
    """Calculates the curvature radius for a given true air speed

    Args:
        TAS (int/float): True air speed in kt/s

    Returns:
        float: Curvature radius in meter
    """
    TAS_ms = kts_to_ms(TAS)
    g = 9.81
    beta = 25
    radius = TAS_ms**2/(g*math.tan(beta/360*(2*math.pi)))
    return radius


def update_time(current_time, dt):
    """Calculates the next time step

    Args:
        current_time (datetime.time): Current time
        dt (int): Timestep in s

    Returns:
        datetime.time: Next point in time
    """
    hr = current_time.hour
    m = current_time.minute
    s = current_time.second
    return datetime.time(hr+(m+(dt+s)//60)//60, (m+(dt+s)//60)%60, (dt+s)%60)


def datetime_to_seconds(current_time):
    """Calculates datetime in seconds with midnight (00:00:00) as reference point

        Args:
            current_time (datetime.time): Current time

        Returns:
            float: Current time in seconds relative to reference point (00:00:00)
        """
    hr = current_time.hour
    m = current_time.minute
    s = current_time.second
    return 3600 * hr + 60 * m + s


def seconds_to_datetime(current_time):
    """Calculates current datetime from datetime given in seconds after the reference point at midnight (00:00:00)

        Args:
            current_time (float/int): Current time in seconds relative to reference point (00:00:00)

        Returns:
            datetime.time: Current time
        """
    hr = current_time//3600
    m = (current_time - hr * 3600) // 60
    s = current_time - hr * 3600 - m * 60
    return datetime.time(hr, m, s)