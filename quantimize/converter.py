import math
import datetime


def kts_to_ms(kts):
    """Convert kts to ms

    Args:
        kts (int/float): Knots

    Returns:
        float: ms
    """
    return 1.852*kts/3.6


def ms_to_kts(ms):
    """Convert ms to kts

    Args:
        ms (int/float): m/s

    Returns:
        float: kts
    """
    return ms/1.852*3.6


def ms_to_kmm(ms):
    """
    Convert ms to kmm
    :param ms: m/s
    :return: km/min
    """
    return ms*60/1000


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
    return round(distance, 2)


def calculate_min_radius(TAS):
    """Calculates the curveture radius for a given true air speed

    Args:
        TAS (int/float): True air spee

    Returns:
        float: Curveture radius
    """
    TAS_ms = kts_to_ms(TAS)
    g = 9.81
    beta = 25
    radius = TAS_ms**2/g*math.tan(beta/(2*math.pi))
    return radius

def update_time(current_time, dt):
    """Calculates the next time step

    Args:
        current_time (datetime.time): Current time
        dt (int): Timestep

    Returns:
        datetime.time: Next timestep
    """
    hr = current_time.hour
    m = current_time.minute
    return datetime.time(hr+(dt+m)//60, (dt+m) % 60)