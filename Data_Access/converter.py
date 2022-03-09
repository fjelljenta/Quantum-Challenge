import math

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
    TAS_ms = kts_to_ms(TAS)
    g = 9.81
    beta = 25
    radius = TAS_ms**2/g*math.tan(beta/(2*math.pi))
    return radius