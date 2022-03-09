import json

with open("bada_data.json", "rb") as f:
    flight_level_data = json.loads(f.read().decode("utf-8"))


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
    return str(arb_fl)

def get_flight_level_data(flight_level):
    """Return information to the next available flight level

    Args:
        flight_level (int/float/str): Flight level

    Returns:
        dict: Dictionary containing information about the flight level
    """
    flight_level = get_fl(flight_level)
    return flight_level_data[flight_level]