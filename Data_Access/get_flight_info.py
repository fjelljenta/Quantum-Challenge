import json
import datetime
import copy

with open("flights.json","rb") as f:
    flight_data = json.loads(f.read().decode("utf-8"))

def get_flight_info(flight_nr):
    """Return the flight information for a given flight number

    Args:
        flight_nr (int/str/float): Flight number

    Returns:
        dict: Flight number information
    """
    start_time = datetime.datetime.strptime(flight_data[str(flight_nr)]["start_time"], "%H:%M:%S")
    start_time = datetime.time(start_time.hour, start_time.minute, start_time.second)
    this_flight = copy.copy(flight_data[str(flight_nr)])
    this_flight["start_time"] = start_time
    return this_flight