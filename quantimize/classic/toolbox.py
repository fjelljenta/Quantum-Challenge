from quantimize.data_access import *
from quantimize.converter import *
import numpy as np

'atmo_data[latitude][longitude][flight level][time][merged]'
def straight_line_solution(flight_nr, dt):
    """
    :param flight_nr: Flight number
    :param dt: The time step in minutes
    :return: Trajectory in the format of a list of tuples (time, longitude, latitude)
    """
    info = get_flight_info(flight_nr)
    c = info['start_longitudinal'], info['start_latitudinal'], info['end_longitudinal'], info['end_latitudinal']
    slope = (c[3]-c[1])*111/((c[2]-c[0])*85)
    flight_level = info['start_flightlevel']
    speed = ms_to_kmm(kts_to_ms(get_flight_level_data(flight_level)['CRUISE']['TAS']))
    total_distance = coordinates_to_distance(*c)
    current_coord = c[0], c[1], flight_level, info['start_time']
    trajectory = [current_coord]
    current_distance = 0
    while current_distance < total_distance:
        current_distance += speed*dt
        time = update_time(current_coord[3], dt)
        longitude = current_coord[0] + speed * dt * np.cos(np.arctan(slope)) / 85
        latitude = current_coord[1] + speed * dt * np.sin(np.arctan(slope)) / 111
        current_coord = longitude, latitude, flight_level, time
        trajectory.append(current_coord)
        print(current_distance)
    trajectory[-1] = (c[2], c[3], flight_level, current_coord[3])
    return trajectory