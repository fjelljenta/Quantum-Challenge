from quantimize.data_access import *
from quantimize.converter import *
import numpy as np

'atmo_data[latitude][longitude][flight level][time][merged]'
def straight_line_solution(flight_nr, dt):
    info = get_flight_info(flight_nr)
    c = info['start_longitudinal'], info['start_latitudinal'], info['end_longitudinal'], info['end_latitudinal']
    slope = (c[3]-c[1])/(c[2]-c[0])
    flight_level = info['start_flightlevel']
    speed = ms_to_kmm(kts_to_ms(get_flight_level_data(flight_level)['CRUISE']['TAS']))
    total_distance = coordinates_to_distance(*c)
    current_coord = info['start_time'], c[0], c[1]
    trajectory = [current_coord]
    current_distance = 0
    while current_distance < total_distance:
        current_distance += speed*dt
        time = update_time(current_coord[0], dt)
        longitude = current_coord[1] + speed * dt * np.cos(np.arctan(slope)) / 85
        latitude = current_coord[2] + speed * dt * np.sin(np.arctan(slope)) / 111
        current_coord = time, longitude, latitude
        trajectory.append(current_coord)
    trajectory[-1] = (current_coord[0], c[2], c[3])
    return trajectory