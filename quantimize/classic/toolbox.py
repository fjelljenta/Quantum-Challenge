import quantimize.converter as cv
import quantimize.data_access as da
import numpy as np

'atmo_data[latitude][longitude][flight level][time][merged]'
def straight_line_solution(flight_nr, dt):
    """
    :param flight_nr: Flight number
    :param dt: The time step in minutes
    :return: Trajectory in the format of a list of tuples (time, longitude, latitude)
    """
    info = da.get_flight_info(flight_nr)
    c = info['start_longitudinal'], info['start_latitudinal'], info['end_longitudinal'], info['end_latitudinal']
    slope = (c[3]-c[1])*111/((c[2]-c[0])*85)
    flight_level = info['start_flightlevel']
    speed = cv.ms_to_kmm(cv.kts_to_ms(da.get_flight_level_data(flight_level)['CRUISE']['TAS']))
    total_distance = cv.coordinates_to_distance(*c)
    current_coord = c[0], c[1], flight_level, info['start_time']
    trajectory = [current_coord]
    current_distance = 0
    while current_distance < total_distance:
        current_distance += speed*dt
        time = cv.update_time(current_coord[3], dt)
        longitude = current_coord[0] + speed * dt * np.cos(np.arctan(slope)) / 85
        latitude = current_coord[1] + speed * dt * np.sin(np.arctan(slope)) / 111
        current_coord = longitude, latitude, flight_level, time
        trajectory.append(current_coord)
    trajectory[-1] = (c[2], c[3], flight_level, current_coord[3])
    return trajectory

def compute_cost(trajectory, dt):
    dCCO2 = 6.94
    cost = 0
    for coordinate in trajectory:
        cost += (da.get_merged_atmo_data(*coordinate)+dCCO2) * da.get_flight_level_data(coordinate[2])['CRUISE']['fuel'] * dt
    return cost
