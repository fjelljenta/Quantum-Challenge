import quantimize.converter as cv
import quantimize.data_access as da
import numpy as np

'atmo_data[latitude][longitude][flight level][time][merged]'
def straight_line_solution(flight_nr, dt):
    """
    :param flight_nr: Flight number
    :param dt: The time step in minutes
    :return: Trajectory in the format of a list of tuples (time, longitude, latitude) embeded in a dict with flightnumber
    """
    info = da.get_flight_info(flight_nr)
    c = info['start_longitudinal'], info['start_latitudinal'], info['end_longitudinal'], info['end_latitudinal']
    slope = (c[3]-c[1])*111/((c[2]-c[0])*85)
    flight_level = info['start_flightlevel']
    speed = cv.ms_to_kms(cv.kts_to_ms(da.get_flight_level_data(flight_level)['CRUISE']['TAS']))
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
    return {"flight_nr":flight_nr, "trajectory":trajectory}

def compute_cost(trajectory, dt):
    """Returns the cost for a given list of trajectories in 10**-12

    Args:
        trajectory (list of trajectory dicts): List of trajectory dicts
        dt (int): Timestep in seconds

    Returns:
        float, int: cost in 10**-12 K and flight number
    """
    #dCCO2 = 6.94 Should be already in the merged data (see data PDF)
    cost = 0
    start_level = trajectory["trajectory"][0][2]
    for coordinate in trajectory["trajectory"]:
        if coordinate[2] == start_level:
            cost += (da.get_merged_atmo_data(*coordinate)) * da.get_flight_level_data(coordinate[2])['CRUISE']['fuel'] * dt/60
        elif coordinate[2] < start_level:
            cost += (da.get_merged_atmo_data(*coordinate)) * da.get_flight_level_data(coordinate[2])['DESCENT']['fuel'] * dt/60
            start_level = coordinate[2]
        elif coordinate[2] > start_level:
            cost += (da.get_merged_atmo_data(*coordinate)) * da.get_flight_level_data(coordinate[2])['CLIMB']['fuel'] * dt/60
            start_level = coordinate[2]
    return cost, trajectory["flight_nr"]
