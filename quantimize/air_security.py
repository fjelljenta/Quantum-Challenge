from tabnanny import check
import quantimize.converter as cv
import quantimize.data_access as da
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


def check_height(fl_1, fl_2):
    """Compares the flight level height

    Args:
        fl_1 (int/float/str): Flight level
        fl_2 (int/float/str): Flight level

    Returns:
        boolean: True for enough distance, false for to close
    """
    diff = abs(int(fl_1)-int(fl_2))
    if diff >= 10:
        return True
    else:
        return False


def check_horizontal_distance(long_1, lat_1, long_2, lat_2):
    """Checks the horizontal distance between two flights

    Args:
        long_1 (int/float/str): Longitudinal coordinate of flight 1
        lat_1 (int/float/str): Latitudinal coordinate of flight 1
        long_2 (int/float/str): Longitudinal coordinate of flight 2
        lat_2 (int/float/str): Latitudinal coordinate of flight 2

    Returns:
        boolean: True for enough distance, false for to close
    """
    dist = cv.coordinates_to_distance(long_1, lat_1, long_2, lat_2)
    if dist > 9.26:
        return True
    else:
        return False


def check_position_safety(flight_1, flight_2):
    """Checks if the position (horizontal and vertical) between two flights is safe

    Args:
        flight_1 (dic): information about flight 1
        flight_2 (dic): information about flight 2

    Returns:
        boolean: True for enough distance, false for to close
    """
    if check_horizontal_distance(flight_1[0], flight_1[1], flight_2[0], flight_2[1]) or check_height(flight_1[2], flight_2[2]):
        return True
    else:
        return False


def check_safety(list_of_trajectories, dt):
    """ Checks if the safety regulations are met

    Args:
        list_of_trajectories (list of trajectory dicts): list of flight trajectory dicts
        dt (int): time step

    Returns:
        error_list (list): list consisting of the safety violations

    """
    time_list, time_grid = da.create_time_grid(dt)
    time_grid = da.map_trajectory_to_time_grid(list_of_trajectories, time_grid)
    error_list = []
    for time_step in time_list:
        long_step = time_grid[time_step]["LONG"]
        lat_step = time_grid[time_step]["LAT"]
        fl_step = time_grid[time_step]["FL"]
        for i in range(len(long_step)):
            for j in range(i+1, len(long_step), 1):
                flight_1 = (long_step[i], lat_step[i],
                            fl_step[i], time_grid[time_step]["FL_NR"][i])
                flight_2 = (long_step[j], lat_step[j],
                            fl_step[j], time_grid[time_step]["FL_NR"][j])
                if not check_position_safety(flight_1, flight_2):
                    error_list.append((time_step, flight_1, flight_2))
    return error_list


def distance_between_trajectories_at_time(traj1, traj2, t):
    """ Returns distance between two given flights at a given time

    Args:
        traj1 (list): first flight trajectory
        traj2 (list): second flight trajectory
        t (int): time of day expressed in seconds

    Returns:
        (float, float): distance between flights with respect to x-y plane, flight level difference between flights

    """

    # We find the index of the latest coordinate earlier than the given time

    lower = 0
    higher = len(traj1)-1
    index = 0

    while True:
        index = (lower+higher)//2
        if cv.datetime_to_seconds(traj1[index][3]) < t:
            if cv.datetime_to_seconds(traj1[index+1][3]) > t:
                break
            lower = max(int(index), lower+1)
        elif cv.datetime_to_seconds(traj1[index][3]) > t:
            if cv.datetime_to_seconds(traj1[index-1][3]) < t:
                index -= 1
                break
            higher = min(int(index)+1, higher-1)
        else:
            break

    # Assuming the true trajectory is linear between two consecutive points
    # in our list, we can calculate the position of the airplane accordingly

    x1 = ((cv.datetime_to_seconds(traj1[index+1][3])-t)/(cv.datetime_to_seconds(traj1[index+1][3])-cv.datetime_to_seconds(traj1[index][3])))*traj1[int(index)][0] + \
        ((t-cv.datetime_to_seconds(traj1[index][3]))/(cv.datetime_to_seconds(
            traj1[index+1][3])-cv.datetime_to_seconds(traj1[index][3])))*traj1[int(index)+1][0]
    y1 = ((cv.datetime_to_seconds(traj1[index+1][3])-t)/(cv.datetime_to_seconds(traj1[index+1][3])-cv.datetime_to_seconds(traj1[index][3])))*traj1[int(index)][1] + \
        ((t-cv.datetime_to_seconds(traj1[index][3]))/(cv.datetime_to_seconds(
            traj1[index+1][3])-cv.datetime_to_seconds(traj1[index][3])))*traj1[int(index)+1][1]
    z1 = ((cv.datetime_to_seconds(traj1[index+1][3])-t)/(cv.datetime_to_seconds(traj1[index+1][3])-cv.datetime_to_seconds(traj1[index][3])))*traj1[int(index)][2] + \
        ((t-cv.datetime_to_seconds(traj1[index][3]))/(cv.datetime_to_seconds(
            traj1[index+1][3])-cv.datetime_to_seconds(traj1[index][3])))*traj1[int(index)+1][2]

    while True:
        index = (lower+higher)//2
        if cv.datetime_to_seconds(traj2[index][3]) < t:
            if cv.datetime_to_seconds(traj2[index+1][3]) > t:
                break
            lower = max(int(index), lower+1)
        elif cv.datetime_to_seconds(traj2[index][3]) > t:
            if cv.datetime_to_seconds(traj2[index-1][3]) < t:
                index -= 1
                break
            higher = min(int(index)+1, higher-1)
        else:
            break

    x2 = ((cv.datetime_to_seconds(traj2[index+1][3])-t)/(cv.datetime_to_seconds(traj2[index+1][3])-cv.datetime_to_seconds(traj2[index][3])))*traj2[int(index)][0] + \
        ((t-cv.datetime_to_seconds(traj2[index][3]))/(cv.datetime_to_seconds(
            traj2[index+1][3])-cv.datetime_to_seconds(traj2[index][3])))*traj2[int(index)+1][0]
    y2 = ((cv.datetime_to_seconds(traj2[index+1][3])-t)/(cv.datetime_to_seconds(traj2[index+1][3])-cv.datetime_to_seconds(traj2[index][3])))*traj2[int(index)][1] + \
        ((t-cv.datetime_to_seconds(traj2[index][3]))/(cv.datetime_to_seconds(
            traj2[index+1][3])-cv.datetime_to_seconds(traj2[index][3])))*traj2[int(index)+1][1]
    z2 = ((cv.datetime_to_seconds(traj2[index+1][3])-t)/(cv.datetime_to_seconds(traj2[index+1][3])-cv.datetime_to_seconds(traj2[index][3])))*traj2[int(index)][2] + \
        ((t-cv.datetime_to_seconds(traj2[index][3]))/(cv.datetime_to_seconds(
            traj2[index+1][3])-cv.datetime_to_seconds(traj2[index][3])))*traj2[int(index)+1][2]

    # print((x1,y1,z1), (x2,y2,z2))

    return cv.coordinates_to_distance(x1, y1, x2, y2), abs(z2-z1)


def check_safety_2(list_of_trajectory_dicts):
    """ Checks if the safety regulations are met

    Args:
        list_of_trajectory_dicts (list of trajectory dicts): list of flight trajectory dicts
        dt (int): time step

    Returns:
        error_list (list): list consisting of the safety violations

    """
    error_list = []
    for i in range(len(list_of_trajectory_dicts)):
        for j in range(i+1, len(list_of_trajectory_dicts)):

            traj1 = list_of_trajectory_dicts[i]["trajectory"]
            traj2 = list_of_trajectory_dicts[j]["trajectory"]

            # Here we define the relevant time interval when both airplanes are flying

            start_time = traj1[0][3] if traj1[0][3] > traj2[0][3] \
                else traj2[0][3]  # Start times in data are all within the 6 to 8 AM range. No flights just before midnight
            end_time = traj1[-1][3] if traj1[-1][3] < traj2[0][3] \
                else traj2[-1][3]

            max_speed_xy = cv.ms_to_kms(cv.kts_to_ms(459))  # km/s
            max_speed_z = 3830/(60*100)  # (100 ft)/s or flight_level/s
            unsafe_radius = 9.26
            unsafe_height = 10

            t = cv.datetime_to_seconds(start_time)
            current_distance_xy, current_distance_z = distance_between_trajectories_at_time(
                traj1, traj2, t)

            while t < cv.datetime_to_seconds(end_time):
                if current_distance_xy < unsafe_radius:
                    if current_distance_z < unsafe_height:
                        error_list += [(list_of_trajectory_dicts[i]["flight_nr"],
                                        list_of_trajectory_dicts[j]["flight_nr"])]
                        break
                    else:
                        # The minimum time it would take for both planes to get dangerously close along the z axis
                        t += max(1, (current_distance_z -
                                 unsafe_height)/(2*max_speed_z))
                        current_distance_xy, current_distance_z = distance_between_trajectories_at_time(
                            traj1, traj2, t)
                else:
                    # The minimum time it would take for both planes to be dangerously close along the xy plane
                    t += max(1, (current_distance_xy-unsafe_radius) /
                             (2*max_speed_xy))

    return error_list


def radius_control(trajectory):
    """Returns true or false depending on if the turning angle was above 25 degree or below

    Args:
        trajectory (dict/list): List of trajectory points

    Returns:
        Boolean: True for angle smaller than 25 degree, false for angle greater 25 degree
    """
    flag = False
    flight_path = cv.check_trajectory_dict(trajectory)
    for i in range(len(flight_path)-2):
        v1 = np.array([flight_path[i][0]-flight_path[i+1][0],
                      flight_path[i][1]-flight_path[i+1][1]])
        v2 = np.array([flight_path[i+1][0]-flight_path[i+2][0],
                      flight_path[i+1][1]-flight_path[i+2][1]])
        dotproduct= np.dot(v1 / np.linalg.norm(v1), v2/np.linalg.norm(v2))
        if dotproduct > 1:
            theta = 0.0
        elif dotproduct < -1:
            theta = 180.0
        else:
            theta = np.arccos(dotproduct) * 180 / np.pi
        if theta > 25:
            flag = True
    if flag:
        return False
    else:
        return True


def safe_algorithm(flights, algorithm, **kwargs):
    """ Calculates the trajectory for a list of flight numbers and checks the trajectories for radius boundaries and possible violations for the air safety. On errors, the trajectories are re-calculated.

    Args:
        flights (list): list of flight numbers to check
        algorithm (trajectory algorithm): algorithm to calculate the trajectory

    Returns:
        list: list of corrected flight trajectories
    """
    dt = kwargs.get("dt", 15)
    trajectories = []

    radius_check_needed=True
    radius_runs = 0
    radius_errors = []
    # calculate flights and check radius
    for flight in tqdm(flights):
        while radius_check_needed:
            if radius_runs > 5:
                radius_errors.append(flight)
                break
            next_trajectory = algorithm(flight, dt, only_trajectory_dict=True, timed_trajecotory=False)
            radius_check_needed = not radius_control(next_trajectory)
            radius_runs = radius_runs+1
        radius_runs = 0
        radius_check_needed = True

        trajectories.append(next_trajectory)

    crash_check_needed = True
    old_compute_again_len = len(flights)
    same_correction_count = 0
    print("Begin collision check")

    while crash_check_needed:

        # todo: check radius

        safety_errors=check_safety(trajectories, dt)
        safety_errors_flightnumbers= [[safety_errors[i][1][-1],safety_errors[i][2][-1]]
                                      for i in range(len(safety_errors))]

        crashing_flights = []
        for crash in safety_errors_flightnumbers:
            if (not crash in crashing_flights) and (not [crash[1], crash[0]] in crashing_flights):
                crashing_flights.append(crash)

        crash_dict = {}
        for crash in crashing_flights:
            try:
                crash_dict[str(crash[0])].append(crash[1])
            except:
                crash_dict[str(crash[0])] = [crash[1]]
            try:
                crash_dict[str(crash[1])].append(crash[0])
            except:
                crash_dict[str(crash[1])] = [crash[0]]

        compute_again = []
        while crash_dict != {}:

            # find key with longest list of crashings
            longest_key_value = 0
            for key, value in crash_dict.items():
                if len(value) > longest_key_value:
                    longest_key_value = len(value)
                    longest_key = key
            longest_key_int = int(longest_key)

            # save in compute_again list
            compute_again.append(longest_key_int)

            del_keys = [longest_key]

            # delete this number from the values of all other keys
            for key, value in crash_dict.items():
                if longest_key_int in value:
                    value.remove(longest_key_int)
                    if len(value) == 0:
                        del_keys.append(key)

            for key in del_keys:
                del crash_dict[key]

        print('compute_again: ', compute_again)

        for flight in compute_again:
            if flight in radius_errors:
                radius_errors.remove(flight)

            while radius_check_needed:
                if radius_runs > 5:
                    radius_errors.append(flight)
                    break
                next_trajectory = algorithm(flight, dt, only_trajectory_dict=True, timed_trajecotory=False)
                radius_check_needed = not radius_control(next_trajectory)
                radius_runs = radius_runs + 1
            radius_runs = 0
            radius_check_needed = True

            trajectories[flights.index(flight)] = next_trajectory

        crash_check_needed = len(compute_again)

        #break for too many rounds
        if (old_compute_again_len == len(compute_again)):
            same_correction_count = same_correction_count +1
            if same_correction_count > 3:
                print("flight(s) ", compute_again, " lead to non-correctable crashings, ",
                                                   "change starting time or starting level of these flights")
                if len(radius_errors):
                    print("flight(s) ", radius_errors, " have non-correctable too small curve-radius, ",
                                                   "change starting time or starting level of these flights")
                return trajectories
        else:
            same_correction_count = 0
            old_compute_again_len = len(compute_again)

    if len(radius_errors):
        print("flight(s) ", radius_errors, " have non-correctable too small curve-radius, ",
              "change starting time or starting level of these flights")
        return trajectories

    print("air security protocol suceeded")
    return trajectories


def radius_check_for_flight(flight_number, algorithm, dt, run):
    """ Calculates the trajectory for flight_number and the given algorithm and evaluates the radius requirements.

    Args:
        flight_number (int): flight number
        algorithm (trajectory algorithm): algorithm to calculate the trajectory
        dt (int): time step for the calculation
        run (int): number of already checked radii

    Returns:
        dict: trajcetory dict with valid radius
    """
    trajectory = algorithm(flight_number, dt)
    if not radius_control(trajectory) and run<5:
        print("Flight",flight_number,"has radius issues")
        trajectory = radius_check_for_flight(flight_number, algorithm, dt, run+1)
    if run == 5:
        print("Flight",flight_number,"was uncorrectable")
    return trajectory


def list_conflicts(list_of_conflicts):
    """ Reformats the list_of_conflicts to show which flights collide with which

    Args:
        list_of_conflicts (list): list with list of timestamp, flight1, flight2

    Returns:
        list: list of flight numbers that have collision problems
    """
    pairs = []
    for conflict in list_of_conflicts:
        pairs.append((conflict[1][-1],conflict[2][-1]))
    pairs = set(pairs)
    conflicts = {}
    for pair in pairs:
        try:
            conflicts[pair[0]].append(pair[1])
        except:
            conflicts[pair[0]] = [pair[1]]
        try:
            conflicts[pair[1]].append(pair[0])
        except:
            conflicts[pair[1]] = [pair[0]]
    for conflict in conflicts:
        conflicts[conflict] = list(set(conflicts[conflict]))
    #print(conflicts)
    return sorted(conflicts, key=lambda k: len(conflicts[k]), reverse=True)


def safe_algorithm_2(list_of_flights, algorithm, dt=15, check_max=10):
    """ Alternative way to check air security, using multiprocessing for speedup. Calulates the valid trajectories, regarding radius and then re calculates air collisions

    Args:
        list_of_flights (list): list of flight numbers
        algorithm (trajectory algorithm): algorithm used to calculate the trajectories

    Returns:
        list, list: list of checked trajectories, list of remaining conflicts
    """
    prep_list = []
    check_run = 0
    for flight in list_of_flights:
        prep_list.append((flight,algorithm,dt,0))
    with Pool() as p:
        trajectories = p.starmap(radius_check_for_flight, prep_list)
    print("Finished trajectory calculation")
    
    while check_run < check_max:
        print("Running check:",check_run)
        safety_errors = check_safety(trajectories, dt)
        conflicts = list_conflicts(safety_errors)
        #print(conflicts)
        if len(conflicts) == 0:
            break
        prep_list = []
        for conflict in conflicts:
            prep_list.append((conflict, algorithm, dt, 0))
        with Pool() as p:
            corrected_trajectories = p.starmap(radius_check_for_flight, prep_list)
        for i, traj in zip(conflicts, corrected_trajectories):
            trajectories[list_of_flights.index(i)] = traj
        check_run+=1

    return trajectories, conflicts