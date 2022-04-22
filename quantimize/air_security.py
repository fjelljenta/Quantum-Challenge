import quantimize.converter as cv
import quantimize.data_access as da
import numpy as np
# from quantimize.classic.toolbox import straight_line_solution

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
    if check_horizontal_distance(flight_1[0],flight_1[1], flight_2[0],flight_2[1]) or check_height(flight_1[2], flight_2[2]):
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
            for j in range(i+1,len(long_step),1):
                flight_1 = (long_step[i], lat_step[i], fl_step[i], time_grid[time_step]["FL_NR"][i])
                flight_2 = (long_step[j], lat_step[j], fl_step[j], time_grid[time_step]["FL_NR"][j])
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

    lower=0
    higher=len(traj1)-1
    index=0

    while True:
        index=(lower+higher)//2
        if cv.datetime_to_seconds(traj1[index][3])<t:
            if cv.datetime_to_seconds(traj1[index+1][3])>t:
                break
            lower=max(int(index), lower+1)
        elif cv.datetime_to_seconds(traj1[index][3])>t:
            if cv.datetime_to_seconds(traj1[index-1][3])<t:
                index-=1
                break
            higher=min(int(index)+1, higher-1)
        else:
            break

    # Assuming the true trajectory is linear between two consecutive points
    # in our list, we can calculate the position of the airplane accordingly

    x1=((cv.datetime_to_seconds(traj1[index+1][3])-t)/(cv.datetime_to_seconds(traj1[index+1][3])-cv.datetime_to_seconds(traj1[index][3])))*traj1[int(index)][0] + \
    ((t-cv.datetime_to_seconds(traj1[index][3]))/(cv.datetime_to_seconds(traj1[index+1][3])-cv.datetime_to_seconds(traj1[index][3])))*traj1[int(index)+1][0]
    y1=((cv.datetime_to_seconds(traj1[index+1][3])-t)/(cv.datetime_to_seconds(traj1[index+1][3])-cv.datetime_to_seconds(traj1[index][3])))*traj1[int(index)][1] + \
    ((t-cv.datetime_to_seconds(traj1[index][3]))/(cv.datetime_to_seconds(traj1[index+1][3])-cv.datetime_to_seconds(traj1[index][3])))*traj1[int(index)+1][1]
    z1=((cv.datetime_to_seconds(traj1[index+1][3])-t)/(cv.datetime_to_seconds(traj1[index+1][3])-cv.datetime_to_seconds(traj1[index][3])))*traj1[int(index)][2] + \
    ((t-cv.datetime_to_seconds(traj1[index][3]))/(cv.datetime_to_seconds(traj1[index+1][3])-cv.datetime_to_seconds(traj1[index][3])))*traj1[int(index)+1][2]

    while True:
        index=(lower+higher)//2
        if cv.datetime_to_seconds(traj2[index][3])<t:
            if cv.datetime_to_seconds(traj2[index+1][3])>t:
                break
            lower=max(int(index), lower+1)
        elif cv.datetime_to_seconds(traj2[index][3])>t:
            if cv.datetime_to_seconds(traj2[index-1][3])<t:
                index-=1
                break
            higher=min(int(index)+1, higher-1)
        else:
            break

    x2=((cv.datetime_to_seconds(traj2[index+1][3])-t)/(cv.datetime_to_seconds(traj2[index+1][3])-cv.datetime_to_seconds(traj2[index][3])))*traj2[int(index)][0] + \
    ((t-cv.datetime_to_seconds(traj2[index][3]))/(cv.datetime_to_seconds(traj2[index+1][3])-cv.datetime_to_seconds(traj2[index][3])))*traj2[int(index)+1][0]
    y2=((cv.datetime_to_seconds(traj2[index+1][3])-t)/(cv.datetime_to_seconds(traj2[index+1][3])-cv.datetime_to_seconds(traj2[index][3])))*traj2[int(index)][1] + \
    ((t-cv.datetime_to_seconds(traj2[index][3]))/(cv.datetime_to_seconds(traj2[index+1][3])-cv.datetime_to_seconds(traj2[index][3])))*traj2[int(index)+1][1]
    z2=((cv.datetime_to_seconds(traj2[index+1][3])-t)/(cv.datetime_to_seconds(traj2[index+1][3])-cv.datetime_to_seconds(traj2[index][3])))*traj2[int(index)][2] + \
    ((t-cv.datetime_to_seconds(traj2[index][3]))/(cv.datetime_to_seconds(traj2[index+1][3])-cv.datetime_to_seconds(traj2[index][3])))*traj2[int(index)+1][2]

    # print((x1,y1,z1), (x2,y2,z2))

    return cv.coordinates_to_distance(x1,y1,x2,y2), abs(z2-z1)


def check_safety_2(list_of_trajectory_dicts):
    """ Checks if the safety regulations are met

    Args:
        list_of_trajectory_dicts (list of trajectory dicts): list of flight trajectory dicts
        dt (int): time step

    Returns:
        error_list (list): list consisting of the safety violations

    """
    error_list=[]
    for i in range(len(list_of_trajectory_dicts)):
        for j in range(i+1, len(list_of_trajectory_dicts)):

            traj1=list_of_trajectory_dicts[i]["trajectory"]
            traj2=list_of_trajectory_dicts[j]["trajectory"]

            # Here we define the relevant time interval when both airplanes are flying

            start_time=traj1[0][3] if traj1[0][3]>traj2[0][3] \
            else traj2[0][3]  # Start times in data are all within the 6 to 8 AM range. No flights just before midnight
            end_time=traj1[-1][3] if traj1[-1][3]<traj2[0][3] \
            else traj2[-1][3]

            max_speed_xy=cv.ms_to_kms(cv.kts_to_ms(459)) # km/s
            max_speed_z=3830/(60*100) # (100 ft)/s or flight_level/s
            unsafe_radius=9.26
            unsafe_height=10

            t=cv.datetime_to_seconds(start_time)
            current_distance_xy, current_distance_z = distance_between_trajectories_at_time(traj1, traj2, t)

            while t<cv.datetime_to_seconds(end_time):
                if current_distance_xy<unsafe_radius:
                    if current_distance_z<unsafe_height:
                        error_list+=[(list_of_trajectory_dicts[i]["flight_nr"],list_of_trajectory_dicts[j]["flight_nr"])]
                        break
                    else:
                        t+=max(1,(current_distance_z-unsafe_height)/(2*max_speed_z)) # The minimum time it would take for both planes to get dangerously close along the z axis
                        current_distance_xy, current_distance_z = distance_between_trajectories_at_time(traj1, traj2, t)
                else:
                    t+=max(1,(current_distance_xy-unsafe_radius)/(2*max_speed_xy)) # The minimum time it would take for both planes to be dangerously close along the xy plane

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
        v1 = np.array([flight_path[i][0]-flight_path[i+1][0],flight_path[i][1]-flight_path[i+1][1]])
        v2 = np.array([flight_path[i+1][0]-flight_path[i+2][0],flight_path[i+1][1]-flight_path[i+2][1]])
        #theta = np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))*180/np.pi
        theta = np.arccos(np.dot(v1/ np.linalg.norm(v1), v2/np.linalg.norm(v2))) * 180 / np.pi
        if theta > 25:
            flag = True
    if flag:
        return False
    else:
        return True
