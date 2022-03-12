import quantimize.converter as cv
import quantimize.data_access as da

def check_height(fl_1, fl_2):
    """Compares the flight level hight

    Args:
        fl_1 (int): Flightlevel
        fl_2 (int): Flightlevel

    Returns:
        boolean: True for enough distance, false for to close
    """
    diff = abs(int(fl_1)-int(fl_2))
    if diff >= 10:
        return True
    else:
        return False

def check_horizontal_distance(long_1, lat_1, long_2, lat_2):
    dist = cv.coordinates_to_distance(long_1, lat_1, long_2, lat_2)
    if dist > 9.26:
        return True
    else:
        return False

def check_position_safety(flight_1, flight_2):
    if check_horizontal_distance(flight_1[0],flight_1[1], flight_2[0],flight_2[1]) or check_height(flight_1[2], flight_2[2]):
        return True
    else:
        return False

def check_safety(list_of_trajectories, dt):
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
