import quantimize.converter as cv
import quantimize.data_access as da
import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt


def compute_cost(trajectory):
    """Returns the cost for a given list of trajectories in 10**-12

    Args:
        trajectory (list of trajectory dicts): List of trajectory dicts

    Returns:
        float, int: cost in 10**-12 K and flight number
    """
    cost = 0
    flight_path = cv.check_trajectory_dict(trajectory)
    t = cv.datetime_to_seconds(flight_path[0][3])
    start_level = flight_path[0][2]
    # we compare the flight levels between neighbors to know in which mode (climb/cruise/descend) the plane is flying
    for coordinate in flight_path[1:]:
        # incremental cost equals the time change (in min) times fuel consumpution rate (in kg / min ) times
        # change in temperature per portion of fuel (in K / kg)
        if coordinate[2] == start_level:
            cost += (da.get_merged_atmo_data(*coordinate)) * da.get_flight_level_data(coordinate[2])['CRUISE']['fuel'] \
                    * (cv.datetime_to_seconds(coordinate[3]) - t) / 60
        else:
            dt1 = compute_time_to_reach_other_flightlevel(start_level, coordinate[2])
            if coordinate[2] < start_level:
                cost += (da.get_merged_atmo_data(*coordinate)) * \
                        da.get_flight_level_data(coordinate[2])['DESCENT']['fuel'] \
                        * dt1 / 60 + \
                        (da.get_merged_atmo_data(*coordinate)) * \
                        da.get_flight_level_data(coordinate[2])['CRUISE']['fuel'] \
                        * (cv.datetime_to_seconds(coordinate[3]) - t - dt1) / 60
                # print(dt1)
            else:
                cost += (da.get_merged_atmo_data(*coordinate)) * \
                        da.get_flight_level_data(coordinate[2])['CLIMB']['fuel'] \
                        * dt1 / 60 + \
                        (da.get_merged_atmo_data(*coordinate)) * \
                        da.get_flight_level_data(coordinate[2])['CRUISE']['fuel'] \
                        * (cv.datetime_to_seconds(coordinate[3]) - t - dt1) / 60
                # print(dt1)
            start_level = coordinate[2]
        t = cv.datetime_to_seconds(coordinate[3])
    return cost


def timed_trajectory(trajectory, dt):
    """Returns a trajectory for given time steps,
    end_time might be excluded if starttime-endtime is no multiple of dt.

        Args:
            trajectory (list): List of trajectory points with constant dx
            dt (int): Timestep in seconds

        Returns:
            (list): List of trajectory points with constant dt
    """
    flight_path = cv.check_trajectory_dict(trajectory)
    start_time = cv.datetime_to_seconds(flight_path[0][3])
    end_time = cv.datetime_to_seconds(flight_path[-1][3])
    time_trajectory = []
    for t in np.arange(start_time, end_time, dt):
        time_trajectory.append(trajectory_at_time(flight_path, cv.seconds_to_datetime(t)))

    return time_trajectory


def trajectory_at_time(trajectory, datetime):
    """Returns one trajectory entry (x,y and z position) which approximates the position for a given time linearly,
        if time is out of bounds of flight time, an empty array is returned

        Args:
            trajectory (list): List of trajectory points with constant dx
            datetime (int): Time in seconds

        Returns:
            (list): trajectory point at wished time
        """
    time = cv.datetime_to_seconds(datetime)
    length = len(trajectory)
    timelist = [cv.datetime_to_seconds(trajectory[k][3]) for k in range(length)]
    if time < timelist[0] or time > timelist[-1]:
        return []
    abs_time_diff_list = np.abs(timelist - time * np.ones(length))
    index_min = np.argmin(abs_time_diff_list)
    trajectorypoint = trajectory[index_min]
    if timelist[index_min] <= time:
        multiplicator = (time - timelist[index_min]) / (timelist[index_min + 1] - timelist[index_min])
        trajectorypoint = [trajectorypoint[k] + multiplicator * (trajectory[index_min + 1][k] - trajectorypoint[k]) for
                           k in range(3)]
    else:
        multiplicator = (timelist[index_min] - time) / (timelist[index_min] - timelist[index_min - 1])
        trajectorypoint = [trajectorypoint[k] - multiplicator * (trajectorypoint[k] - trajectory[index_min - 1][k]) for
                           k in range(3)]
    trajectorypoint.append(datetime)
    return trajectorypoint


def straight_line_solution(flight_nr, dt):
    """Returns the straight line solution for a given flight

    Args:
        flight_nr: Flight number
        dt: The time step in seconds

    Returns:
        Trajectory in the format of a list of tuples (time, longitude, latitude) embeded in a dict with flightnumber
    """
    trajectory = straight_line_trajectory(flight_nr, dt)
    return {"flight_nr": flight_nr, "trajectory": trajectory}


def straight_line_trajectory(flight_nr, dt):
    """Computes and returns the straight line solution for a given flight
       The straight line solution assumes the same flight level as starting point for the ending point
    Args:
        flight_nr: Flight number
        dt: The time step in seconds

    Returns:
        Trajectory in the format of a list of tuples (time, longitude, latitude)
    """
    info = da.get_flight_info(flight_nr)
    trajectory = straight_line_trajectory_core(info['start_longitudinal'], info['end_longitudinal'],
                                               info['start_latitudinal'], info['end_latitudinal'],
                                               info['start_flightlevel'], info['start_time'], dt)
    trajectory = timed_trajectory(trajectory, dt)
    return trajectory


def straight_line_trajectory_core(start_longitudinal, end_longitudinal, start_latitudinal, end_latitudinal,
                                  start_flightlevel, start_time, dt):
    slope = ((end_latitudinal - start_latitudinal) * 111) / \
            ((end_longitudinal - start_longitudinal) * 85)
    # compute the slope of the straight line connecting start and end points in x-y plane
    speed = cv.ms_to_kms(cv.kts_to_ms(da.get_flight_level_data(start_flightlevel)['CRUISE']['TAS']))
    # The speed is remaining the same throughout the trajectory as we stay on the same flight level
    total_distance = cv.coordinates_to_distance(start_longitudinal, start_latitudinal,
                                                end_longitudinal, end_latitudinal)
    current_coord = start_longitudinal, start_latitudinal, start_flightlevel, start_time
    trajectory = [current_coord]
    current_displacement = 0
    while current_displacement < total_distance:
        # We keep record of the distance travelled so that we know when to stop
        current_displacement += speed * dt
        time = cv.update_time(current_coord[3], dt)
        longitude = current_coord[0] + speed * dt * np.cos(np.arctan(slope)) / 85
        latitude = current_coord[1] + speed * dt * np.sin(np.arctan(slope)) / 111
        current_coord = longitude, latitude, start_flightlevel, time
        trajectory.append(current_coord)
    # The last coordinate is in general beyond the ending point as we are taking a constant time step, so we need to
    # correct it by replacing with the desired ending point.
    total_time = int(total_distance / speed)
    end_time = cv.update_time(start_time, total_time)
    trajectory[-1] = (end_longitudinal, end_latitudinal, start_flightlevel, end_time)
    return trajectory


def curve_3D_solution(flight_nr, ctrl_pts):
    """Returns the curved 3D solution for a given flight number

    Args:
    flight_nr (int): flight number

    Returns:
    dictionary with flight numbers and corresponding trajectory

    """
    trajectory = curve_3D_trajectory(flight_nr, ctrl_pts)
    corrected_trajectory = correct_for_boundaries(trajectory)
    return {"flight_nr": flight_nr, "trajectory": corrected_trajectory}


def curve_3D_trajectory(flight_nr, ctrl_pts):
    """
    The curve 3D curve is constructed by multiplying two spline curves, spline_xy and spline_z together.
    Spline_xy curve is parametrized by 5 control points in the x-y plane. The first and last points are fixed, as they
    are the starting and ending points of the trajectory. 3 control points ( 3 x-coordinates + 3 y-coordinates =
    6 paramters) are variable.
    Spline_z curve is parametrized by 6 control points in the vertical plane, spanned by the z_axis and the vector along
    straight_line trajectory. The first control point is fixed as it;s the starting point of the trajectory.
    The coordinates of the vector along straight_line trajectory is also fixed and set to be evenly separated. The
    z-coordinates (flight level) of the remaining 5 control points are variable.

    Args:
    flight_nr (int): flight number
    ctrl_pts: numpy array containing control points

    Returns:
    trajectory (dict): dictionary containing the flights and the corresponding trajectories

    """
    ctrl_pts = list(ctrl_pts)
    info = da.get_flight_info(flight_nr)
    x = [info['start_longitudinal']] + ctrl_pts[:3] + [info['end_longitudinal']]
    # The x-coordinates for the x-y spline curve
    y = [info['start_latitudinal']] + ctrl_pts[3:6] + [info['end_latitudinal']]
    # The y-coordinates for the x-y spline curve
    z = [info['start_flightlevel']] + ctrl_pts[6:]
    # The z-coordinates for the z-spline curve
    total_distance = cv.coordinates_to_distance(info['start_longitudinal'], info['start_latitudinal'],
                                                info['end_longitudinal'], info['end_latitudinal'])
    spline_xy = fit_spline(x, y)
    spline_z = fit_spline(np.linspace(0, total_distance, 6), z)
    trajectory = curve_3D_trajectory_core(flight_nr, spline_xy, spline_z, 0.5)
    return trajectory


def curve_3D_trajectory_core(flight_nr, spline_xy, spline_z, dx):
    """
    The core function for computing the trajectory out of the input spline curves
    We break the trajectory into small straight line pieces according to a constant step in longitude. If in a piece,
    the plane is climbing (or descending), we will again break it into 2 smaller parts, the first being in climbing (or
    descending) mode with fixed ROC (or ROD), and the second being in cruise mode.

    Args:
        flight_nr (int): flight number
        spline_xy : The function of the xy-spline curve, takes x as input and gives y as output
        spline_z : The function of the z-spline curve, takes distance as input and gives z as output
        dx : The incremental step in longitude

    Returns:
        trajectory : Trajectory in the format of a list of tuples (time, longitude, latitude)

    """

    info = da.get_flight_info(flight_nr)
    # To know if the plane is flying from west to east or from east to west
    sign = 1 if info['start_longitudinal'] <= info['end_longitudinal'] else -1
    current_coord = info['start_longitudinal'], info['start_latitudinal'], info['start_flightlevel'], info['start_time']
    trajectory = [current_coord]
    current_distance = 0
    slope = (info['end_latitudinal'] - info['start_latitudinal']) * 111 / \
            ((info['end_longitudinal'] - info['start_longitudinal']) * 85)
    total_distance = cv.coordinates_to_distance(info['start_longitudinal'], info['start_latitudinal'],
                                                info['end_longitudinal'], info['end_latitudinal'])
    # To compute the incremental change in displacement along the straight-line trajectory,
    # since z-spline uses displacement along the straight-line trajectory as input
    dd = np.sqrt(dx ** 2 + (slope * dx) ** 2) * 85
    while True:  # This should always be true if the plane is on the trajectory
        longitude = current_coord[0] + sign * dx
        # Stop the while loop is the plane will reach the ending point after updating
        if sign == 1 and longitude >= info['end_longitudinal']:
            break
        elif sign == -1 and longitude <= info['end_longitudinal']:
            break
        # Otherwise continue
        else:
            latitude = float(spline_xy(longitude))
            current_distance += dd
            flight_level = float(spline_z(min(current_distance, total_distance)))
            # We break each segment into 2 parts, the first being in climbing (or
            # descending) mode with fixed ROC (or ROD), and the second being in cruise mode.
            # We begin by computing the time needed for the first part.
            dt_1 = compute_time_to_reach_other_flightlevel(flight_level, current_coord[2])
            # We take the speed for the first part at the average flight level,
            # and the speed for the second part at the second flight level
            # We notice that the speed is the same regardless of the flying mode in the data.
            speed_1 = cv.ms_to_kms(cv.kts_to_ms(da.get_flight_level_data((flight_level +
                                                                          current_coord[2]) / 2)['CRUISE']['TAS']))
            speed_2 = cv.ms_to_kms(cv.kts_to_ms(da.get_flight_level_data(flight_level)['CRUISE']['TAS']))
            # We compute the displacement projected to x-y plane traveled in the first part
            dxy = np.sqrt((speed_1 * dt_1) ** 2 - cv.fl_to_km(current_coord[2] - flight_level) ** 2)
            # We compute the remaining distance needed in x-y plane and hence the remaining time to travel
            dt_2 = int((cv.coordinates_to_distance(current_coord[0], current_coord[1],
                                                   longitude, latitude) - dxy) / speed_2)
            dt = int(dt_1 + dt_2)
            # We update the coordinate at which part 2 finishes
            current_coord = longitude, latitude, flight_level, cv.update_time(current_coord[3], dt)
            # trajectory.append(intermediate_coord)
            trajectory.append(current_coord)
    # Add the last piece to complete the trajectory. The last speed is chosen to be in cruise mode for convenience.
    speed = cv.ms_to_kms(cv.kts_to_ms(da.get_flight_level_data(current_coord[2])['CRUISE']['TAS']))
    dt = int(cv.coordinates_to_distance(current_coord[0], current_coord[1],
                                        info['end_longitudinal'], info['end_latitudinal']) / speed)
    time = cv.update_time(current_coord[3], dt)
    trajectory.append((info['end_longitudinal'], info['end_latitudinal'], current_coord[2], time))
    return trajectory


def compute_time_to_reach_other_flightlevel(fl1, fl2):
    if fl1 == fl2:
        dt = 0
    elif fl1 < fl2:  # the plane needs to climb
        dt = (fl2 - fl1) / \
             cv.ftm_to_fls(da.get_flight_level_data((fl1 + fl2) / 2)['CLIMB']['ROC'])
    else:  # the plane needs to descend. Also works for cruise mode, as the numerator will be simply 0.
        dt = (fl1 - fl2) / \
             cv.ftm_to_fls(da.get_flight_level_data((fl1 + fl2) / 2)['DESCENT']['ROD'])
    return dt


def fit_spline(x, y, k=2):
    """
    A function that takes in the x and y coordinates of control points, obtain the analytical function of the spline
    curve that interpolates the control points, and returns it.
    Args:
        x : x-coordinates of control points
        y : y-coordinates of control points
        k : optional, the default is 2

    Returns:
        spline : The function of the spline curve, takes x as input and gives y as output

    """

    t, c, k = interpolate.splrep(x, y, s=0, k=k)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    return spline


def plot_b_spline(spline, x, y, N=100):
    """
    makes a plot of the spline curve, and marks the control points on the plot
    Args:
        spline : The function of the spline curve, takes x as input and gives y as output
        x : x-coordinates of control points
        y : y-coordinates of control points
        N : optional, the default is 100

    """
    xx = np.linspace(np.min(x), np.max(x), N)
    plt.plot(x, y, 'bo', label='Control points')
    plt.plot(xx, spline(xx), 'r', label='BSpline')
    plt.grid()
    plt.legend(loc='best')
    plt.show()


def partition(index):
    """
    partitions a list into continuous parts
    eg. [1,2,3,5,6,7,8,11,12,15] --> [[1, 2, 3], [5, 6, 7, 8], [11, 12], [15]]

    :param index: a list of indices
    :return: a list of continuous lists of indices
    """
    parts = []
    a = [index[i] - index[i - 1] for i in range(1, len(index))]
    part = [index[0]]
    j = 0
    while j < len(a):
        if a[j] == 1:
            part.append(index[j + 1])
        else:
            parts.append(part)
            part = [index[j + 1]]
        j += 1
    parts.append(part)
    return parts


def correct_time_for_trajectory(trajectory, time_correction):
    return [(coordinate[0], coordinate[1], coordinate[2], cv.update_time(coordinate[3], time_correction))
            for coordinate in trajectory]


def correct_for_boundaries(trajectory):
    end = False
    trajectory_corrected = [trajectory[0]]
    # Extract the size of the typical dt
    dt = cv.datetime_to_seconds(trajectory[2][3]) - cv.datetime_to_seconds(trajectory[0][3])
    # find points where it goes in and out of our area
    index = []
    for i in range(len(trajectory)):
        """
        #checking lattitudinal value
        if i[1]<-34 or i[1]>60:
            index.append(i)#i-1 is last point in our defined area
        #checking longitudinal value
        if i[0]<-100 or i[0]>30:
            index.append(i)
        """
        if trajectory[i][2] < 100 or trajectory[i][2] > 400:  # checking FL
            index.append(i)
    if len(index) == 0:
        return trajectory
    # partition the bad indices into continuous parts
    partitioned_index = partition(index)
    # find the starting and ending indices for each part
    start_end_parts = [(0, 0)] + [(part[0], part[-1]) for part in partitioned_index]
    time_correction = 0  # The time to correct for following coordinates
    for i in range(1, len(start_end_parts)):
        # append the original good part to the new trajectory,
        # the good part just need to be updated with shifted time caused by changes in the previous part of trajectory
        trajectory_corrected += correct_time_for_trajectory(
            trajectory[start_end_parts[i - 1][1] + 1: start_end_parts[i][0] - 1], time_correction)
        # append the corrected bad part to the new trajectory
        start_longitudinal, start_latitudinal, start_flightlevel, start_time = trajectory[start_end_parts[i][0] - 1]
        try:
            end_longitudinal, end_latitudinal, end_flightlevel, end_time = trajectory[start_end_parts[i][1] + 1]
        except:
            end_longitudinal, end_latitudinal, end_flightlevel, end_time = trajectory[start_end_parts[i][1]]
            end = True
        corrected_part = straight_line_trajectory_core(start_longitudinal, end_longitudinal,
                                                       start_latitudinal, end_latitudinal,
                                                       start_flightlevel, start_time, dt)
        # compute the time saved (or increased, which is unlikely) by correcting the trajectory
        time_correction += int((cv.datetime_to_seconds(corrected_part[-1][3]) -
                                cv.datetime_to_seconds(corrected_part[0][3])) - (cv.datetime_to_seconds(end_time) -
                                                                                 cv.datetime_to_seconds(start_time)))
        trajectory_corrected += corrected_part
    # append the last piece of good part to the new trajectory
    if not end:
        trajectory_corrected += correct_time_for_trajectory(
            trajectory[start_end_parts[-1][1] + 1:], time_correction)
    return trajectory_corrected


# We compute the intermediate point separating the two parts and update the coordinate
"""
intermediate_coord = (longitude * (1 - dt_2 / dt) + current_coord[0] * dt_2 / dt,
                      latitude * (1 - dt_2 / dt) + current_coord[1] * dt_2 / dt,
                      flight_level,
                      cv.update_time(current_coord[3], dt_1))
"""
