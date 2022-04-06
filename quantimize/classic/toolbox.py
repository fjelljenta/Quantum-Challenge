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
    t = cv.datetime_to_seconds(trajectory["trajectory"][0][3])
    start_level = trajectory["trajectory"][0][2]
    # we compare the flight levels between neighbors to know in which mode (climb/cruise/descend) the plane is flying
    for coordinate in trajectory["trajectory"][1:]:
        # incremental cost equals the time change (in min) times fuel consumpution rate (in kg / min ) times
        # change in temperature per portion of fuel (in K / kg)
        if coordinate[2] == start_level:
            cost += (da.get_merged_atmo_data(*coordinate)) * da.get_flight_level_data(coordinate[2])['CRUISE']['fuel']\
                    * (cv.datetime_to_seconds(coordinate[3])-t) / 60
        elif coordinate[2] < start_level:
            cost += (da.get_merged_atmo_data(*coordinate)) * da.get_flight_level_data(coordinate[2])['DESCENT']['fuel']\
                    * (cv.datetime_to_seconds(coordinate[3])-t) / 60
            start_level = coordinate[2]
        else:
            cost += (da.get_merged_atmo_data(*coordinate)) * da.get_flight_level_data(coordinate[2])['CLIMB']['fuel']\
                    * (cv.datetime_to_seconds(coordinate[3])-t) / 60
            start_level = coordinate[2]
        t = cv.datetime_to_seconds(coordinate[3])
    return cost


def straight_line_solution(flight_nr, dt):
    """Returns the straight line solution for a given flight 
    
    Args:
        flight_nr: Flight number
        dt: The time step in minutes
        
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
    slope = (info['end_latitudinal'] - info['start_latitudinal']) * 111 / \
            ((info['end_longitudinal'] - info['start_longitudinal']) * 85)
    # compute the slope of the straight line connecting start and end points in x-y plane
    speed = cv.ms_to_kms(cv.kts_to_ms(da.get_flight_level_data(info['start_flightlevel'])['CRUISE']['TAS']))
    # The speed is remaining the same throughout the trajectory as we stay on the same flight level
    total_distance = cv.coordinates_to_distance(info['start_longitudinal'], info['start_latitudinal'],
                                                info['end_longitudinal'], info['end_latitudinal'])
    current_coord = info['start_longitudinal'], info['start_latitudinal'], info['start_flightlevel'], info['start_time']
    trajectory = [current_coord]
    current_displacement = 0
    while current_displacement < total_distance:
        # We keep record of the distance travelled so that we know when to stop
        current_displacement += speed * dt
        time = cv.update_time(current_coord[3], dt)
        longitude = current_coord[0] + speed * dt * np.cos(np.arctan(slope)) / 85
        latitude = current_coord[1] + speed * dt * np.sin(np.arctan(slope)) / 111
        current_coord = longitude, latitude, info['start_flightlevel'], time
        trajectory.append(current_coord)
    # The last coordinate is in general beyond the ending point as we are taking a constant time step, so we need to
    # correct it by replacing with the desired ending point.
    total_time = int(total_distance/speed)
    end_time = cv.update_time(info['start_time'], total_time)
    trajectory[-1] = (info['end_longitudinal'], info['end_latitudinal'], info['start_flightlevel'], end_time)
    return trajectory


def curve_3D_solution(flight_nr, ctrl_pts):
    """Returns the curved 3D solution for a given flight number

    Args:
    flight_nr (int): flight number

    Returns:
    dictionary with flight numbers and corresponding trajectory

    """
    trajectory = curve_3D_trajectory(flight_nr, ctrl_pts)
    return {"flight_nr": flight_nr, "trajectory": trajectory}


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
    trajectory = curve_3D_trajectory_core(flight_nr, spline_xy, spline_z, 0.3)
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
    # To compute the incremental change in displacement along the straight-line trajectory,
    # since z-spline uses displacement along the straight-line trajectory as input
    dd = np.sqrt(dx**2 + (slope*dx)**2)*85
    while True:  # This should always be true if the plane is on the trajectory
        longitude = current_coord[0] + sign * dx
        latitude = float(spline_xy(longitude))
        current_distance += dd
        flight_level = float(spline_z(current_distance))
        # Stop the while loop is the plane will reach the ending point after updating
        if sign == 1 and longitude > info['end_longitudinal']:
            break
        elif sign == -1 and longitude < info['end_longitudinal']:
            break
        # Otherwise continue
        else:
            # We break each segment into 2 parts, the first being in climbing (or
            # descending) mode with fixed ROC (or ROD), and the second being in cruise mode.
            # We begin by computing the time needed for the first part.
            if flight_level > current_coord[2]:  # the plane needs to climb
                dt_1 = (flight_level - current_coord[2]) / \
                       cv.ftm_to_fls(da.get_flight_level_data((flight_level+current_coord[2])/2)['CLIMB']['ROC'])
            else:  # the plane needs to descend. Also works for cruise mode, as the numerator will be simply 0.
                dt_1 = (current_coord[2] - flight_level) / \
                       cv.ftm_to_fls(da.get_flight_level_data((flight_level+current_coord[2])/2)['DESCENT']['ROD'])
            # We take the speed for the first part at the average flight level,
            # and the speed for the second part at the second flight level
            # We notice that the speed is the same regardless of the flying mode in the data.
            speed_1 = cv.ms_to_kms(cv.kts_to_ms(da.get_flight_level_data((flight_level +
                                                                          current_coord[2])/2)['CRUISE']['TAS']))
            speed_2 = cv.ms_to_kms(cv.kts_to_ms(da.get_flight_level_data(flight_level)['CRUISE']['TAS']))
            # We compute the displacement projected to x-y plane traveled in the first part
            dxy = np.sqrt((speed_1 * dt_1)**2 - cv.fl_to_km(current_coord[2] - flight_level)**2)
            # We compute the remaining distance needed in x-y plane and hence the remaining time to travel
            dt_2 = int((cv.coordinates_to_distance(current_coord[0], current_coord[1],
                                                   longitude, latitude) - dxy) / speed_2)
            dt_1 = int(dt_1)
            dt = dt_1 + dt_2
            # We compute the intermediate point separating the two parts and update the coordinate
            intermediate_coord = (longitude * dt_2 / dt + current_coord[0] * (1 - dt_2 / dt),
                                  latitude * dt_2 / dt + current_coord[1] * (1 - dt_2 / dt),
                                  flight_level,
                                  cv.update_time(current_coord[3], dt_1))
            # We update the coordinate at which part 2 finishes
            current_coord = longitude, latitude, flight_level, cv.update_time(current_coord[3], dt)
            trajectory.append(intermediate_coord)
            trajectory.append(current_coord)
    # Add the last piece to complete the trajectory. The last speed is chosen to be in cruise mode for convenience.
    speed = cv.ms_to_kms(cv.kts_to_ms(da.get_flight_level_data(current_coord[2])['CRUISE']['TAS']))
    dt = int(cv.coordinates_to_distance(current_coord[0], current_coord[1],
                                        info['end_longitudinal'], info['end_latitudinal']) / speed)
    time = cv.update_time(current_coord[3], dt)
    trajectory.append((info['end_longitudinal'], info['end_latitudinal'], current_coord[2], time))
    return trajectory


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


"""
def correct_for_boundaries(trajectory):
    dt=cv.datetime_to_seconds(trajectory[0][3])-cv.datetime_to_seconds(trajectory[1][3])
    #find points where it goes in and out of our area
    index=[]
    for i in len(trajectory):
        #checking lattitudinal value
        if i[1]<-34 or i[1]>60:
            index.append(i)#i-1 is last point in our defined area
        #checking longitudinal value
        if i[0]<-100 or i[0]>30:
            index.append(i)
        #checking FL
        if i[2]<100 or long>400:
            index.append[i]

    index_shift =[x + 1 for x in index]
    bad_parts=[]
    a=0
    for i in index:
        if index[i]-index_shift[i]!=a:
            bad_parts.append(index_shift[i])
            a=index[i]-index_shift[i]
        else:
            pass

    trajectory_corrected=[]
    number=len(bad_parts)
    time_dif=0
    for i in range(0,number,2):
        start_point=bad_parts[i]
        end_point=bad_parts[i+2]
        trajectory_corrected.append(trajectory[x][0], trajectory[x][1], trajectory[x][2],
                                    cv.update_time(trajectory[x][3], time_dif) for x < start_point)
        start_latitudinal = trajectory[start_point][1]
        end_latitudinal=trajectory[end_point][1]
        start_longitudinal = trajectory[start_point][0]
        end_longitudinal = trajectory[end_point][0]
        start_flightlevel = trajectory[start_point][2]
        end_flightlevel = trajectory[end_point][2]
        start_time = trajectory[start_point][3]
        slope = (end_latitudinal - start_latitudinal) * 111 / \
                ((end_longitudinal - start_longitudinal) * 85)
        total_distance = cv.coordinates_to_distance3D(start_longitudinal, start_latitudinal, start_flightlevel,
                                                    end_longitudinal, end_latitudinal, end_flightlevel)
        current_coord = start_longitudinal, start_latitudinal, flight_level, start_time
        trajectory_corrected.append(current_coord)
        current_distance = 0
        while current_distance < total_distance:
            current_distance += speed*dt
            time = cv.update_time(current_coord[3], dt)
            longitude = current_coord[0] + speed * dt * np.cos(np.arctan(slope)) / 85
            latitude = current_coord[1] + speed * dt * np.sin(np.arctan(slope)) / 111
            if end_flightlevel > start_flightlevel:
                dt1 = int((end_flightlevel - start_flightlevel) /
                          cv.ftm_to_fls(da.get_flight_level_data((start_flightlevel+end_flightlevel)/2)['CLIMB']['ROC']))
            elif start_flightlevel > end_flightlevel:
                dt1 = int((end_flightlevel - start_flightlevel) /
                          cv.ftm_to_fls(da.get_flight_level_data((start_flightlevel+end_flightlevel)/2)['DESCENT']['ROD']))
            else:
                pass
            speed = cv.ms_to_kms(cv.kts_to_ms(da.get_flight_level_data((start_flightlevel+
                                                                        end_flightlevel)/2)['CRUISE']['TAS']))
            dt = int(cv.coordinates_to_distance(current_coord[0], current_coord[1], longitude, latitude) / speed)
            intermediate_coord = (longitude * (1 - dt1 / dt) + current_coord[0] * dt1 / dt,
                                  latitude * (1 - dt1 / dt) + current_coord[1] * dt1 / dt,
                                  end_flightlevel,
                                  cv.update_time(current_coord[3], dt1))
            current_coord = longitude, latitude, end_flightlevel, cv.update_time(current_coord[3], dt)
            trajectory_corrected.append(intermediate_coord)
            trajectory_corrected.append(current_coord)

        trajectory_corrected.append(end_longitudinal, end_latitudinal, end_flightlevel, current_coord[3])

        time_dif=cv.datetime_to_seconds(current_coord[3])-cv.datetime_to_seconds(trajectory[end_point][3])
        # return(trajectory_corrected)
"""