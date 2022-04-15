import quantimize.classic.classic_solution as csol
import quantimize.classic.toolbox as ct
import quantimize.quantum.quantum_solution as qsol
import quantimize.quantum.QGA as qga
import quantimize.quantum.quantum_neural_network as qna
import quantimize.converter as cv
import quantimize.data_access as da

import time


#decorator: give solution_function as argument

#computing_time and climate_impact combined

#given: cost(trajectory),

#decorator
def computiation_time(flight_number, algorithm):
    start = time.perf_counter()
    trajectory = algorithm(flight_number)
    stop = time.perf_counter()
    computation_time= stop-start
    return computation_time, trajectory

def average_computation_time(algorithm, **kwargs):
    flightlist= kwargs.get('flights', range(100))
    return_trajectory=kwargs.get('return_trajectory', True)

    number_of_flights=len(flightlist)
    trajectory_list = []
    computation_time_list = []
    for flight_number in flightlist:
        if flight_number != 41:
            computation_time_tmp, trajectory_tmp =computiation_time(flight_number, algorithm)
            computation_time_list.append(computation_time_tmp)
            trajectory_list.append(trajectory_tmp)
        else:
            number_of_flights = number_of_flights -1
    average_computation_time=sum(computation_time_list)/number_of_flights
    if return_trajectory:
        return average_computation_time, trajectory_list
    else:
        return average_computation_time


def average_cost(trajectory_list):
    number_of_flights = len(trajectory_list)
    cost_list = []
    for flight in trajectory_list:
        cost=ct.compute_cost(flight)
        cost_list.append(cost)
    average_cost_per_flight=sum(cost_list)/number_of_flights
    return average_cost_per_flight


def averaged_flight_time(trajectory_list):
    number_of_flights = len(trajectory_list)
    time_list = []
    for flight in trajectory_list:
        start = cv.datetime_to_seconds(flight[0][3])
        stop = cv.datetime_to_seconds(flight[-1][3])
        time_list.append(stop-start)
    average_time_per_flight = sum(time_list) / number_of_flights
    return average_time_per_flight

def fuel_consumption(trajectory):

    """Returns the fuel_consumption for a given list of trajectories in 10**-12

    Args:
        trajectory (list): List describing one flight trajectory

    Returns:
        float, int: fuel_consumption in kg and flight number
    """
    fuel = 0
    flight_path = cv.check_trajectory_dict(trajectory)
    t = cv.datetime_to_seconds(flight_path[0][3])
    start_level = flight_path[0][2]
    # we compare the flight levels between neighbors to know in which mode (climb/cruise/descend) the plane is flying
    for coordinate in flight_path[1:]:
        # incremental  fuel consumpution rate (in kg / min ) times
        # change in temperature per portion of fuel (in K / kg)
        if coordinate[2] == start_level:
            fuel += da.get_flight_level_data(coordinate[2])['CRUISE']['fuel'] \
                * (cv.datetime_to_seconds(coordinate[3]) - t) / 60
        else:
            dt1 = ct.compute_time_to_reach_other_flightlevel(start_level, coordinate[2])
            if coordinate[2] < start_level:
                fuel += da.get_flight_level_data(coordinate[2])['DESCENT']['fuel'] \
                        * dt1 / 60 + \
                        da.get_flight_level_data(coordinate[2])['CRUISE']['fuel'] \
                        * (cv.datetime_to_seconds(coordinate[3]) - t - dt1) / 60
            else:
                fuel += da.get_flight_level_data(coordinate[2])['CLIMB']['fuel'] \
                        * dt1 / 60 + \
                        da.get_flight_level_data(coordinate[2])['CRUISE']['fuel'] \
                        * (cv.datetime_to_seconds(coordinate[3]) - t - dt1) / 60
        #start_level = coordinate[2]
        #t = cv.datetime_to_seconds(coordinate[3])
        return fuel


def average_fuel(trajectory_list):
    number_of_flights = len(trajectory_list)
    fuel_list = []
    for flight in trajectory_list:
        fuel=fuel_consumption(flight)
        fuel_list.append(fuel)
    average_fuel_per_flight=sum(fuel_list)/number_of_flights
    return average_fuel_per_flight

