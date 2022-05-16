import quantimize.classic_summary as csol
import quantimize.classic.toolbox as ct
import quantimize.quantum_summary as qsol
import quantimize.converter as cv
import quantimize.data_access as da
import numpy as np
import quantimize.benchmarking.toolbox as bt
import time
from multiprocessing import Pool
import quantimize.air_security as ais


def computation_time(flight_number, algorithm):
    """Calculation of the computation time for one flight and a with the second argument
    specified optimization algorithm

    Args:
        flight_number (int): flight number
        algorithm (function): algorithm used to optimize the trajectory

    Returns:
        float: computation time needed for the optimization in seconds
        list: optimized trajectory of the given flight
    """
    start = time.perf_counter()
    trajectory = algorithm(flight_number)
    stop = time.perf_counter()
    computation_t = stop-start
    return computation_t, trajectory


def average_computation_time(algorithm, **kwargs):
    """Averaged computation time for a with the first argument specified algorithm for several/all flights

    Args:
        algorithm (function): algorithm used to optimize the trajectory
        **kwargs: possibility to hand in a list of flight numbers, if no done all flights will be considered
                  possibility to specify if a list of all trajectories should be returned, default value: TRUE

    Returns:
        float: Averaged computation time for a with the first argument specified algorithm for several/all flights
        list: trajectory_list: list containing all trajectories

    """
    flightlist = kwargs.get('flights', range(100))
    return_trajectory = kwargs.get('return_trajectory', True)

    number_of_flights = len(flightlist)
    trajectory_list = []
    computation_time_list = []
    for flight_number in flightlist:
        if flight_number != 41:
            computation_time_tmp, trajectory_tmp = computation_time(flight_number, algorithm)
            computation_time_list.append(computation_time_tmp)
            trajectory_list.append(trajectory_tmp)
        else:
            number_of_flights = number_of_flights - 1
    average_computation_t = sum(computation_time_list)/number_of_flights
    if return_trajectory:
        return average_computation_t, trajectory_list
    else:
        return average_computation_t


def average_computation_time_multiprocessing(algorithm, flights, run, **kwargs):
    """Averaged computation time for a with the first argument specified algorithm for all flights

        Args:
            algorithm (function): algorithm used to optimize the trajectory
            flights (list): list with flight numbers that shall be processed
            run: counter for the repititions in the benchmarking-procedure
            **kwargs: possibility to specify whether the Benchmarking should be done including air security,
                      default value: False

        Returns:
            float: Averaged computation time for a with the first argument specified algorithm for all flights
            list: trajectory_list: list containing all trajectories

        """
    air_security = kwargs.get("Air_security", False)
    start_time = time.perf_counter()
    if air_security:
        print("\n")
        trajectories, conflicts = ais.safe_algorithm_2(flights, algorithm)
    else:
        algorithmarguments = [[flightnumber] for flightnumber in flights]
        with Pool() as p:
            trajectories = p.starmap(algorithm, algorithmarguments)
    print("Finished run ", run+1, " of trajectory calculation with ", str(algorithm)[1:-23])
    for i, trajectory in enumerate(trajectories):
        trajectories[i] = cv.check_trajectory_dict(trajectory)
    stop_time = time.perf_counter()
    computation_t = (stop_time - start_time)/len(flights)
    return computation_t, trajectories


def average_cost(trajectory_list):
    """Computation of the average climate impacted for several/all flights

    Args:
        trajectory_list (list): list, which contains all the trajectories of all the flights
            we want to take into account

    Returns:
        float: average_cost_per_flight average climate impact per flight in 10^(-12)K

    """
    number_of_flights = len(trajectory_list)
    cost_list = []
    for flight in trajectory_list:
        cost = ct.compute_cost(flight)
        cost_list.append(cost)
    average_cost_per_flight = sum(cost_list)/number_of_flights
    return average_cost_per_flight


def averaged_flight_time(trajectory_list):
    """Computation of the averaged flight time per flight

    Args:
        trajectory_list (list): list, which contains all the trajectories of all the flights
            we want to take into account

    Returns:
        float: average_time_per_flight averaged flight time per flight in seconds

    """
    number_of_flights = len(trajectory_list)
    time_list = []
    for flight in trajectory_list:
        start = cv.datetime_to_seconds(flight[0][3])
        stop = cv.datetime_to_seconds(flight[-1][3])
        time_list.append(stop-start)
    average_time_per_flight = sum(time_list) / number_of_flights
    return average_time_per_flight


def fuel_consumption(trajectory):
    """Returns the fuel consumption for a given trajectory in 10**-12

    Args:
        trajectory (list): List describing one flight trajectory

    Returns:
        float: fuel_consumption in kg for a given flight number
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
            start_level = coordinate[2]
        t = cv.datetime_to_seconds(coordinate[3])
    return fuel


def average_fuel(trajectory_list):
    """Computation of the averaged fuel consumption over several/all flights

    Args:
        trajectory_list (list): list, which contains all the trajectories of all the flights
            we want to take into account

    Returns:
        float: average fuel consumption per flight

    """
    number_of_flights = len(trajectory_list)
    fuel_list = []
    for flight in trajectory_list:
        fuel = fuel_consumption(flight)
        fuel_list.append(fuel)
    average_fuel_per_flight = sum(fuel_list)/number_of_flights
    return average_fuel_per_flight


def costlist_creation(algorithm, flights, runs, **kwargs):
    """Calculation of lists for computation cost, climate cost, flight time and fuel for a number of flights
        with flight trajectories defined via algorithm

        Args:
            algorithm (function): algorithm to use for trajectory calculation
            flights (list): list of the flight numbers of interest
            runs (int): number of runs for the benchmarking
            **kwargs: possibility to specify whether the Benchmarking should be done including air security,
                      default value: False

        Returns:
            dict: Costlists for computation cost, climate cost, flight time and fuel

        """
    cost_comp, cost, flight_time, fuel = np.zeros(runs), np.zeros(runs), np.zeros(runs), np.zeros(runs)
    for counter in range(runs):
        cost_comp[counter], trajectory = average_computation_time_multiprocessing(algorithm, flights, counter, **kwargs)
        cost[counter] = average_cost(trajectory)
        flight_time[counter] = averaged_flight_time(trajectory)
        fuel[counter] = average_fuel(trajectory)
    return {"cost_comp": cost_comp, "cost": cost, "flight_time": flight_time, "fuel": fuel}


def mean_and_error_list_creation(calculation_list):
    """calculates mean value and error for a given list of floats

        Args:
            calculation_list (list): list of floats

        Returns:
            float: mean value, error

        """
    mean = np.mean(calculation_list, axis=1)
    error = np.std(calculation_list, axis=1)
    return mean, error


def benchmark_wrapper(flights, runs, **kwargs):
    """Calculation of the mean values and errors of the averaged computation time, averaged climate cost,
        averaged flight timeand of the averaged fuel consumption

    Args:
        flights (list): list of the flight numbers of interest
        runs (int): number of runs for the benchmarking

    Returns:
        float: Mean_comp_time, Error_comp_time, Mean_cost, Error_cost, Mean_flight_time, Error_flight_time,
            Mean_fuel, Error_fuel

    """

    if kwargs.get("Air_security", False):
        ga_costlist = costlist_creation(csol.genetic_algorithm_solution, flights, runs, **kwargs)
        qga_costlist = costlist_creation(qsol.quantum_genetic_algorithm_solution, flights, runs, **kwargs)
        costlists = [ga_costlist, qga_costlist]
    else: 
        sl_costlist = costlist_creation(csol.straight_line_solution, flights, runs, **kwargs)
        ga_costlist = costlist_creation(csol.genetic_algorithm_solution, flights, runs, **kwargs)
        qga_costlist = costlist_creation(qsol.quantum_genetic_algorithm_solution, flights, runs, **kwargs)
        costlists = [sl_costlist, ga_costlist, qga_costlist]

    cost_comp, cost, flight_time, fuel = [], [], [], []
    for costlist in costlists:
        cost_comp.append(costlist.get("cost_comp"))
        cost.append(costlist.get("cost"))
        flight_time.append(costlist.get("flight_time"))
        fuel.append(costlist.get("fuel"))

    mean_comp_time, error_comp_time = mean_and_error_list_creation(cost_comp)
    mean_cost, error_cost = mean_and_error_list_creation(cost)
    mean_flight_time, error_flight_time = mean_and_error_list_creation(flight_time)
    mean_fuel, error_fuel = mean_and_error_list_creation(fuel)

    return mean_comp_time, error_comp_time, mean_cost, error_cost, mean_flight_time, error_flight_time, mean_fuel, error_fuel
