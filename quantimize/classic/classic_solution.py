from quantimize.classic.toolbox import *
from geneticalgorithm import geneticalgorithm as ga
from functools import partial


def run_genetic_algorithm(flight_nr, **kwargs):
    """Genetic algorithm for the classical solution for a certain flight
    
    Args:
        flight_nr (int): flight number

    Returns:
        report : model report
        solution (dict): output dictionary
        trajectory (dict): dictionary containing the flights and the corresponding tranjectories

    """
    varbound = generate_search_bounds(flight_nr, **kwargs)
    algorithm_param = {'max_num_iteration': kwargs.get("max_iter", 100),
                       'population_size': kwargs.get("pop_size", 10),
                       'mutation_probability': kwargs.get("mut_prob", 0.1),
                       'elit_ratio': kwargs.get("elit_ratio", 0.01),
                       'crossover_probability': kwargs.get("co_prob", 0.5),
                       'parents_portion': kwargs.get("pp", 0.3),
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': kwargs.get("max_iter_w_i", 50)}
    model = ga(fitness_function_single_flight(flight_nr), dimension=11, variable_type='real',
               variable_boundaries=varbound, algorithm_parameters=algorithm_param, convergence_curve=False)
    model.run()
    report = model.report
    solution = model.output_dict
    trajectory = curve_3D_trajectory(flight_nr, solution['variable'])
    return report, solution, trajectory


def fitness_function_single_flight(flight_nr):
    """
    
    Args:
        flight_nr (int): flight number

    Returns:
        partial

    """
    return partial(fitness_function, flight_nr)


def fitness_function(flight_nr, ctrl_pts):
    """    Takes in a list of 11 parameters for control points, first 3 for x, next 3 for y, last 5 for z.
    obtain a trajectory and then compute cost
    
    Args:
        flight_nr (int): flight number
        ctrl_pts : control points
    
    Returns:
        cost (float): Environmental cost of a tranjectory
    """
    trajectory = curve_3D_solution(flight_nr, ctrl_pts)
    cost = compute_cost(trajectory)
    return cost


def generate_search_bounds(flight_nr, **kwargs):
    """Generation of search boundaries for a certain flight

    Args:
    flight_nr (int): flight number

    Returns:
        array with boundaries

    """
    info = da.get_flight_info(flight_nr)
    total_distance = cv.coordinates_to_distance(info['start_longitudinal'], info['start_latitudinal'],
                                                info['end_longitudinal'], info['end_latitudinal'])
    dx = np.abs(info['end_longitudinal'] - info['start_longitudinal']) / 60 * 0.05 * total_distance/85
    # x shouldn't vary a lot
    dy = np.abs(info['end_latitudinal'] - info['start_latitudinal']) / 26 * 0.1 * total_distance/111
    x1 = 3/4*info['start_longitudinal'] + 1/4*info['end_longitudinal']
    x1_bound = [max(x1-dx, -30), min(x1+dx, 30)]
    x2 = 1/2*info['start_longitudinal'] + 1/2*info['end_longitudinal']
    x2_bound = [max(x2-dx, -30), min(x2+dx, 30)]
    x3 = 1/4*info['start_longitudinal'] + 3/4*info['end_longitudinal']
    x3_bound = [max(x3-dx, -30), min(x3+dx, 30)]
    y1 = 3/4*info['start_latitudinal'] + 1/4*info['end_latitudinal']
    y1_bound = [max(y1-dy, 34), min(y1+dy, 60)]
    y2 = 1/2*info['start_latitudinal'] + 1/2*info['end_latitudinal']
    y2_bound = [max(y2-dy, 34), min(y2+dy, 60)]
    y3 = 1/4*info['start_latitudinal'] + 3/4*info['end_latitudinal']
    y3_bound = [max(y3-dy, 34), min(y3+dy, 60)]
    z_bound = [100, 400]
    return np.array([x1_bound] + [x2_bound] + [x3_bound] + [y1_bound] + [y2_bound] + [y3_bound] + [z_bound]*5)


