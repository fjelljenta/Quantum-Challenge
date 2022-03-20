from quantimize.classic.toolbox import *
from geneticalgorithm import geneticalgorithm as ga
from functools import partial


def run_genetic_algorithm(flight_nr):
    varbound = generate_search_bounds(flight_nr)
    algorithm_param = {'max_num_iteration': 100,
                       'population_size': 10,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': 50}
    model = ga(fitness_function_single_flight(flight_nr), dimension=11, variable_type='real',
               variable_boundaries=varbound, algorithm_parameters=algorithm_param)
    model.run()
    report = model.report
    solution = model.output_dict
    trajectory = curve_3D_trajectory(flight_nr, solution['variable'])
    return report, solution, trajectory


def fitness_function_single_flight(flight_nr):
    return partial(fitness_function, flight_nr)


def fitness_function(flight_nr, ctrl_pts):
    """
    Takes in a list of 11 parameters for control points, first 3 for x, next 3 for y, last 5 for z.
    obtain a trajectory and then compute cost
    :return:
    """
    trajectory = curve_3D_trajectory(flight_nr, ctrl_pts)
    cost = compute_cost(trajectory)
    print(cost)
    return cost


def generate_search_bounds(flight_nr):
    info = da.get_flight_info(flight_nr)
    total_distance = cv.coordinates_to_distance(info['start_longitudinal'], info['start_latitudinal'],
                                                info['end_longitudinal'], info['end_latitudinal'])
    dx = (info['end_longitudinal'] - info['start_longitudinal']) / 60 * 0.05*total_distance/85
    dy = (info['end_latitudinal'] - info['start_latitudinal']) / 26 * 0.1*total_distance/111
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
    z_bound = [120, 380]
    return np.array([x1_bound] + [x2_bound] + [x3_bound] + [y1_bound] + [y2_bound] + [y3_bound] + [z_bound]*5)


