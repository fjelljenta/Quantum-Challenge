import quantimize.data_access as da
import quantimize.classic.toolbox as ct
import quantimize.classic.classic_solution as csol
from geneticalgorithm import geneticalgorithm as ga
import quantimize.quantum.QGA as qga
import quantimize.quantum.quantum_neural_network as qnn
import quantimize.quantum.quantum_solution as qsol




def sl_for_benchmarking(flight_nr):
    """Computes and returns the straight line solution for a given flight
       The straight line solution assumes the same flight level as starting point for the ending point
    Args:
        flight_nr: Flight number

    Returns:
        Trajectory in the format of a list of tuples (time, longitude, latitude)
    """
    dt=10
    info = da.get_flight_info(flight_nr)
    trajectory = ct.straight_line_trajectory_core(info['start_longitudinal'], info['end_longitudinal'],
                                               info['start_latitudinal'], info['end_latitudinal'],
                                               info['start_flightlevel'], info['start_time'], dt)
    return trajectory


def ga_for_benchmarking(flight_nr, **kwargs):
    """Genetic algorithm for the classical solution for a certain flight

    Args:
        flight_nr (int): flight number

    Returns:
        trajectory (dict): dictionary containing the flights and the corresponding tranjectories

    """
    varbound = csol.generate_search_bounds(flight_nr)
    algorithm_param = {'max_num_iteration': kwargs.get("max_iter", 100),
                       'population_size': kwargs.get("pop_size", 10),
                       'mutation_probability': kwargs.get("mut_prob", 0.1),
                       'elit_ratio': kwargs.get("elit_ratio", 0.01),
                       'crossover_probability': kwargs.get("co_prob", 0.5),
                       'parents_portion': kwargs.get("pp", 0.3),
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': kwargs.get("max_iter_w_i", 50)}
    model = ga(csol.fitness_function_single_flight(flight_nr), dimension=11, variable_type='real',
               variable_boundaries=varbound, algorithm_parameters=algorithm_param, convergence_curve=False)
    model.run()
    solution = model.output_dict
    trajectory = ct.curve_3D_trajectory(flight_nr, solution['variable'])
    return trajectory


def qga_for_benchmarking(flight_nr):
    trajectory = qga.Q_GA(flight_nr)
    return trajectory


def qnn_for_benchmarking(flight_nr):
    n_qubits=6
    report_ga, init_solution_ga, trajectory_ga = csol.run_genetic_algorithm(flight_nr)
    print(init_solution_ga)
    qnn.quantum_neural_network(flight_nr, n_qubits, init_solution_ga)
    return trajectory

def qaoa_for_benchmarking(flight_nr):
    return None