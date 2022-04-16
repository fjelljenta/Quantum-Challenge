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
    report, solution, trajectory = run_genetic_algorithm(flight_nr, **kwargs)
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