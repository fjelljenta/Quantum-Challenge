"""
Main run for quantum algorithm here, import functions from files in quantum folder
"""

import quantimize.classic.toolbox as toolbox
import quantimize.quantum.QGA as qga
import quantimize.quantum.quantum_neural_network as qna
import quantimize.quantum.quantum_solution as qsol

def quantum_genetic_algorith_solution(flight_nr, dt):
    """Returns the quantum_genetic algorithm solution for a given flight and maps it to constant time points

    Args:
        flight_nr (int): Flight number

    Returns:
        dict: Trajectory in the format of a list of tuples (long, lat, time) embeded in a dict with flightnumber
    """
    trajectory = qga.Q_GA(flight_nr)
    trajectory = toolbox.timed_trajectory(trajectory, dt)
    return {"flight_nr": flight_nr, "trajectory": trajectory}

def quantum_neural_network(flight_nr, n_qubits, init_solution):
    return qna.quantum_neural_network(flight_nr, n_qubits, init_solution)

def sample_grid():
    return qsol.sample_grid()

def run_QAOA(cg, orientation=0, verbose=False):
    return qsol.run_QAOA(cg, orientation, verbose)
    
def compute_cost(trajectory):
    """Wrapper for the compute cost function

    Args:
        trajectory (list): List of trajectory points

    Returns:
        float: Cost of the flight
    """
    return toolbox.compute_cost(trajectory)