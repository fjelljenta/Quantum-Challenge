"""
Main run for quantum algorithm here, import functions from files in quantum folder
"""

import quantimize.classic.toolbox as toolbox
import quantimize.quantum.QGA as qga
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

def compute_cost(trajectory):
    """Wrapper for the compute cost function

    Args:
        trajectory (list): List of trajectory points

    Returns:
        float: Cost of the flight
    """
    return toolbox.compute_cost(trajectory)