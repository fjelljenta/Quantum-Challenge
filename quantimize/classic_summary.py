"""
Main run for classic algorithm here, import functions from files in classic folder
"""

import quantimize.classic.toolbox as toolbox
import quantimize.classic.classic_solution as csol

def straight_line_solution(flight_nr, dt):
    """Returns the straight line solution for a given flight

    Args:
        flight_nr: Flight number
        dt: The time step in seconds

    Returns:
        Trajectory in the format of a list of tuples (time, longitude, latitude) embeded in a dict with flightnumber
    """
    trajectory = toolbox.straight_line_trajectory(flight_nr, dt)
    trajectory = toolbox.timed_trajectory(trajectory, dt)
    return {"flight_nr": flight_nr, "trajectory": trajectory}


def genetic_algorith_solution(flight_nr, dt):
    """Returns the genetic algorithm solution for a given flight and maps it to constant time points

    Args:
        flight_nr (int): Flight number

    Returns:
        dict: Trajectory in the format of a list of tuples (long, lat, time) embeded in a dict with flightnumber
    """
    report, solution, trajectory = csol.run_genetic_algorithm(flight_nr)
    trajectory = toolbox.timed_trajectory(trajectory, dt)
    return report, solution, {"flight_nr": flight_nr, "trajectory": trajectory}

def compute_cost(trajectory):
    """Wrapper for the compute cost function

    Args:
        trajectory (list): List of trajectory points

    Returns:
        float: Cost of the flight
    """
    return toolbox.compute_cost(trajectory)