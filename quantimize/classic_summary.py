import quantimize.classic.toolbox as toolbox
import quantimize.classic.classic_solution as csol


def straight_line_solution(flight_nr, dt, **kwargs):
    """Returns the straight line solution for a given flight

    Args:
        flight_nr: Flight number
        dt: The time step in seconds

    Returns:
        Trajectory in the format of a list of tuples (time, longitude, latitude) embedded in a dict with the flight number
    """
    trajectory = toolbox.straight_line_trajectory(flight_nr, dt)
    if kwargs.get("timed_trajectory", True):
        trajectory = toolbox.timed_trajectory(trajectory, dt)
    return {"flight_nr": flight_nr, "trajectory": trajectory}


def genetic_algorithm_solution(flight_nr, dt, **kwargs):
    """Returns the genetic algorithm solution for a given flight and maps it to constant time points

    Args:
        flight_nr (int): Flight number

    Returns:
        dict: Trajectory in the format of a list of tuples (long, lat, time) embedded in a dict with the flight number
    """
    # tries to run the code and gives it another shot, if it fails. Common error is "hour must be in 0..23"
    run_complete = False
    while not run_complete:
        try:
            report, solution, trajectory = csol.run_genetic_algorithm(
                flight_nr, **kwargs)
            run_complete = True
        except:
            print("Retry GA")
    if kwargs.get("timed_trajectory", True):
        trajectory = toolbox.timed_trajectory(trajectory, dt)
    if kwargs.get("only_trajectory_dict", False):
        return {"flight_nr": flight_nr, "trajectory": trajectory}
    return report, solution, {"flight_nr": flight_nr, "trajectory": trajectory}


def compute_cost(trajectory):
    """Wrapper for the computation of the cost function

    Args:
        trajectory (list): List of trajectory points

    Returns:
        float: Cost of the flight
    """
    return toolbox.compute_cost(trajectory)
