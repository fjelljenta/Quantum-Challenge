import quantimize.data_access as da
import quantimize.classic.toolbox as ct
import quantimize.classic.classic_solution as csol
import quantimize.quantum.QGA as qga
import matplotlib.pyplot as plt
import numpy as np


def sl_for_benchmarking(flight_nr):
    """Computes and returns the straight line solution for a given flight
       The straight line solution assumes the same flight level as starting point for the ending point
    Args:
        flight_nr: Flight number

    Returns:
        Trajectory in the format of a list containing coordinates and time stamps
    """
    dt = 10
    info = da.get_flight_info(flight_nr)
    trajectory = ct.straight_line_trajectory_core(info['start_longitudinal'], info['end_longitudinal'],
                                                  info['start_latitudinal'], info['end_latitudinal'],
                                                  info['start_flightlevel'], info['start_time'], dt)
    return trajectory


def ga_for_benchmarking(flight_nr, **kwargs):
    """Computes and returns the genetic algorithm solution for a given flight
    Args:
        flight_nr: Flight number

    Returns:
        Trajectory in the format of a list containing coordinates and time stamps
    """
    report, solution, trajectory = csol.run_genetic_algorithm(flight_nr, **kwargs)
    return trajectory


def qga_for_benchmarking(flight_nr, **kwargs):
    """Computes and returns the quantum genetic algorithm solution for a given flight
    Args:
        flight_nr: Flight number

    Returns:
        Trajectory in the format of a list containing coordinates and time stamps
    """
    trajectory = qga.Q_GA(flight_nr)
    return trajectory


def plot_graph(titel, y_label, mean_value_list, error_list):
    """Creation of bar plots with error bars

    Args:
        titel (string): title of the plot
        y_label (string): y-axis label of the plot
        mean_value_list (list): list of mean values for the bars
        error_list (list): list of the corresponding errors for the error bar

    Returns:
        plot
    """
    if len(mean_value_list) == 3:
        algorithms = ['Straight line', 'GA', 'QGA']
    else:
        algorithms = ['GA', 'QGA']
    x_pos = np.arange(len(algorithms))
    fig, ax = plt.subplots()
    ax.bar(x_pos, mean_value_list, yerr=error_list, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms)
    ax.set_title(titel)
    ax.yaxis.grid(True)
    plt.tight_layout()
    #plt.savefig('bar_plot_with_error_bars.png')
    plt.show()
