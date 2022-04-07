"""
Main file for me to test code of the levels below (because of imports of functions...)
"""

from json import tool
import quantimize.visualisation as vs
import quantimize.converter as cv
import quantimize.data_access as da
import quantimize.air_security as ais
import quantimize.classic.toolbox as toolbox
import quantimize.classic.classic_solution as cs
from quantimize.classic.toolbox import straight_line_trajectory as sls
import matplotlib.pyplot as plt
import datetime


#map = vs.make_map()


"""
problems = ais.check_safety(list_of_trajectories, dt)
for problem in problems:
    print(problem[0], problem[1][-1], problem[2][-1])

cost_list = []
for flight in list_of_trajectories:
    cost, flight_nr = toolbox.compute_cost(flight, dt)
    cost_list.append((cost, flight_nr))
print(cost_list)
"""
ax, m = vs.make_3d_map()
for i in range(10):
    report, solution, trajectory = cs.run_genetic_algorithm(i)
    ax, m = vs.plot_flight_path_on_map_3d(ax, m, trajectory)
plt.show()
