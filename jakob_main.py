"""
Main file for me to test code of the levels below (because of imports of functions...)
"""

from json import tool
import quantimize.visualisation as vs
import quantimize.data_access as da
import quantimize.air_security as ais
import quantimize.classic.toolbox as toolbox
from quantimize.classic.toolbox import straight_line_solution as sls
import matplotlib.pyplot as plt
import datetime

"""
map = vs.make_map()
for i in range(10):
    cs = vs.draw_flight_path_on_map(map, sls(i,1))
map.colorbar(cs, location="bottom")
plt.show()
"""

list_of_trajectories = []
flight_list = range(10) #[20,30,40]
dt = 15
for i in flight_list:
    list_of_trajectories.append(sls(i,dt))

#vs.animate_flight_path_on_map(list_of_trajectories, dt)

problems = ais.check_safety(list_of_trajectories, dt)
for problem in problems:
    print(problem[0], problem[1][-1], problem[2][-1])

cost_list = []
for flight in list_of_trajectories:
    cost, flight_nr = toolbox.compute_cost(flight, dt)
    cost_list.append((cost, flight_nr))
print(cost_list)
