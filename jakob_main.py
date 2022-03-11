"""
Main file for me to test code of the levels below (because of imports of functions...)
"""

import quantimize.visualisation as vs
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
dt = 1
for i in range(10):
    list_of_trajectories.append(sls(i,dt))

vs.animate_flight_path_on_map(list_of_trajectories, dt)