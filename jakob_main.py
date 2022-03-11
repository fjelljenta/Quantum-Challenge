"""
Main file for me to test code of the levels below (because of imports of functions...)
"""

from quantimize.visualisation import draw_flight_path_on_map as fpm, make_map as mm
from quantimize.classic.toolbox import straight_line_solution as sls
import matplotlib.pyplot as plt

map = mm()
for i in range(10):
    fpm(map,sls(i,5))
plt.show()