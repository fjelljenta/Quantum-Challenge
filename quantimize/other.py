"""
File with testing or onetime code
"""
"""
Find the max and min value of the atmospheric data
import quantimize.data_access as da
import datetime

##########################################################

data_min_max = "-0.047278133046295995 0.34942841580572637"

##########################################################

timelist = [datetime.time(6), datetime.time(12), datetime.time(18)]
min_value = 1
max_value = 0
count = 0
while count <= 10:
    for i in range(-30,31,2):
        for j in range(34,61,2):
            for k in range(100,400,20):
                for l in timelist:
                    merged_info = da.get_merged_atmo_data(i,j,k,l)
                    if merged_info < min_value:
                        min_value = merged_info
                    if merged_info > max_value:
                        max_value = merged_info
    print(count)
    count += 1

print(min_value, max_value)
"""