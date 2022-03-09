import simplejson
import pandas as pd

data = {}
base_dir = "C:/Users/Jakob/Documents/TUM/Master/4.Semester-QST/Quantum-Challenge/Data/"
df = pd.read_csv(base_dir+"bada_data.csv", sep=";", header=7, decimal=",")
for key,value in df.iterrows():
    if value["FL"] >= 100:
        data[int(value["FL"])] = {"CRUISE":{},"CLIMB":{},"DESCENT":{}}
        data[int(value["FL"])]["CRUISE"]["TAS"] = value["TAS [kts]"]
        data[int(value["FL"])]["CRUISE"]["fuel"] = value["fuel [kg/min]"]
        data[int(value["FL"])]["CLIMB"]["TAS"] = value["TAS [kts].1"]
        data[int(value["FL"])]["CLIMB"]["fuel"] = value["fuel [kg/min].1"]
        data[int(value["FL"])]["CLIMB"]["ROC"] = value["ROC [ft/min]"]
        data[int(value["FL"])]["DESCENT"]["TAS"] = value["TAS [kts].2"]
        data[int(value["FL"])]["DESCENT"]["fuel"] = value["fuel [kg/min].2"]

with open(base_dir+"bada_data.json","wb") as f:
    f.write(simplejson.dumps(data, ignore_nan=True).encode("utf-8"))