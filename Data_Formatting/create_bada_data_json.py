import simplejson
import pandas as pd

data = {}
base_dir = "C:/Users/Jakob/Documents/TUM/Master/4.Semester-QST/Quantum-Challenge/Data/"
df = pd.read_csv(base_dir+"bada_data.csv", sep=";", header=7)
for key,value in df.iterrows():
    data[value["FL"]] = {"CRUISE":{},"CLIMB":{},"DESCENT":{}}
    data[value["FL"]]["CRUISE"]["TAS"] = value["TAS [kts]"]
    data[value["FL"]]["CRUISE"]["fuel"] = float(value["fuel [kg/min]"].replace(",","."))
    data[value["FL"]]["CLIMB"]["TAS"] = value["TAS [kts].1"]
    data[value["FL"]]["CLIMB"]["fuel"] = float(value["fuel [kg/min].1"].replace(",","."))
    data[value["FL"]]["CLIMB"]["ROC"] = value["ROC [ft/min]"]
    data[value["FL"]]["DESCENT"]["TAS"] = value["TAS [kts].2"]
    data[value["FL"]]["DESCENT"]["fuel"] = float(value["fuel [kg/min].2"].replace(",","."))
    data[value["FL"]]["DESCENT"]["ROC"] = value["ROC [ft/min]"]

with open("bada_data.json","wb") as f:
    f.write(simplejson.dumps(data, ignore_nan=True).encode("utf-8"))