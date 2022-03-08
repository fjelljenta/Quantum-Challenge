import json
import pandas as pd

data = {}
base_dir = "C:/Users/Jakob/Documents/TUM/Master/4.Semester-QST/Quantum-Challenge/Data/"
df = pd.read_csv(base_dir+"flights.csv", sep=";")
for key, value in df.iterrows():
    data[value["flight_number"]] = {
        "start_time":value["start_time"], 
        "start_flightlevel":value["start_flightlevel"],
        "start_longitudinal":value["start_longitudinal"], 
        "start_latitudinal":value["start_latitudinal"], 
        "end_longitudinal":value["end_longitudinal"], 
        "end_latitudinal":value["end_latitudinal"]
    }

with open("flights.json", "wb") as f:
    f.write(json.dumps(data).encode("utf-8"))