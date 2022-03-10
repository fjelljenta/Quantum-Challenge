import pandas as pd
import json
import simplejson
import os
import xarray as xr

def create_flight_schedule():
    """Creates the flight schedule as json file
    """
    data = {}
    base_dir = os.getcwd()
    df = pd.read_csv(base_dir+"/data/flights.csv", sep=";")
    for key, value in df.iterrows():
        data[value["flight_number"]] = {
            "start_time":value["start_time"], 
            "start_flightlevel":value["start_flightlevel"],
            "start_longitudinal":value["start_longitudinal"], 
            "start_latitudinal":value["start_latitudinal"], 
            "end_longitudinal":value["end_longitudinal"], 
            "end_latitudinal":value["end_latitudinal"]
        }

    with open(base_dir+"/data/flights.json", "wb") as f:
        f.write(json.dumps(data).encode("utf-8"))

def create_bada_data():
    """Creates the flight level information as json file
    """
    data = {}
    base_dir = os.getcwd()
    df = pd.read_csv(base_dir+"/data/bada_data.csv", sep=";", header=7, decimal=",")
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

    with open(base_dir+"/data/bada_data.json","wb") as f:
        f.write(simplejson.dumps(data, ignore_nan=True).encode("utf-8"))

def convert_hPa_to_FL(hPA):
    """Converts hPa to flight level according to given PDF

    Args:
        hPA (int): Flight level in hPA

    Returns:
        int: Flight level in FL
    """
    if hPA == 600:
        return 140
    elif hPA == 550:
        return 160
    elif hPA == 500:
        return 180
    elif hPA == 450:
        return 200
    elif hPA == 400:
        return 240
    elif hPA == 350:
        return 260
    elif hPA == 300:
        return 300
    elif hPA == 250:
        return 340
    elif hPA == 225:
        return 360
    elif hPA == 200:
        return 380
    else:
        return 0

def create_atmo_data():
    """Converts the atmospheric data to json file
    """
    data = {}
    base_dir = os.getcwd()
    ds = xr.open_dataset(base_dir+"/data/aCCF_0623_p_spec.nc")
    df = ds.to_dataframe()

    for key, value in df.iterrows():
        FL = convert_hPa_to_FL(key[2])
        if FL != 0 and key[0]>=-30:
            if key[0] in data:
                if key[1] in data[key[0]]:
                    if FL in data[key[0]][key[1]]:
                        data[key[0]][key[1]][FL][str(key[4]).split(" ")[1]] = {"MERGED":value["MERGED"]}
                    else:
                        data[key[0]][key[1]][FL] = {}
                        data[key[0]][key[1]][FL][str(key[4]).split(" ")[1]] = {"MERGED":value["MERGED"]}
                else:
                    data[key[0]][key[1]] = {}
                    data[key[0]][key[1]][FL] = {}
                    data[key[0]][key[1]][FL][str(key[4]).split(" ")[1]] = {"MERGED":value["MERGED"]}
            else:
                data[key[0]] = {}
                data[key[0]][key[1]] = {}
                data[key[0]][key[1]][FL] = {}
                data[key[0]][key[1]][FL][str(key[4]).split(" ")[1]] = {"MERGED":value["MERGED"]}
        
    with open(base_dir+"/data/atmo.json", "wb") as f:
        f.write(json.dumps(data).encode("utf-8"))

if __name__ == "__main__":
    while True:
        print("Which data should be converted?")
        print("1) Flight schedule")
        print("2) Flight level info")
        print("3) Atmospheric data")
        print("4) Exit")
        menu = input("Please choose: ")
        if menu == "1":
            create_flight_schedule()
        elif menu == "2":
            create_bada_data()
        elif menu == "3":
            create_atmo_data()
        elif menu == "4":
            break
        else:
            print("Wrong input")