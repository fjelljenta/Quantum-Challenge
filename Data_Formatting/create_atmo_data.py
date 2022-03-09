import json
import xarray as xr

data = {}
base_dir = "C:/Users/Jakob/Documents/TUM/Master/4.Semester-QST/Quantum-Challenge/Data/"

def convert_hPa_to_FL(hPA):
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

ds = xr.open_dataset(base_dir+"aCCF_0623_p_spec.nc")
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
    
with open(base_dir+"atmo.json", "wb") as f:
    f.write(json.dumps(data).encode("utf-8"))