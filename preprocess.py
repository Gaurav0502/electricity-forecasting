import pandas as pd
import numpy as np

DATA_DIR = {
  "electricity": "data/LD2011_2014.txt",
  "temperature": "data/lisbon_temp_2011-2014.csv",
  "precipitation": "data/lisbon_precip_2011-2014.csv"
}

YEARS = [2011, 2012, 2013, 2014]

# replaces comma with dot
def str2float(x: str):
    return float(str(x).replace(",", "."))

def preprocess_electricity():

    df = pd.read_csv(DATA_DIR["electricity"], sep=";")

    # giving a name to the time column
    df = df.rename({"Unnamed: 0": "timestamp"}, axis = 1)

    # converting object type cols to float64
    cols = [i for i in df.columns if i != "timestamp"]
    for col in cols:
        df[col] = df[col].apply(str2float)
    
    return df

def preprocess_weather(dataset_name: str):

    df = pd.read_csv(DATA_DIR[dataset_name])

    if dataset_name == "precipitation":
       df = df.rename(columns = {"Year": "year",
                                 "Precipitation (mm)": "precip"})

    # filter data for only 2011-2014
    df = df[df["year"].isin(YEARS)]

    return df

# preprocess data
def preprocess(dataset_name: str):

    valid_datasets = ["electricity", "temperature", "precipitation"]

    if dataset_name not in valid_datasets:
        raise ValueError(f"The input dataset_name can only take the following values: {valid_datasets}")

    if dataset_name == "electricity":
        return preprocess_electricity()

    else:
        return preprocess_weather(dataset_name)
    
    return df