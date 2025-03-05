import pandas as pd
import numpy as np
import json

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
                                 "Month": "month",
                                 "Day": "day",
                                 "Precipitation (mm)": "precip"})

    # filter data for only 2011-2014
    df = df[df["year"].isin(YEARS)].reset_index(drop = True)

    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.drop(["year", "month", "day"], axis = 1)

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

def preprocess_modelling(as_numpy: bool, clustered: bool, stationary: bool ):

    # reading the dataset
    electricity = preprocess("electricity")

    # making timestamp as datetime and getting relevant columns
    electricity["timestamp"] = pd.to_datetime(electricity["timestamp"])
    cols = [i for i in electricity.columns if i != "timestamp"]

    # finding consumer added in 2013 and 2014
    x = (electricity[cols] != float(0)).idxmax()
    y = pd.DataFrame(electricity.loc[x, "timestamp"]).reset_index()
    x = pd.DataFrame(x).reset_index().rename(columns = {0:"index","index":"consumer"})
    df = y.drop_duplicates().merge(x, how = "inner", on = "index")
    df = df[df["timestamp"].dt.year.isin([2013, 2014])] 

    # removing consumers added in 2013 and 2014
    consumers_13_14 = df["consumer"].values
    electricity = electricity.drop(consumers_13_14, axis = 1)

    # aggregating electricity consumption to compute daily
    # consumption
    electricity["date"] = electricity["timestamp"].dt.date
    electricity = electricity.drop(["timestamp"], axis = 1)
    electricity = electricity.groupby(by = ["date"]).sum()

    if clustered:
        with open("clusters.json", "r") as f:
            clusters = json.load(fp = f)
            cluster_0 = clusters["0"]
            cluster_1 = clusters["1"]
            x = cluster_1

            electricity["cluster_0"] = electricity[cluster_0].sum(axis = 1)
            electricity["cluster_1"] = electricity[cluster_1].sum(axis = 1)
            electricity = electricity.drop(cluster_0 + cluster_1, axis = 1)
            if stationary:
                electricity["cluster_0"] = electricity["cluster_0"].diff()
                electricity["cluster_1"] = electricity["cluster_1"].diff()
            

    # converting to numpy format
    # for tslearn for clustering
    if as_numpy == True:
        a = electricity.values.T
        a = np.expand_dims(a, axis = -1)
        return a
    
    return electricity