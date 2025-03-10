import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
import matplotlib.pyplot as plt
import json

LIST_TYPE = "<class 'list'>"

def plot_mape_boxplots(model: list, mape: dict):

    if not list(mape.keys()) == ["cluster_0", "cluster_1"]:
        raise ValueError("the dictionary input must have cluster_0 and cluster_1 as keys.")
    
    if  not str(type(mape["cluster_0"])) == str(type(mape["cluster_1"])) == LIST_TYPE:
        raise ValueError("dictionary values must a list of MAPE values.")
    
    with open("model-results", "w") as f: 
        json.dump(mape, f, indent = 4)
    
    labels = list(mape.keys())
    values = list(mape.values()) 

    sns.boxplot(data = values)
    plt.xticks(ticks = range(len(labels)), labels = labels)
    plt.xlabel("Consumer Clusters (based  on electricity consumption)")
    plt.ylabel("MAPE score")
    plt.title(f"Variation in performance of {model} for different train-test splits")

    
    