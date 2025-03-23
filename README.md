# Daily Electricity Usage Forecasting

## Aim
To forecast the daily electricity consumption with weather data (daily temperature ranges and precipitation).

## Environment Setup

- Clone this repository.

```bash

git clone https://github.com/Gaurav0502/electricity-forecasting.git

```

- Install all packages in the ```requirements.txt``` file.

```bash

pip install -r requirements.txt

```

- Download and store all the three datasets from following sources:

1. Electricity dataset: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

2. Weather dataset: https://www.kaggle.com/datasets/gauravpendharkar/portuguese-weather-data-from-2011-to-2014

- Ensure the all the data and code files have the directory structure as follows:

```bash
.
├── README.md
├── clustering.ipynb
├── clusters.json
├── data
│   ├── LD2011_2014.txt
│   ├── lisbon_precip_2011-2014.csv
│   └── lisbon_temp_2011-2014.csv
├── eda.ipynb
├── model.py
├── modelling.ipynb
├── preprocess.py
└── requirements.txt

```

The next section explains each of the files.

## File Description
1. ``preprocess.py``: This file contains the code to preprocess the electricity and weather data before exploratory data analysis, clustering, and modelling.

2. ``cluster.json``: This file contains metadata about the cluster (i.e. the clients belonging to the cluster)

3. ```model.py```: This file contains a generic template for a model and its evaluation. All trained models inherit the ```Model``` class and define the abstract functions ```train_model(self, train, train_idx)``` and ```get_forecasts(self, test, test_idx)``` because every models has different formats for training and testing data.

4. ```clustering.ipynb```: This notebook contains all the code required for clustering the electricity time-series.

5. ```eda.ipynb```: This notebook contains some preliminary analysis of the electricity and weather datasets.

6. ```modelling.ipynb```: This notebook has the code for model building and evaluation.

<b><u>Note</u></b>: For all the code in this repository to execute correctly, the data directory must be include three CSV files namely: ```LD2011_2014.txt```, ```lisbon_precip_2011-2014.csv```, and ```lisbon_temp_2011-2014.csv```.

## Additional Links

### Technical Report

Link to report and slides: https://drive.google.com/drive/folders/1v5LcC3QKL5l75ygeN_MewOSIYVWC7Viq?usp=sharing

### Data Sources

Electricity: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

Weather: https://www.ipma.pt/en/oclima/series.longas/list.jsp
