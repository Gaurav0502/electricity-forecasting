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

<b><u>Note</u></b>: For all the code in this repository to execute correctly, the data directory must include three CSV files namely: ```LD2011_2014.txt```, ```lisbon_precip_2011-2014.csv```, and ```lisbon_temp_2011-2014.csv```.

## Documentation
1. ```preprocess.py```:
    - ```preprocess(dataset_name: str) -> DataFrame```: performs <b>preliminary</b> preprocessing on the ```dataset_name```. The ```dataset_name``` must be one among the following: ```["electricity", "temperature", "precipitation"]```.

    - ```preprocess_modelling(as_numpy: bool, clustered: bool, stationary: bool) -> DataFrame/NumPy array```: performs the preprocessing required for modelling for <b>electricity</b> dataset before modelling. 

        - ```as_numpy```: if set to ```true``` returns the data in numpy format (required for LSTM and tslearn - clustering)
        - ```clustered```: if set to ```true```, it clusters the data and provides the clustered data as output.
        - ```stationary```: if set to ```true```, it returns the first order differencing of the time-series
    - ```preprocess_electricity() -> DataFrame``` and ```preprocess_weather(dataset_name: str) -> DataFrame``` are helper functions that used by ```preprocess(dataset_name: str) -> DataFrame```.
2. ```model.py```:

    - ```Model.standardize(self, train, test) -> StandardScaler object, train DataFrame, test DataFrame```: standardizes the train and test data based on the statistical properties of the train dataset

    - ```Model.destandardize(self, data_st, scaler) -> NumPy array```: destandardizes ```data_st``` based on the ```scaler.inverse_tranform```.

    - ```Model.cross_validate(self) -> None```: Cross validates based on the train-test split window and logs the forecasts, train data indices, test data indices, and MAPE with the test data.

    - ```Model.mape_boxplot_by_client(self, num_clients: int) -> None```: plots the MAPE boxplots at client level considering the prediction of the model (on clustered data) as the prediction for each client.

    - ```Model.mape_boxplot_by_step(self, models: List[Models])-> None```: plots the aggregated MAPE boxplots for each forecast.

    - ```train_model(self, train, train_idx) -> SARIMAX/Prophet/TensorFlow Model object``` and ```get_forecasts(self, test, test_idx) -> NumPy array/DataFrame``` are abstract methods for model training and getting the forecasts defined by the respective model that inherits the ```Model``` class.

## Hyperparameters (for reproducing results)

1. SARIMA:

```py

param = {
    "order": (1, 1, 0), 
    "seasonal_order": (1, 1, 0, 12)
}

# No MA component because it was statistically insignificant.

```

2. Facebook Prophet:

```py

# No hyperparameter as such

# Add exogenous variables and country holidays for Portugal

```

3. LSTM:

```py

params = {
    "num_units": 60,
    "activation_function": 'relu',
    "optimizer": "adam",
    "loss_function": "mse",
    "batch_size": 32,
    "num_epochs": 300
}

# with EarlyStopping.
# evaluation is done using MAPE only.

```

## Additional Links

### Technical Report

Link to report and slides: https://drive.google.com/drive/folders/1v5LcC3QKL5l75ygeN_MewOSIYVWC7Viq?usp=sharing

### Data Sources

Electricity: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

Weather: https://www.ipma.pt/en/oclima/series.longas/list.jsp
