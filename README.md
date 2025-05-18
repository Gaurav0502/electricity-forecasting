<div align="center">
<h1>Daily Electricity Usage Forecasting</h1>
</div>

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

### References

Electricity: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

Weather: https://www.ipma.pt/en/oclima/series.longas/list.jsp
