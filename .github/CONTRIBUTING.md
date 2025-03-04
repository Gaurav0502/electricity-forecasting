# Contribution guidelines

(This applies only to the collaborators added to this repository)

- Work only in your respective branches and NOT in main.

```bash
git checkout <branch-name>
```
- When using a dataset, do not directly load the raw dataset from data directory. Instead use preprocess.py script.

```py
import preprocess # will create a folder __pycache__ (ignored by git)

# for EDA
electricity = preprocess.preprocess("electricity")
temp = preprocess.preprocess("temperature")
precip = preprocess.preprocess("precipitation")

# for clustering
electricity = preprocess.preprocess_modelling(as_numpy = True, clustered = False)

# for modelling 
# (set as_numpy = True if model needs numpy arrays as input)
# (otherwise dataframe will be returned)
electricity = preprocess.preprocess_modelling(as_numpy = False, clustered = True)
temp = preprocess.preprocess("temperature")
precip = preprocess.preprocess("precipitation")
```
Any other input to the function will result in ValueError!

- If you use any new module from Python, please add it inside requirements.txt.


# Guidelines for pull request
Before making a pull request, ensure the following requirements are fulfilled:

- Pull the code from main to the respective branch.

```bash
git pull origin main
```

- The dataset(s) are populated under the data directory (will be ignored by Git). Due to the size of the dataset, no data files must be pushed into your respective branch.

- No documents must be pushed to Github (.pptx, .pdf, .docx, etc). If you have any documents locally, populate them into the documents directory (will ignored by Git). The documents will be added as google drive links in the README.md file.

- The overall directory structure must be as follows:

```bash
.
├── README.md
├── data
│   ├── LD2011_2014.txt
│   ├── lisbon_precip_2011-2014.csv
│   └── lisbon_temp_2011-2014.csv
├── eda.ipynb
├── preprocess.py
├── requirements.txt
```
