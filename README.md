## Getting Started

### Running the app locally

First create a virtual environment with conda or venv inside a temp folder, then activate it.

```
virtualenv venv

# Windows
venv\Scripts\activate
# Or Linux
source venv/bin/activate

```

Clone the git repo, then install the requirements with pip

```

git clone https://github.com/alssalo/Salary-Prediction-deployment
cd Salary-Prediction-deployment
pip install -r requirements.txt

```

Run the app

```

python app.py

```

## About the app

The purpose of this Dash app is to predict the salary of an employee for the given inputs and visualize the trends.

## Built With

- [Dash](https://dash.plot.ly/) - Main server and interactive components
- [Plotly Python](https://plot.ly/python/) - Used to create the interactive plots
