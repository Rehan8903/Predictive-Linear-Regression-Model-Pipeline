from sklearn.datasets import fetch_california_housing
import pandas as pd

def load_data():
    housing = fetch_california_housing()

    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["MedHouseValue"] = housing.target

    return df