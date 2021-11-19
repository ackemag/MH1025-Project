import os
import tarfile
import urllib.request
import pandas as pd



DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml2/tree/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):

    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    print("HYEJ", tgz_path)
    print("HOUSING URL: " + housing_url)
    urllib.request.urlretrieve(housing_url, "housing.tgz")

    housing_tgz = tarfile.open("housing.tgz")
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()



def load_housing_data(housing_path=HOUSING_PATH):

    print("HOUSING URL: " + housing_path)

    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

print(housing["ocean_proximity"].value_counts())