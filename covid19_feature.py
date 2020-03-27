"""This script aims to be able to generate the data of the COVID-19 cases and
get it ready to be used as a feature for the models.
"""

import glob
import os.path
import numpy as np
import pandas as pd

from io import StringIO

from classes.csv_handler import read_csv_as_df, df_to_csv

from classes.github_handler import (get_files_from_github_folder,
                                    download_as_csv)

date_format = r'%m-%d-%Y'

# Columns from source - Avoid Latitude and Longitude
covid_columns = ["Province/State", 
                 "Country/Region", 
                 "Lat",
                 "Long"]

def _read_covid_raw(folder, type):
  """Read the COVID raw data from the CSV specified by the type parameter, and
  returns the feature dataframe ready to use by the model
  """

  # Get CSV raw files
  files = glob.glob(folder + "/*.csv")
  raw_df = None

  # Load the specified one
  for file in files:
    if type in file:
      raw_df = read_csv_as_df(file)

  if raw_df is None:
    raise Exception(f"Type {type} not found in raw data")

  # Remove unnecesary columns
  remove = "cols"
  axis = 1 if "cols" in remove else 0
  raw_df = raw_df.drop(covid_columns[0], axis)
  raw_df = raw_df.drop(covid_columns[2], axis)
  raw_df = raw_df.drop(covid_columns[3], axis)

  # Group by country
  raw_df = raw_df.groupby(by=covid_columns[1], as_index=False).sum()

  return raw_df

def gen_covid19_feat(covid19_dr_url,
                     covid19_raw_url,
                     output_raw="./data/raw/covid/",
                     output_csv="./data/features/covid"):
  """This function receives the URL from the repository where CSV with the
  time series reports of the COVID-19 are stored. Then it downloads them all
  into the specified as RAW folder, and creates their respective CSV in the
  features folder but with all data grouped by country.
  """

  # 1. Get the files and save them to Raw data folder

  # Get file names in the repo
  covid_files_list = get_files_from_github_folder(covid19_dr_url)

  # Prepare URLs to download
  covid_url_list = [[output_raw + "/" + file + ".csv", 
                    covid19_raw_url + "/" + file + ".csv"] 
                    for file in covid_files_list]

  # Download
  for file in covid_url_list:
    # If exist, delete the old version so we can have the latest
    if os.path.isfile(file[0]):
      os.remove(file[0])

    # Download it
    download_as_csv(file[1], file[0])

  # 2. Output the clean output dataframe with only usable data
  types = ['confirmed', 'deaths', 'recovered']
  results = []
  
  for case_type in types:

    case_df = _read_covid_raw(output_raw, case_type)
    results.append(case_df)

    output_file = output_csv + f"/{case_type}.csv"

    # 3. Write to features folder
    if os.path.isfile(output_file):
      os.remove(output_file)

    df_to_csv(output_file, case_df)

  return results[0], results[1], results[2]
