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
                 "Last Update", 
                 "Confirmed", 
                 "Deaths", 
                 "Recovered"]

def _gen_cases_df(countries, dates, dfs_dict, case):
  """Generates the cases dataframe for a specific case type

  Outputs
  -----
  A dataframe where columns are dates, rows are countries and cells are cases
  number
  """

  # Create empty df
  out_df = pd.DataFrame(columns=dates, index=countries)
  out_df.fillna(0, inplace=True)

  # Query and insert the data from each dataframe
  print(f"++ Inserting {case} ++")
  for date in dates:

    # Name of the dataframe
    df_name = date + ".csv"

    # Take the dataframe from this day
    day_df = dfs_dict[df_name]
    date_loc = out_df.columns.get_loc(date)

    # Insert the data from this this day into the output dataframe
    # One value per country (existing in the df)
    match = 0
    no_match = 0
    
    # print(f"Insering date {date}")
    for country in day_df.index.to_numpy(dtype=str):
      try:
        if country in out_df.index:
          # print(f"Insert country {country} value {day_df[day_df.index == country][case][0]}")
          # out_df[out_df.index == country][date] = day_df[day_df.index == country][case][0]
          out_df.loc[out_df.index == country, date_loc] = day_df[day_df.index == country][case][0]
          match += 1
        else:
          no_match += 1
          print(f"Have to map: {country}")
          # print(f"Latest list: {out_df.index["Hong" in out_df.index]}")
          for pais in out_df.index:
            if "Hong" in pais:
              print(pais)
          input()
      except Exception as e:
        print("Exception: " + str(e))

    print(f"Matches: {match}\t-\tNo match: {no_match}")

    # print(day_df)

  print(out_df)

  return out_df

def _read_all_covid_raw(folder):
  """Read all the CSV files that it can found in a folder, and returns the
  composed dataframes for each kind of case, altogether in a dictionary
  """

  # Get CSV raw files
  files = glob.glob(folder + "/*.csv")
  dates = []

  # Read each one of them in a Pandas Dataframe
  raw_df_dic = {}
  latest_file = ""
  for file in files:
    try:
      name = file.split("\\")[-1]
      raw_df_dic[name] = read_csv_as_df(file)[covid_columns]
      cols_to_save = [ col for col in covid_columns if "Province" not in col ]
      raw_df_dic[name] = raw_df_dic[name][cols_to_save]
      raw_df_dic[name] = raw_df_dic[name].groupby([covid_columns[1]],
                                                  as_index=False).sum()
      raw_df_dic[name].set_index(covid_columns[1], inplace= True)
      # print(raw_df_dic[name])
      # print(raw_df_dic[name][ raw_df_dic[name].index == "Mainland China" ])

      latest_file = name
      dates.append(name.split(".")[0])
    except Exception as e:
      print(f"Error when processing file: {name}")
      print(str(e))

  # Identify how many countries there are. For it, take the latest file
  # countries = raw_df_dic[latest_file][covid_columns[1]].to_numpy(dtype=str)
  countries = raw_df_dic[latest_file].index.to_numpy(dtype=str)
  # amount = len(countries)

  # Collect the data for each case type in its own dataframe
  cases_df =  {}
  for case in covid_columns[-3:]:
    cases_df[case] = _gen_cases_df(countries, dates, raw_df_dic, case)
    print(cases_df[case])
    #print(cases_df[case].shape)

  return cases_df

def gen_covid19_feat(covid19_dr_url,
                     covid19_raw_url,
                     output_raw="./data/raw/covid/",
                     output_csv="./data/features/covid/covid.csv"):
  """This function receives the URL from the repository where CSV with the
  daily reports of the COVID-19 are stored. Then it downloads them all into the
  specified as RAW folder, and creates an unique CSV as feature output.
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
    if not os.path.isfile(file[0]):
      download_as_csv(file[1], file[0])

  # 2. Create an unique CSV per case with an unique dataframe
  # Confirmed Cases CSV - Row per Country, Column per day
  # Deaths Cases CSV - Row per Country, Column per day
  # Recovered Cases CSV - Row per Country, Column per day

  # Load all files each one in a dataframe
  covid_raw_path = "./data/raw/covid/"
  conf_cases_df, deaths_df, recov_df = _read_all_covid_raw(covid_raw_path)
