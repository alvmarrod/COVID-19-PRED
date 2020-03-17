import numpy as np
import pandas as pd

if __name__ == "__main__":
  from classes.github_parser import github_parser
else:
  from utilities.classes.github_parser import github_parser

import requests

from io import StringIO

# Column order imposed by the data source of the COVID-19
covid_columns = ["Province/State", 
                 "Country/Region", 
                 "Last Update", 
                 "Confirmed", 
                 "Deaths", 
                 "Recovered"]

covid_data_folder= r"https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports"

date_format = r'%m-%d-%Y'

covid_raw_base_url = r"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/"

population_csv_columns = ["Rank",
                          "name",
                          "pop2019",
                          "pop2018",
                          "GrowthRate",
                          "area",
                          "Density"]

def write_to_csv(file, df):
  df.to_csv(file, index=False)

def parse_github_folder(url):
  """Parses the github folder passed by URL and returns the latest file name
  """

  r = requests.get(url, auth=('', ''))
  html = r.text

  # Parse it and check the latest file by it's name
  parser = github_parser()
  parser.feed(html)

  return parser.latest_file

def get_locations_df(url):
  """Returns a pd.Dataframe with the Countries from the file
  specified by the given URL
  """

  # Get the file from the URL
  r = requests.get(url, auth=('', ''))
  html = r.text

  # Read it to a pd.dataframe
  df = pd.read_csv(StringIO(html), header='infer')

  # Country is the column 1
  cols = [covid_columns[1]]

  return df[cols]

def csv_as_df(file):

  df = pd.read_csv(file, sep=',', header='infer')
  return df

def add_population_data(df):
  """Enrichs the given pd.Dataframe with population and population density that
  can be retrieved from the Population CSV
  """

  # Load the data from Population CSV
  population_df = csv_as_df(r'.\data\population.csv')

  # Clean the population DF and leave only Name, Pop2019 and Density
  
  population_df = population_df[[
    population_csv_columns[1],
    population_csv_columns[2],
    population_csv_columns[6]
  ]]

  # Rename Name column to match covid dataframe for merge
  df.rename(columns={
    covid_columns[1]: "Country"
  }, inplace=True)

  population_df.rename(columns={
    population_csv_columns[1]: "Country",
    population_csv_columns[2]: "Population",
    }, inplace=True)
  

  df = pd.merge(df, population_df, on="Country", how='inner').drop_duplicates()

  # print(df["Population"][df["Country"]=="Pakistan"])
  return df

def latest_locations_with_popden():

  # Look for the latest available file in the repo
  latest_covid_file_url = parse_github_folder(covid_data_folder)

  download_url = covid_raw_base_url + "/" + latest_covid_file_url.strftime(date_format + ".csv")
  covid_locations = get_locations_df(download_url)

  # print("Recovering population data per country...")
  covid_locations = add_population_data(covid_locations)

  return covid_locations

if __name__ == "__main__":

  covid_locations = latest_locations_with_popden()

  # Save the data to a file.
  write_to_csv("./data/population_needed.csv", covid_locations)