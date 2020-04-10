"""This file aims to create the input for the model
"""

import numpy as np
import pandas as pd

import datetime as dt

def expand_cases_to_vector(data):
  """Receives the cases dataframe and expands it, creating a row per each day
  time lapse.
  """

  dates_num = len(data.columns) - 1
  dates = data.columns[1:].tolist()
  countries = data["Country"].unique().tolist()

  #print(f"Total countries: {len(countries)}")
  #print(f"Total Dates: {dates_num}")

  # Empty template to fill with data
  rows = (dates_num-1) * len(countries)

  output = pd.DataFrame( \
                      columns=[
                      "Country",
                      "Date",
                      "Cases",
                      "Popden",
                      "Masks",
                      "Poprisk",
                      "Lockdown",
                      "Borders",
                      "NextDay"
                    ])

  for j in range(len(countries)):

    # Cases data from the country row
    country_data = data[ data["Country"] == countries[j] ].values.tolist()
    country_data = country_data[0][1:]

    for i in range(dates_num-1):

      output.loc[len(output)] = [countries[j],
                                 dates[i],
                                 country_data[i], 
                                 0, 0, 0, 0, 0, 
                                 country_data[i+1]]

  return output

def split_by_date(df, limit_date):
  """Splits the dataframe in two new dataframes based on the
  specified limit date in format mm/dd/yy
  """

  mask = [dt.datetime.strptime(date, r"%m/%d/%y") <= \
          dt.datetime.strptime(limit_date, r"%m/%d/%y") \
          for date in df["Date"].values.tolist()]
  inverse = [not item for item in mask]

  before = df.loc[mask]
  after = df.loc[inverse]

  return before, after

def fill_df_column_with_value(df, colname, source_df, source_col, default):
  """Takes the specified column from the source dataframe and inserts its data
  wisely in the specified column of the target dataframe. It admits a default
  value in case the value doesn't exist in the source.

  Wisely stands for based on the "Country" value of both dataframes.
  """

  countries = df["Country"].unique().tolist()

  for country in countries:

    #a, b, c = colname, country, source_df.loc[ source_df["Country"] == country, source_col].values
    #print(f"Feature {a} inserting country {b} value {c}")
    if country in source_df["Country"].tolist():
      df.loc[ df["Country"] == country, colname ] = \
            source_df.loc[ source_df["Country"] == country, source_col].values
    else:
      df.loc[ df["Country"] == country, colname ] = default

  # return df

def fill_df_column_date_based(df, colname, source_df, default):
  """Takes the specified column from the source dataframe and inserts its data
  wisely in the specified column of the target dataframe. It admits a default
  value in case the value doesn't exist in the source.

  Wisely stands for based on the "Country" value of both dataframes, and
  matching the dates in source and target.

  For those dates where no value is specified in the source, the latest value
  is kept.
  """

  countries = df["Country"].unique().tolist()

  for country in countries:

    if country in source_df["Country"].tolist():

      data = source_df.loc[ source_df["Country"] == country].T.values
      # Extend the latest value up to the lenght needed
      lenght_difference = len(df[df["Country"] == country]) - len(data)

      #from IPython import embed
      #embed()

      # Extend
      data = data.tolist() + [data[-1].tolist()] * (lenght_difference+1)
      # If we have further data than needed, should cut
      if (lenght_difference+1) < 0:
        data = data[:(lenght_difference+1)]
      
      # Remove the country name at the beginning
      data = data[1:]
      
      df.loc[ df["Country"] == country, colname ] = data
            
    else:

      df.loc[ df["Country"] == country, colname ] = default

def clean_invalid_data(df):
  """Removes those samples in the dataframe where current and next day cases
  are 0
  """
  
  # Removing samples with 0 to 0 cases
  mask = (df["NextDay"]!=0)
  df = df.loc[mask]

  return df

def normalize_data(df, bias=0):
  """Normalizes the data from numeric columns in the dataframe. Allows
  to set a bias to add after normalization.

  Takes into account that Cases and NextDay should be normalized by the
  same max value.
  """

  same = ["Cases", "NextDay"]
  same_max = 0
  
  for col in df.columns:
    if np.issubdtype(df[col].dtype, np.number):
      if col in same:
        same_max = max(same_max, df[col].max())
      else:
        max_val = df[col].max()
        df.loc[:, col] = (df.loc[:, col] / max_val) + bias

  # Normalize the cases cols
  col = same[0]
  df.loc[:, col] = (df.loc[:, col] / same_max) + bias
  col = same[1]
  df.loc[:, col] = (df.loc[:, col] / same_max) + bias

  return df