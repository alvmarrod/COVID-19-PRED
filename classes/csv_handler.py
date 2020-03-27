"""This file provides functions to work with ease with CSV files when
we desire to load/save their data into/from Pandas Dataframes
"""

import pandas as pd

def df_to_csv(file, df, index=False):
  """Writes the provided dataframe to the specified file
  """
  df.to_csv(file, index=False)

def read_csv_as_df(file):
  """Reads the specified file expecting ',' as separator and
  returns it as a Pandas Dataframe
  """
  df = pd.read_csv(file, sep=',', header='infer')
  return df