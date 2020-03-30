"""This script takes the data about Closing Borders and gets it ready to be
used as a feature for the models.
"""

import os
import pandas as pd

from classes.csv_handler import read_csv_as_df, df_to_csv

def _read_borders_raw(raw_csv):
  """Read the raw data from the CSV specified by the parameter, and
  returns the feature dataframe.
  """

  # Load the specified one
  raw_df = read_csv_as_df(raw_csv)

  if raw_df is None:
    raise Exception(f"Raw data not found in file {raw_csv}")

  return raw_df

def gen_borders_feat(input_raw="./data/raw/govme/borders.csv",
                      output_folder="./data/features/govme"):
  """This function process the raw CSV file with the Borders data and creates
  the output CSV file with the format ready for usage by the model.
  """

  # 1. Get the data from the raw CSV file
  borders_raw_df = _read_borders_raw(input_raw)

  # 2. Apply transformations. Not needed in this feature.
  borders_feat = borders_raw_df

  output_csv = output_folder + "/borders.csv"

  # 3. Write to features folder
  if os.path.isfile(output_csv):
    os.remove(output_csv)

  df_to_csv(output_csv, borders_feat)

  return borders_feat