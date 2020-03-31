"""This script takes the data about Population Density and gets it ready to be
used as a feature for the models.
"""

import os
import numpy as np
import pandas as pd

from io import StringIO
from sklearn.cluster import KMeans

from classes.plt_handler import plot_clustering
from classes.csv_handler import read_csv_as_df, df_to_csv

popden_raw_cols = ["Rank",
                   "name",
                   "pop2019",
                   "pop2018",
                   "GrowthRate",
                   "area",
                   "Density"]

def _read_popden_raw(raw_csv, handicaps=None):
  """Read the raw data from the CSV specified by the type parameter, and
  returns the feature dataframe ready removing undesired columns. Results are
  rounded to multiples of 5 and are applied a handicap.
  """

  # Load the specified one
  raw_df = read_csv_as_df(raw_csv)

  if raw_df is None:
    raise Exception(f"Raw data not found in file {raw_csv}")

  # Remove unnecesary columns
  remove = "cols"
  axis = 1 if "cols" in remove else 0
  raw_df = raw_df.drop(popden_raw_cols[0], axis)
  raw_df = raw_df.drop(popden_raw_cols[2], axis)
  raw_df = raw_df.drop(popden_raw_cols[3], axis)
  raw_df = raw_df.drop(popden_raw_cols[4], axis)
  raw_df = raw_df.drop(popden_raw_cols[5], axis)

  # Apply existing handicaps
  name = popden_raw_cols[1]
  col = popden_raw_cols[-1]
  if not handicaps is None:
    for k in handicaps:
      indexes = raw_df[name][raw_df[name] == k].index.tolist()
      for index in indexes:
        raw_df.loc[index, col] = raw_df.loc[index, col] * handicaps[k]

  # Make the results to be multiple of 5 and at least 5
  if np.issubdtype(raw_df[col].dtype, np.number):
    raw_df[col] = (raw_df[col] / 5).round() * 5
    # Make sure it is at least 5 and not 0
    raw_df[col] = raw_df[col] + (raw_df[col] == 0) * 5

  return raw_df

def _apply_classification(popden_df, K=3):
  """Applies K-Means to the population density dataframe that is passed and
  returns a dictionary with the clusters.

  It also stores the output picture of the classification.
  """

  # Convert the dataframe to an array
  col = popden_raw_cols[-1]
  popden_array = popden_df[col].to_numpy(dtype=float)

  data_2_feed = []
  for k in popden_array:
    data_2_feed.append([k, 0])

  # Apply K-Means
  kmeans = KMeans(n_clusters=K, random_state=0).fit(data_2_feed)

  clusters = {}
  for label in kmeans.labels_:
    clusters[label] = \
      [data_2_feed[i][0] for i in range(0, len(data_2_feed)) if kmeans.labels_[i] == label]

  return clusters


def gen_popden_feat(input_raw="./data/raw/popden/",
                    output_folder="./data/features/popden",
                    handicaps=None,
                    remove_over=False):
  """This function process the raw CSV file with all the world's population
  density and creates the output CSV file with the classification of each
  country based on its poulation density.

  It also allows to apply a specific correction factor to countries based on
  a dictionary passed as argument.
  """

  # 1. Get the data from the raw CSV file
  # Results are already multiple of 5, we apply the handicaps
  popden_raw_df = _read_popden_raw(input_raw, handicaps)

  # 2. We apply the classification
  if remove_over:
    den = popden_raw_cols[-1]
    name = popden_raw_cols[1]
    indexes = popden_raw_df[popden_raw_df[den] < 10000][name].index.tolist()
    classification_df = popden_raw_df.iloc[indexes]
  else:
    classification_df = popden_raw_df
  clusters = _apply_classification(classification_df, K=3)

  # Generate the CSV output with the classification
  # Work with the original dataframe
  popden_feat_df = popden_raw_df.copy(deep=False)

  # Draw it
  draw_title = "Population Density Feature - Clustering by K-Means"
  output_png = output_folder + "/popden.png"

  frontiers = plot_clustering(clusters, draw_title, "Density", output_png)

  # Now we generate the final classification. We take advantage from frontiers
  # calculated when plotting
  
  # Generate new classification
  col = popden_raw_cols[-1]
  popden_feat_df["Classification"] = [3 if density >= frontiers[1] else \
                                      2 if density >= frontiers[0] else \
                                      1 for density in popden_feat_df[col]]
  # We can drop density already
  remove = "cols"
  axis = 1 if "cols" in remove else 0
  #print(popden_feat_df)
  #from IPython import embed
  #embed()
  popden_feat_df = popden_feat_df.drop(col, axis)

  # Rename column to "Country"
  popden_feat_df.rename(columns={
    popden_raw_cols[1]: "Country"
  }, inplace=True)

  output_csv = output_folder + "/popden.csv"

  # 3. Write to features folder
  if os.path.isfile(output_csv):
    os.remove(output_csv)

  df_to_csv(output_csv, popden_feat_df)

  return popden_feat_df