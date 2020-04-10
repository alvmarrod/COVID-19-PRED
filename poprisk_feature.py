"""This script takes the data about Population Risk Weighted and classifies it
in low (1), medium (2) or high (3) risk by applying a K-Means algorithm.
"""

import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from classes.plt_handler import plot_clustering
from classes.csv_handler import (df_to_csv, read_csv_as_df)

COL_WR = "Weighted Risk"

def _apply_classification(poprisk_df, K=3):
  """Applies K-Means to the population risk dataframe that is passed and
  returns a dictionary with the clusters.

  It also stores the output picture of the classification.
  """

  # Convert the dataframe to an array
  poprisk_array = poprisk_df[COL_WR].to_numpy(dtype=float)

  data_2_feed = []
  for k in poprisk_array:
    data_2_feed.append([k, 0])

  # Apply K-Means
  kmeans = KMeans(n_clusters=K, random_state=0).fit(data_2_feed)

  clusters = {}
  for label in kmeans.labels_:
    clusters[label] = \
      [data_2_feed[i][0] for i in range(0, len(data_2_feed)) if kmeans.labels_[i] == label]

  return clusters

def gen_poprisk_feat(input_raw,
                     output_folder="./data/features/poprisk"):
  """This function process the raw CSV file with all our hand-baked population
  risk rates and creates the output CSV file with the classification of each
  country based on it.
  """

  # 1. Get the data from the raw CSV file
  poprisk_raw_df = read_csv_as_df(input_raw)

  # Convert the dataframe to an array
  poprisk_raw_df[COL_WR].replace(0, 0.5, inplace=True)

  clusters = _apply_classification(poprisk_raw_df, K=3)

  # from IPython import embed
  # embed()

  # Generate the CSV output with the classification
  # Work with the original dataframe
  poprisk_feat_df = poprisk_raw_df.copy(deep=False)

  # Draw it
  draw_title = "Population Risk Feature - Clustering by K-Means"
  output_png = output_folder + "/poprisk.png"

  frontiers = plot_clustering(clusters, draw_title, "Risk", output_png)

  # Now we generate the final classification. We take advantage from frontiers
  # calculated when plotting

  # Generate new classification
  risk_col = "Weighted Risk"
  poprisk_feat_df["Classification"] = [3 if risk >= frontiers[1] else \
                                       2 if risk >= frontiers[0] else \
                                       1 for risk in poprisk_feat_df[risk_col]]

  # We can drop density already
  remove = "cols"
  axis = 1 if "cols" in remove else 0
  poprisk_feat_df = poprisk_feat_df.drop(risk_col, axis)

  output_csv = output_folder + "/poprisk.csv"

  # 3. Write to features folder
  if os.path.isfile(output_csv):
    os.remove(output_csv)

  # Save to CSV file
  df_to_csv(output_csv, poprisk_feat_df)

  return poprisk_feat_df