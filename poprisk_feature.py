"""This script takes the data about Population Risk Weighted and classifies it
in low (1), medium (2) or high (3) risk by applying a K-Means algorithm.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from classes.csvhandler import df_to_csv, read_csv_as_df

def _sort_cluster_arrays(arrays):
  """Calculates for the given array which is their natural order:
  minimum values < ... < maxium values

  Returns the given array sorted like this
  """
  output = arrays

  for i in range(0, len(output)):
    for j in range(i+1, len(output)):
      if min(output[j]) < min(output[i]):
        output[i], output[j] = output[j], output[i]

  return output

def _calculate_frontiers(clusters):
  """Calculates the frontiers between the clusters

  Returns two float values specifying these frontiers
  """
  frontier_1 = (min(clusters[1]) + max(clusters[0])) / 2
  frontier_2 = (min(clusters[2]) + max(clusters[1])) / 2

  return frontier_1, frontier_2

def generate_poprisk_feature(poprisk_weighted_file,
                             output_png="./poprisk.png",
                             output_csv="./poprisk.csv"):

  # Read weighted risk
  pop_risk_weighted_df = read_csv_as_df(poprisk_weighted_file)

  # Convert the dataframe to an array
  pop_risk_weighted_df["Weighted Risk"].replace(0, 0.5, inplace=True)
  prw_array = pop_risk_weighted_df["Weighted Risk"].to_numpy(dtype=float)

  data_2_feed = np.array(
    [[pop_risk_weighted_df["Weighted Risk"][i], 0] for i in range(0, len(prw_array))]
  )

  #  Apply K-Means with K=3
  K = 3
  kmeans = KMeans(n_clusters=K, random_state=0).fit(data_2_feed)

  # Plot the results

  # from IPython import embed
  # embed()

  # Draw it
  fig = plt.figure()

  # Separate in the three clusters
  cluster_1 = [data_2_feed[i][0] for i in range(0, len(data_2_feed)) if kmeans.labels_[i] == 0]
  cluster_2 = [data_2_feed[i][0] for i in range(0, len(data_2_feed)) if kmeans.labels_[i] == 1]
  cluster_3 = [data_2_feed[i][0] for i in range(0, len(data_2_feed)) if kmeans.labels_[i] == 2]

  # from IPython import embed
  # embed()

  # Know which one are the low and which one are the high risk
  cluster_1, cluster_2, cluster_3 = _sort_cluster_arrays([cluster_1, cluster_2, cluster_3])

  # Continue plotting
  y_1 = np.zeros(len(cluster_1))
  y_2 = np.zeros(len(cluster_2))
  y_3 = np.zeros(len(cluster_3))

  plt_clu_1 = plt.scatter(cluster_1, y_1, c='b', marker='x')
  plt_clu_2 = plt.scatter(cluster_2, y_2, c='g', marker='x')
  plt_clu_3 = plt.scatter(cluster_3, y_3, c='r', marker='x')

  # Put the boundaries of each cluster
  frontier_1, frontier_2 = _calculate_frontiers([cluster_1, cluster_2, cluster_3])
  plt_frontier_1 = plt.axvline(x=frontier_1, c='r', ls='-.')
  plt_frontier_2 = plt.axvline(x=frontier_2, c='c', ls='--')

  # plt.xticks(x, labels, rotation='vertical')
  # Put the legend
  plt.legend((plt_clu_1, 
              plt_clu_2, 
              plt_clu_3, 
              plt_frontier_1, 
              plt_frontier_2),
             ("Low Risk", 
              "Medium Risk", 
              "High Risk", 
              f"Frontier 1 - {frontier_1}", 
              f"Frontier 2 - {frontier_2}"))

  # Put the title
  plt.title("Population Risk Feature - Clustering by K-Means")

  # Apply grid and x wide
  plt.xlim(0, 1)
  plt.grid()
  # Show it
  # plt.show()

  fig.savefig(output_png)