import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt

import get_population_density as popden

# Column order imposed by the data source of the COVID-19
covid_columns = ["Province/State", 
                 "Country/Region", 
                 "Last Update", 
                 "Confirmed", 
                 "Deaths", 
                 "Recovered"]

def draw_data_in_bar(values, labels, title):
  """Draws as a bar char the specified values and tags it with the 
  specified labels and title
  """

  x = np.arange(1, len(values)+1)

  # Draw it
  plt.bar(x, values)
  # Put the labels
  plt.xticks(x, labels, rotation='vertical')
  # Put the title
  plt.title(title)
  # Show it
  plt.show()

def draw_aggr(df, grp_col, data_col):
  """Draws the specified agreggated data column from a pandas.datataframe
  grouped by the specified aggregation column   
  """

  grouped_data = df.groupby(by=grp_col).sum()

  values = grouped_data[data_col].to_numpy(dtype=int)
  labels = grouped_data.index.to_numpy(dtype=str)

  title = data_col + " Cases"

  draw_data_in_bar(values, labels, title)

if __name__ == "__main__":

  # 1. Get population density values by executing our script
  # pbar.set_description(desc="Removing unnecesary fields...", refresh=True)
  location_popden = popden.latest_locations_with_popden()
  # pbar.update(n=1)

  # Load data from file
  df = pd.read_csv(r'.\data\01-22-2020.csv', header='infer')

  # Replace NaN values for 0
  for j in range(3,6):
    df[covid_columns[j]].fillna(0, inplace=True)

  # print(df)
  from IPython import embed
  embed()

  # Draw our data series
  draw_aggr(df, covid_columns[1], covid_columns[3])