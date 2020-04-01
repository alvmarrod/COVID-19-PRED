import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt

from covid19_feature import gen_covid19_feat
from poprisk_feature import gen_poprisk_feat
from masks_feature import gen_masks_feat
from popden_feature import gen_popden_feat
from lockdown_feature import gen_lockdown_feat
from bordersclosed_feature import gen_borders_feat
from data import (expand_cases_to_vector,
                  fill_df_column_with_value,
                  fill_df_column_date_based)

import pytorch_model as ptm

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

  # 0. Generate features

  # COVID-19
  covid19_repo_url = r"https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series"
  covid_raw_base_url = r"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series"
  covid19_raw_folder = "./data/raw/covid/"
  covid19_feat_folder = "./data/features/covid/"
  confirmed, deaths, recovered = gen_covid19_feat(covid19_repo_url,
                                                  covid_raw_base_url,
                                                  output_raw=covid19_raw_folder,
                                                  output_csv=covid19_feat_folder)

  # Population Density
  popden_raw_file = "./data/raw/popden/popden.csv"
  popden_feat_folder = "./data/features/popden"
  handicaps = {
    "Australia": 3
  }
  popden_df = gen_popden_feat(popden_raw_file,
                              output_folder=popden_feat_folder,
                              handicaps=handicaps,
                              remove_over=True)

  # Masks Usage
  masks_raw_file = "./data/raw/masks/masks.csv"
  masks_feat_folder = "./data/features/masks"
  masks_df = gen_masks_feat(masks_raw_file,
                            output_folder=masks_feat_folder)

  # Population Risk
  poprisk_raw_file = "./data/raw/poprisk/poprisk.csv"
  poprisk_feat_folder = "./data/features/poprisk"
  poprisk_df = gen_poprisk_feat(poprisk_raw_file,
                                output_folder=poprisk_feat_folder)

  # Gov. Measures 1 - Lockdown
  lockdown_raw_file = "./data/raw/govme/lockdown.csv"
  lockdown_feat_folder = "./data/features/govme"
  lockdown_df = gen_lockdown_feat(lockdown_raw_file,
                                  output_folder=lockdown_feat_folder)

  # Gov. Measures 2 - Borders Closed
  borcls_raw_file = "./data/raw/govme/borders.csv"
  borcls_feat_folder = "./data/features/govme"
  borders_df = gen_borders_feat(borcls_raw_file,
                                output_folder=borcls_feat_folder)

  # --------------------------------------------------------

  # 1. Prepare the data
  # Generate an unique dataset with features | output
  # f1, f2, f3, f4, f5, f6, output
  data = expand_cases_to_vector(confirmed)

  # We feed 0 as default population density since it's the lowest and it's not
  # going to be used for training if it doesn't existc
  fill_df_column_with_value(data, "Popden", popden_df, "Classification", 1)

  # 0 Default = No Masks
  fill_df_column_with_value(data, "Masks", masks_df, "Mask", 0)

  # 1 Default = Lowest Risk
  fill_df_column_with_value(data, "Poprisk", poprisk_df, "Classification", 1)

  # 0 Default = Doesn't apply
  fill_df_column_date_based(data, "Lockdown", lockdown_df, 0)
  fill_df_column_date_based(data, "Borders", borders_df, 0)

  #data = pd.merge(data, borders_df, on="Country", how="inner")

  # print(data)

  #from IPython import embed
  #embed()

  # 2. Split in training data, test and prediction data
  np.random.seed(42)
  idx = np.arange(len(data))

  train_idx = idx[:(0.8*len(data))]
  val_idx = idx[(0.8*len(data)):]

  train_x, train_y = data[train_idx][:-1], data[train_idx][-1]
  val_x, val_y = data[val_idx][:-1], data[val_idx][-1]

  #T.manual_seed(1)
  #np.random.seed(1)
  # 1. load data
  #print("Loading Iris data into memory \n")
  #train_file = ".\\Data\\iris_train.txt"
  #test_file = ".\\Data\\iris_test.txt"
  #train_x = np.loadtxt(train_file, usecols=range(0,4),
  #  delimiter=",",  skiprows=0, dtype=np.float32)
  #train_y = np.loadtxt(train_file, usecols=[4],
  #  delimiter=",", skiprows=0, dtype=np.float32)
  #test_x = np.loadtxt(test_file, usecols=range(0,4),
  #  delimiter=",",  skiprows=0, dtype=np.float32)
  #test_y = np.loadtxt(test_file, usecols=[4],
  #  delimiter=",", skiprows=0, dtype=np.float32)
  train_data = None

  # 3. Run Training + Test loop. Plot results to compare.

  batch_size = 12
  
  # Creat the model
  model = ptm.Net(input_size=6,
                  hidden_size=6)
  print(model)

  # Train it
  ptm.train(model,
            dataset=train_data,
            lr=learning_rate,
            batch=batch_size,
            epochs=max_epochs)

  # 4. Plot graphics for best model.
  # Grahpic 1 - Raw data plot (cases reported by governments)
  # Graphic 2 - Predicted data from the model, taking as input the official
  # data reported the each previous day
  # Graphic 3 - Predicted data from the model, taking as input its own output
  # from the previous day except for the very first day, that takes as input
  # the official reports from the previous day



  # 1. Get population density values by executing our script
  # pbar.set_description(desc="Removing unnecesary fields...", refresh=True)
  ###### location_popden = latest_locations_with_popden()
  # pbar.update(n=1)

  # Load data from file
  # df = pd.read_csv(r'.\data\01-22-2020.csv', header='infer')

  # Replace NaN values for 0
  # for j in range(3,6):
  #   df[covid_columns[j]].fillna(0, inplace=True)

  # print(df)
  # from IPython import embed
  # embed()

  # Draw our data series
  # draw_aggr(df, covid_columns[1], covid_columns[3])