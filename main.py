import numpy as np
import pandas as pd

import logging
from tqdm import tqdm
import datetime as dt
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

import torch as T
import torch.utils.data as tud
import pytorch_model as ptm

logging.basicConfig(level=logging.INFO)

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

  logging.info(f"{dt.datetime.now()} - Starting...")
  logging.info(f"{dt.datetime.now()} - Getting COVID-19 Cases Data...")
  confirmed, deaths, recovered = gen_covid19_feat(covid19_repo_url,
                                                  covid_raw_base_url,
                                                  output_raw=covid19_raw_folder,
                                                  output_csv=covid19_feat_folder)

  # Population Density
  logging.info(f"{dt.datetime.now()} - Getting Population Density Data...")
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
  logging.info(f"{dt.datetime.now()} - Getting Masks Usage Data...")
  masks_raw_file = "./data/raw/masks/masks.csv"
  masks_feat_folder = "./data/features/masks"
  masks_df = gen_masks_feat(masks_raw_file,
                            output_folder=masks_feat_folder)

  # Population Risk
  logging.info(f"{dt.datetime.now()} - Getting Population Risk Data...")
  poprisk_raw_file = "./data/raw/poprisk/poprisk.csv"
  poprisk_feat_folder = "./data/features/poprisk"
  poprisk_df = gen_poprisk_feat(poprisk_raw_file,
                                output_folder=poprisk_feat_folder)

  # Gov. Measures 1 - Lockdown
  logging.info(f"{dt.datetime.now()} - Getting Lockdown Data...")
  lockdown_raw_file = "./data/raw/govme/lockdown.csv"
  lockdown_feat_folder = "./data/features/govme"
  lockdown_df = gen_lockdown_feat(lockdown_raw_file,
                                  output_folder=lockdown_feat_folder)

  # Gov. Measures 2 - Borders Closed
  logging.info(f"{dt.datetime.now()} - Getting Borders Closing Data...")
  borcls_raw_file = "./data/raw/govme/borders.csv"
  borcls_feat_folder = "./data/features/govme"
  borders_df = gen_borders_feat(borcls_raw_file,
                                output_folder=borcls_feat_folder)

  # --------------------------------------------------------

  # 1. Prepare the data
  # Generate an unique dataset with features | output
  # f1, f2, f3, f4, f5, f6, output
  logging.info(f"{dt.datetime.now()} - Data merge...")
  pbar = tqdm(total=7)

  pbar.set_description("Expand cases into vector of samples...")
  data = expand_cases_to_vector(confirmed)
  pbar.update(1)

  # We feed 0 as default population density since it's the lowest and it's not
  # going to be used for training if it doesn't existc
  pbar.set_description("Fill with Popden feature...")
  fill_df_column_with_value(data, "Popden", popden_df, "Classification", 1)
  pbar.update(1)

  # 0 Default = No Masks
  pbar.set_description("Fill with Masks feature...")
  fill_df_column_with_value(data, "Masks", masks_df, "Mask", 0)
  pbar.update(1)

  # 1 Default = Lowest Risk
  pbar.set_description("Fill with Poprisk feature...")
  fill_df_column_with_value(data, "Poprisk", poprisk_df, "Classification", 1)
  pbar.update(1)

  # 0 Default = Doesn't apply
  pbar.set_description("Fill with Lockdown feature...")
  fill_df_column_date_based(data, "Lockdown", lockdown_df, 0)
  pbar.update(1)
  pbar.set_description("Fill with Borders feature...")
  fill_df_column_date_based(data, "Borders", borders_df, 0)
  pbar.update(1)

  # Make a copy to mantain all. In the one that continues, remove dates
  pbar.set_description("Removing dates after March 16th for training...")
  # after March 16th
  datacopy = data.copy(deep=True)
  #from IPython import embed
  #embed()
  mask = [dt.datetime.strptime(date, r"%m/%d/%y") <= dt.datetime.strptime("03/16/20", r"%m/%d/%y") for date in data["Date"].values.tolist()]
  data = data.loc[mask]
  pbar.update(1)
  pbar.close()

  # 2. Split in training data, test and prediction data
  device = 'cuda' if T.cuda.is_available() else 'cpu'

  logging.info(f"{dt.datetime.now()} - Data setup to feed the model...")
  pbar = tqdm(total=5)
  pbar.set_description("From Pandas.DataFrame to torch.Tensor...")
  x_cols = (np.arange(len(data.columns)) > 1)
  x_cols[-1] = False
  y_cols = np.arange(len(data.columns)) == (len(data.columns) - 1)

  # Data from Pandas.DataFrame to torch.Tensor
  x_train_tensor = T.tensor(data.loc[:, x_cols].values.astype(np.float32)).float().to(device)
  y_train_tensor = T.tensor(data.loc[:, y_cols].values.astype(np.float32)).float().to(device)
  #y_train_tensor = T.from_numpy(y_train).float().to(device)
  pbar.update(1)

  # From torch.Tensor to torch.Dataset
  pbar.set_description("From torch.Tensor to torch.Dataset...")
  dataset = tud.TensorDataset(x_train_tensor, y_train_tensor)
  pbar.update(1)

  # Split dataset into training and validate parts
  pbar.set_description("Split dataset into training / validate...")
  train_per = round(len(dataset) * 0.9)
  val_per = round(len(dataset) * 0.1)
  epochs = 80
  train_dataset, val_dataset = tud.dataset.random_split(dataset, 
                                                        [train_per, val_per])
  pbar.update(1)

  # From torch.Dataset to torch.DataLoader
  pbar.set_description("From torch.Dataset to torch.DataLoader...")
  batch_size = 12
  learning_rate=0.01

  train_loader = tud.DataLoader(dataset=train_dataset,
                                batch_size=batch_size)
  pbar.update(1)
  pbar.set_description("From torch.Dataset to torch.DataLoader...")

  val_loader = tud.DataLoader(dataset=val_dataset,
                              batch_size=batch_size)
  pbar.update(1)
  pbar.close()

  # 3. Run Training + Test loop. Plot results to compare.
  logging.info(f"{dt.datetime.now()} - Creating the model...")
  model = ptm.Net(input_size=6,
                  hidden_size=6).to(device)
  #print(model)

  #from IPython import embed
  #embed()

  # Train it
  logging.info(f"{dt.datetime.now()} - Training...")
  loss, val_loss = ptm.train(model,
                             train_loader=train_loader,
                             eval_loader=val_loader,
                             device=device,
                             lr=learning_rate,
                             batch=batch_size,
                             epochs=epochs)

  # Test it
  val_loss_test = ptm.test(model, device, data_loader=val_loader)

  # Save model
  T.save(model.state_dict(), f"./model_result_val_loss_{val_loss[-1]}")

  logging.info(f"{dt.datetime.now()} - Latest Loss: {loss[-1]}")
  logging.info(f"{dt.datetime.now()} - Latest Val Loss: {val_loss[-1]}")
  logging.info(f"{dt.datetime.now()} - Test: {val_loss_test}")


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