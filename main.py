import numpy as np
import pandas as pd

import logging
from tqdm import tqdm

from covid19_feature import gen_covid19_feat
from poprisk_feature import gen_poprisk_feat
from masks_feature import gen_masks_feat
from popden_feature import gen_popden_feat
from lockdown_feature import gen_lockdown_feat
from bordersclosed_feature import gen_borders_feat
from data import (expand_cases_to_vector,
                  fill_df_column_with_value,
                  fill_df_column_date_based,
                  split_by_date,
                  clean_invalid_data,
                  normalize_data)

import classes.plt_handler as plt

import torch as T
import torch.utils.data as tud
from torchsummary import summary
import pytorch_model as ptm

import os
import argparse

log_format="%(asctime)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

def arg_val(default, override, kind):
  """Return the argument value to use, making sure it is a list
  """
  if override is None:
    return default
  else:
    return [kind(override)]

def pickle_available(pickle_path):
  """Checks if data used is pickled already

  To-Do: update to check on updates by date from raw data
  """

  if os.path.exists(pickle_path):
    return True


def save_pickle_data(pickle_file, data):
  """Saves the provided data in a pickle file. It assumes pandas dataframe
  as data.
  """

  try:
    data.to_pickle(pickle_file)

  except Exception as e:
    logging.error(f"Wasn't able to save pickled data to {pickle_file}")
    logging.info(e)

def load_pickle_data(pickle_file):
  """Loads and return pickled data. It assumes pandas dataframe
  as data.
  """

  data = None
  try:
    data = pd.read_pickle(pickle_file)
  except Exception as e:
    logging.error(f"Wasn't able to load pickled data from {pickle_file}")
    logging.info(e)
  
  return data

if __name__ == "__main__":

  # Arguments
  desc = [
    "Executes a NN model to solve the COVID-19 confirmed cases. ",
    "Can both train and infer the model.\n",
    "If any parameter is specified when executing, it will override",
    "the default training loop and become a single value execution."
  ]
  descstr = "".join(desc)
  parser = argparse.ArgumentParser(description=descstr)
  parser.add_argument("--exec", help="Execute only train/infer")
  parser.add_argument("--arch", help="Model architecture to use")
  parser.add_argument("--model", help="Model name for inference")
  parser.add_argument("--epochs", help="Number of epochs")
  parser.add_argument("--batch", help="Batch size")
  parser.add_argument("--lr", help="Static learning rate")
  parser.add_argument("--hidden", help="Size of the first hidden layer")
  args = parser.parse_args()

  epoch_array = [50, 100, 150, 200]
  epoch_array = arg_val(epoch_array, args.epochs, int)

  batch_array = [ 8, 12, 48]
  batch_array = arg_val(batch_array, args.batch, int)

  lr_array = [0.0001, 0.00001, 0.000001]
  lr_array = arg_val(lr_array, args.lr, float)

  hidden_array = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
  hidden_array = arg_val(hidden_array, args.hidden, int)

  train = True if args.exec is None else True if "train" in args.exec else False
  infer = True if args.exec is None else True if "infer" in args.exec else False

  if infer:
    if args.model is None:
      logging.error("No model has been defined to infer")
      exit()
    else:
      if not os.path.isfile("./models/" + args.model):
        logging.error("The model file can't be found")
        exit()

    if args.hidden is None:
      logging.error("No model size has been defined")
      exit()

  # Generate output folders if they doesn't exist
  if not os.path.exists("./models"):
    os.mkdir("./models")

  if not os.path.exists("./graphics"):
    os.mkdir("./graphics")

  if not os.path.exists("./pickle_cache"):
    os.mkdir("./pickle_cache")

  # 0. Generate features
  pickle_path_bfr = "./pickle_cache/dataset_bfr.pickle"
  pickle_path_aft = "./pickle_cache/dataset_aft.pickle"
  if not pickle_available(pickle_path_bfr) or \
     not pickle_available(pickle_path_aft):

    # COVID-19
    covid19_repo_url = r"https://github.com/CSSEGISandData/COVID-19/tree/master/" + \
                      r"csse_covid_19_data/csse_covid_19_time_series"
    covid_raw_base_url = r"https://raw.githubusercontent.com/CSSEGISandData/" + \
                        r"COVID-19/master/csse_covid_19_data/csse_covid_19_time_series"
    covid19_raw_folder = "./data/raw/covid/"
    covid19_feat_folder = "./data/features/covid/"

    logging.info(f"Starting...")
    logging.info(f"Getting COVID-19 Cases Data...")
    confirmed, deaths, recovered = gen_covid19_feat(covid19_repo_url,
                                                    covid_raw_base_url,
                                                    input_raw=covid19_raw_folder,
                                                    output_folder=covid19_feat_folder,
                                                    avoid_dwld=True)

    # Population Density
    logging.info(f"Getting Population Density Data...")
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
    logging.info(f"Getting Masks Usage Data...")
    masks_raw_file = "./data/raw/masks/masks.csv"
    masks_feat_folder = "./data/features/masks"
    masks_df = gen_masks_feat(masks_raw_file,
                              output_folder=masks_feat_folder)

    # Population Risk
    logging.info(f"Getting Population Risk Data...")
    poprisk_raw_file = "./data/raw/poprisk/poprisk.csv"
    poprisk_feat_folder = "./data/features/poprisk"
    poprisk_df = gen_poprisk_feat(poprisk_raw_file,
                                  output_folder=poprisk_feat_folder)

    # Gov. Measures 1 - Lockdown
    logging.info(f"Getting Lockdown Data...")
    lockdown_raw_file = "./data/raw/govme/lockdown.csv"
    lockdown_feat_folder = "./data/features/govme"
    lockdown_df = gen_lockdown_feat(lockdown_raw_file,
                                    output_folder=lockdown_feat_folder)

    # Gov. Measures 2 - Borders Closed
    logging.info(f"Getting Borders Closing Data...")
    borcls_raw_file = "./data/raw/govme/borders.csv"
    borcls_feat_folder = "./data/features/govme"
    borders_df = gen_borders_feat(borcls_raw_file,
                                  output_folder=borcls_feat_folder)

    # --------------------------------------------------------

    # 1. Prepare the data
    # Generate an unique dataset with features | output
    # f1, f2, f3, f4, f5, f6, output
    logging.info(f"Data merge...")
    pbar = tqdm(total=10)

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

    # Split by date
    pbar.set_description("Split data up to March, 16th...")
    limit_date = "03/31/20"
    bfr, aft = split_by_date(data, limit_date)
    pbar.update(1)

    # Clean invalid data
    pbar.set_description("Removing samples with 0 next day cases...")
    bfr = clean_invalid_data(bfr)
    aft = clean_invalid_data(aft)
    pbar.update(1)

    # Normalize
    pbar.set_description("Normalize data before training...")
    bfr = normalize_data(bfr, norm_cases=False)
    aft = normalize_data(aft, norm_cases=False)
    pbar.update(1)

    # Save data for next runs
    pbar.set_description("Saving pickled data...")
    save_pickle_data(pickle_path_bfr, bfr)
    save_pickle_data(pickle_path_aft, aft)
    pbar.update(1)
    pbar.close()

  else:

    logging.info(f"Loading pickled data...")
    bfr = load_pickle_data(pickle_path_bfr)
    aft = load_pickle_data(pickle_path_aft)

  # TEMPORAL FIX
  # bfr.loc[:, "NextDay"] = bfr.loc[:, "NextDay"] * 100
  # aft.loc[:, "NextDay"] = aft.loc[:, "NextDay"] * 100
  bfr.loc[:, "Cases"] = np.log(bfr.loc[:, "Cases"])
  bfr.loc[:, "NextDay"] = np.log(bfr.loc[:, "NextDay"])

  aft.loc[:, "Cases"] = np.log(aft.loc[:, "Cases"])
  aft.loc[:, "NextDay"] = np.log(aft.loc[:, "NextDay"])

  #print(bfr.head(5))
  #print(aft.head(5))
  #exit()

  # 2. Split in training data, test and prediction data
  device = 'cuda' if T.cuda.is_available() else 'cpu'

  logging.info(f"Data setup to feed the model...")
  
  x_cols = (np.arange(len(bfr.columns)) > 1)
  x_cols[-1] = False
  y_cols = np.arange(len(bfr.columns)) == (len(bfr.columns) - 1)

  dataset = ptm.df_2_dataset(bfr, x_cols, y_cols, device)

  # Split dataset into training and validate parts
  logging.info("Split dataset into training / validate...")
  train_dataset, val_dataset = ptm.split_dataset(dataset, 0.9, 0.1)

  # 3. Run Training + Test loop. Plot results to compare.
  results = pd.DataFrame(
    columns = [
      "Epochs",
      "Batch",
      "Lr",
      "Hidden",
      "Iter",
      "ValLoss"
    ]
  )

  # This will be overrided later, it's just to check how it is generated
  model = ptm.define_model(input_size=6,
                          hidden_size=hidden_array[0],
                          model=args.arch).to(device)
				                  #model="X*2+1 | X").to(device)

  summary(model, input_size=(0, 1, 6))

  if train:

    logging.info(f"Training")
    for epochs in epoch_array:

      for batch in batch_array:

        # From torch.Dataset to torch.DataLoader
        train_loader = ptm.datagen(dataset=train_dataset,
                                   batch_size=batch)

        val_loader = ptm.datagen(dataset=val_dataset,
                                 batch_size=batch)

        for lr in lr_array:

          for hidden in hidden_array:

            logging.info(f"\tEpochs {epochs} - Batch {batch} - Lr {lr} - hidden {hidden}")

            for i in range(0, 3):

              logging.info(f"\tIteration {i+1}")
              T.manual_seed(i*i)

              model = ptm.define_model(input_size=6,
                                       hidden_size=hidden,
                                       model=args.arch).to(device)
				                               #model="X*2+1 | X").to(device)

              # Train it
              loss, val_loss = ptm.train(model,
                                        train_loader=train_loader,
                                        eval_loader=val_loader,
                                        device=device,
                                        lr=lr,
                                        batch=batch,
                                        epochs=epochs)

              # Plot
              filename = f"./graphics/model_epochs_{epochs}_batch_{batch}_" + \
                        f"lr_{lr}_hidden_{hidden}_i_{i}_losscomparison.png"
              plt.draw_loss(loss, val_loss,
                            title="Model training Loss Comparison",
                            output_png=filename,
                            plot=False,
                            figsize=(21.9, 17.8))

              # Test it
              val_loss_test = ptm.test(model, device, data_loader=val_loader)

              # Save model
              #logging.info(f"Loss: {loss}\n")
              #logging.info(f"Val Loss: {val_loss}\n")
              logging.info(f"Test: {val_loss_test}\n")

              filename = f"./models/model_epochs_{epochs}_batch_{batch}_" + \
                        f"lr_{lr}_hidden_{hidden}_i_{i}_valloss_{val_loss_test}"
              T.save(model.state_dict(),filename)

              # Save to variables to plot
              results.loc[len(results)] = [epochs,
                                          batch,
                                          lr,
                                          hidden,
                                          i,
                                          val_loss_test]

            # exit()

    # Save data to plot to file
    training_df = pd.DataFrame(results)
    training_df.to_csv("./models/index.csv", header=True)

  if infer:

    logging.info(f"Inference")

    infer_dataset = ptm.df_2_dataset(bfr, x_cols, y_cols, device)

    infer_loader = ptm.datagen(dataset=infer_dataset,
                               batch_size=1)

    model = ptm.define_model(input_size=6,
                             hidden_size=int(args.hidden),
                             model="X*2+1 | X").to(device)
    model.load_state_dict(T.load("./models/" + args.model))

    # Infer
    # Results are a list of arrays
    # Each array: [features, y, yhat]
    results = ptm.predict(model, device, data_loader=infer_loader)

    # Plot
    data_2_plot = pd.DataFrame(results, columns=[
      "Cases",
      "Popden",
      "Masks",
      "Poprisk",
      "Lockdown",
      "Borders",
      "NextDay",
      "yhat"
    ])

    plt.draw_comparison(data_2_plot, "NextDay", "yhat",
                        title="Prediction and deviation",
                        plot=True,
                        output_png=f"./results/{args.model}_test.png",
                        figsize=(21.9, 17.8))