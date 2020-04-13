"""This script aims to prepare everything needed to create the model, train
and predict our use case.
"""

import torch as T
import torch.utils.data as tud

import numpy as np
import pandas as pd

from collections import OrderedDict

import logging
from tqdm import tqdm

# -----------------------------------------------------------
def define_model(input_size=6, hidden_size=6, model="X/2"):
  """Constructs our model based on several parameters.

  Should refactor.
  """

  net = None
  model_layers = OrderedDict()

  # Latest layer to name the output layer afterwards
  latest = 1

  # Output layer input
  out_input = 0

  # First hidden layer always equal
  curr_input = input_size
  curr_hidden = hidden_size
  model_layers['fc1'] = T.nn.Linear(curr_input, curr_hidden)
  model_layers['relu1'] = T.nn.LeakyReLU()
  latest += 1
  out_input = curr_hidden

  # The rest is dependent on the model selected
  curr_input = curr_hidden
  curr_hidden = int(curr_hidden / 2) if "X/2" == model else \
                int(curr_hidden*2 + 1) if "X*2+1" == model else \
                int(curr_hidden*2 + 1) if "X*2+1 | X" == model else 0

  if curr_hidden > 0:
    model_layers['fc2'] = T.nn.Linear(curr_input, curr_hidden)
    model_layers['relu2'] = (T.nn.LeakyReLU())
    latest += 1
    out_input = curr_hidden

  curr_input = curr_hidden
  curr_hidden = int(hidden_size) if "X*2+1 | X" == model else 0
  if curr_hidden > 0:
    model_layers['fc3'] = T.nn.Linear(curr_input, curr_hidden)
    model_layers['relu3'] = T.nn.LeakyReLU()
    latest += 1
    out_input = curr_hidden

  # Output layer
  model_layers['fc' + str(latest)] = T.nn.Linear(out_input, 1)
    
  # Construct the sequential mode and return it
  net = T.nn.Sequential(model_layers)

  return net

# -----------------------------------------------------------
def loss_func(loss):
  """Returns the desired loss function
  """
  func = None
  if ("MAE" in loss) or ("mean absolute error" in loss):
    # We are calculating the mean for each step in the batch/mini_batch/epoch
    func = T.nn.L1Loss(reduction='mean')
  elif ("MSE" in loss) or ("mean squared error" in loss):
    # We are calculating the mean for each step in the batch/mini_batch/epoch
    func = T.nn.MSELoss(reduction='mean')

  return func

# -----------------------------------------------------------
def optimizer(opt, model, lr=0.01):
  """Applies the desired optimizer to the given model with the
  given learning rate
  """
  func = None
  if ("SGD" in opt):
    func = T.optim.SGD

  return func(model.parameters(), lr=lr)

# -----------------------------------------------------------
def make_train_step(model, loss_fn, optimizer):
  """Generates a function that performs a step in the train loop
  """
  def train_step(x, y):

    # Sets model to TRAIN mode
    model.train()

    # Makes predictions
    yhat = model(x)

    # Computes loss
    loss = loss_fn(yhat, y)

    # Computes gradients
    loss.backward()

    # Updates parameters and zeroes gradients
    optimizer.step()
    optimizer.zero_grad()

    # Returns the loss
    return loss.item()
  
  return train_step

# -----------------------------------------------------------
def train(model, train_loader, eval_loader, device,
          lr=0.01, batch=8, epochs=600):
  """Trains using as much as n_epochs epochs.
  """

  # 1. Create the optimizer and loss function
  lf_instance = loss_func("MSE")
  opt_instance = optimizer("SGD", model, lr=lr)

  # 2. Creates the training function
  train_step = make_train_step(model, lf_instance, opt_instance)
  losses = []
  val_losses = []

  # 3. Training itself
  pbar = tqdm(total=(epochs*2))
  for epoch in range(epochs):

    pbar.set_description("Training epoch: %s" % epoch)
    for x_batch, y_batch in train_loader:

      # Move to correct device, maybe not needed
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)
      
      loss = train_step(x_batch, y_batch)
      losses.append(loss)

    pbar.update(1)
    pbar.set_description("Evaluating epoch: %s" % epoch)

    # Evaluation side. No gradients required
    with T.no_grad():

      for x_val, y_val in eval_loader:

        x_val = x_val.to(device)
        y_val = y_val.to(device)

        model.eval()

        yhat = model(x_val)
        val_loss = lf_instance(y_val, yhat)
        val_losses.append(val_loss.item())

    pbar.update(1)

  pbar.close()

  return losses, val_losses

# -----------------------------------------------------------
def test(model, device, data_loader=None):
  """Tests and evaluates the model to get the desired metrics.
  """

  if data_loader is None:
    raise Exception("Doesn't exist Data loader for test!")

  # 1. Create the loss function
  lf_instance = loss_func("MAE")
  val_losses = []

  # Evaluation
  with T.no_grad():

    model.eval()

    for x_val, y_val in data_loader:

      x_val = x_val.to(device)
      y_val = y_val.to(device)
      
      yhat = model(x_val)
      val_loss = lf_instance(y_val, yhat)
      val_losses.append(val_loss.item())

  return (sum(val_losses) / len(val_losses))

# -----------------------------------------------------------
def predict(model, device, data_loader):
  """Takes the provided input and model, and returns predictions
  as a python array.
  """
  
  results = []

  with T.no_grad():

    model.eval()

    for x_val, y_val in data_loader:

      x_val = x_val.to(device)
      yhat = model(x_val)

      # logging.info(f"From {x_val.squeeze().tolist()} to: {yhat.squeeze().tolist()}")

      #from IPython import embed
      #embed()

      out = x_val.squeeze().tolist() + [y_val.squeeze().tolist(),
                                        yhat.squeeze().tolist()]
      results.append(out)

  return results

# -----------------------------------------------------------
def datagen(dataset, batch_size, shuffle=True):
  """Takes the provided input and model, and returns a prediction
  """
  
  loader = T.utils.data.DataLoader(dataset=dataset, 
                                   batch_size=batch_size,
                                   shuffle=shuffle)

  return loader

# -----------------------------------------------------------
def df_2_dataset(df, x_cols, y_cols, device):
  """Takes the provided dataframe as input and returns datasets ready
  to be used as input for a dataloader later on.

  Arguments
  ---
  df: the dataframe to be used
  cols_x: a mask (bool) array with the columns to be used from the df
  cols_y: a mask (bool) array with the columns to be used as features
  device: device where the data should be
  """
  
  #from IPython import embed
  #embed()

  # Pass specified cols to tensors
  x_tensor = T.tensor(df.loc[:, x_cols].values.astype(np.float32)).float().to(device)

  if y_cols is not None:
    y_tensor = T.tensor(df.loc[:, y_cols].values.astype(np.float32)).float().to(device)
    dataset = tud.TensorDataset(x_tensor, y_tensor)
  else:
    dataset = tud.TensorDataset(x_tensor)

  return dataset

# -----------------------------------------------------------
def split_dataset(dataset, train_per, val_per):
  """Takes the provided dataset as input, and returns it splitted in
  two taking into account the percentages for training and validation.

  Percentages should be expressed in the 0 to 1 range, e.g. 0.9 for 90%
  """
  
  train_per = round(len(dataset) * train_per)
  val_per = round(len(dataset) * val_per)

  train_dataset, val_dataset = tud.dataset.random_split(dataset, 
                                                        [train_per, val_per])

  return train_dataset, val_dataset