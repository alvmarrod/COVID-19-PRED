"""This script aims to prepare everything needed to create the model, train
and predict our use case.
"""

import torch as T
# import torch.nn.functional as F

import numpy as np
import pandas as pd

# -----------------------------------------------------------
class Net(T.nn.Module):
  def __init__(self, input_size=6, hidden_size=6):
    """Constructs our model.

    Input size: 6 features
    """
    super(Net, self).__init__()

    # 1. Fully connected layer
    self.fc1 = T.nn.Linear(input_size, hidden_size)
    self.relu1 = T.nn.ReLU()
    
    # 2. Fully connect layer
    hl2_size = 3
    self.fc2 = T.nn.Linear(hidden_size, hl2_size)
    self.relu2 = T.nn.ReLU()

    # 3. Output layer
    self.fc3 = T.nn.Linear(hl2_size, 1)

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu1(out)
    out = self.fc2(out)
    out = self.relu2(out)
    out = self.fc3(out)
    return out

# -----------------------------------------------------------
def loss_func(loss):
  """Returns the desired loss function
  """
  func = None
  if ("MAE" in loss) or ("mean absolute error" in loss):
    # We are calculating the mean for each step in the batch/mini_batch/epoch
    func = T.nn.L1Loss(reduction='mean')

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
def calculate_mae(model, data_x, data_y):
  """Calculates the mean absolute-loss of the provided model based on 
  the provided test data
  """

  # Code is for accuracy. To do.

  X = T.Tensor(data_x)
  Y = T.LongTensor(data_y)

  oupt = model(X)

  (_, arg_maxs) = T.max(oupt.data, dim=1)
  num_correct = T.sum(Y==arg_maxs)

  acc = (num_correct * 100.0 / len(data_y))
  return acc.item()

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
    loss = loss_fn(y, yhat)

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
          lr=0.01, batch=8, n_epochs=600):
  """Trains using as much as n_epochs epochs.
  """

  # 1. Create the optimizer and loss function
  lf_instance = loss_func("MAE")
  opt_instance = optimizer("SGD", model, lr=lr)

  # 2. Creates the training function
  train_step = make_train_step(model, lf_instance, opt_instance)
  losses = []
  val_losses = []

  # 3. Training itself
  for epoch in range(n_epochs):

    for x_batch, y_batch in train_loader:

      # Move to correct device, maybe not needed
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)

      loss = train_step(x_batch, y_batch)
      losses.append(loss)

    # Evaluation side. No gradients required
    with T.no_grad():

      for x_val, y_val in eval_loader:

        x_val = x_val.to(device)
        y_val = y_val.to(device)

        model.eval()

        yhat = model(x_val)
        val_loss = lf_instance(y_val, yhat)
        val_losses.append(val_loss.item())


# -----------------------------------------------------------
def test(net, test_x, test_y):
  """Tests and evaluates the model to get the desired metrics.
  """
  
  # set eval mode
  net = net.eval()  

  mae = calculate_mae(net, test_x, test_y)
  print("Mean Absolute Loss on test data = %0.2f%%" % mae)

# -----------------------------------------------------------
def predict(net, input):
  """Takes the provided input and model, and returns a prediction
  """
  
  unk = np.array([[6.1, 3.1, 5.1, 1.1]], dtype=np.float32)
  unk = T.tensor(unk)  # to Tensor
  logits = net(unk)  # values do not sum to 1.0
  probs_t = T.softmax(logits, dim=1)  # as Tensor
  probs = probs_t.detach().numpy()    # to numpy array
  
  print("\nSetting inputs to:")
  for x in unk[0]:
    print("%0.1f " % x, end="")

  print("\nPredicted: (setosa, versicolor, virginica)")

  for p in probs[0]:
    print("%0.4f " % p, end="")

  print("\n\nEnd Iris demo")

# -----------------------------------------------------------
def datagen(dataset, batch, shuffle=True):
  """Takes the provided input and model, and returns a prediction
  """
  
  loader = T.utils.data.DataLoader(dataset=dataset, 
                                   batch_size=batch,
                                   shuffle=shuffle)

  return loader


# -----------------------------------------------------------
class CustomDataset(T.utils.Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

