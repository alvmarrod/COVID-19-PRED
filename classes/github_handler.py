
"""This handler defines functions to allow getting the data needed for the model
from github repos/files
"""

import requests
import pandas as pd
from io import StringIO
from .csv_handler import df_to_csv
from .parsers.github_parser import github_parser

def get_files_from_github_folder(url):
  """Parses the github folder passed by URL and all the filenames from the folder

  Returns
  -------
  An array with the filenames
  """

  r = requests.get(url, auth=('', ''))
  html = r.text

  # Parse it and check the latest file by it's name
  parser = github_parser()
  parser.feed(html)

  return parser.files_dict

def get_latest_file_github_folder(url):
  """Parses the github folder passed by URL and returns the latest file name,
  using the file names as date indicators.
  """

  r = requests.get(url, auth=('', ''))
  html = r.text

  # Parse it and check the latest file by it's name
  parser = github_parser()
  parser.feed(html)

  return parser.latest_file


def download_as_csv(url, target_file=""):  
  """Download the file from the specified URL and stores it in the
  target file name and path
  """

  if target_file != "":

    # Get the file from the URL
    r = requests.get(url, auth=('', ''))
    html = r.text

    # Read it to a pd.dataframe
    df = pd.read_csv(StringIO(html), header='infer')

    # Write to file
    df_to_csv(target_file, df)
  
  else:
    raise Exception("No target file provided")
