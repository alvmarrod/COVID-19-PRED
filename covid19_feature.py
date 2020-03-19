"""This script aims to be able to generate the data of the COVID-19 cases and
get it ready to be used as a feature for the models.
"""

import os.path
import numpy as np
import pandas as pd

from io import StringIO

from classes.github_handler import (get_files_from_github_folder,
                                    download_as_csv)

date_format = r'%m-%d-%Y'

def gen_covid19_feat(covid19_dr_url,
                     covid19_raw_url,
                     output_raw="./data/raw/covid/",
                     output_csv="./data/features/covid/covid.csv"):
  """This function receives the URL from the repository where CSV with the
  daily reports of the COVID-19 are stored. Then it downloads them all into the
  specified as RAW folder, and creates an unique CSV as feature output.
  """

  # 1. Get the files and save them to Raw data folder

  # Get file names in the repo
  covid_files_list = get_files_from_github_folder(covid19_dr_url)

  # Prepare URLs to download
  covid_url_list = [[output_raw + "/" + file + ".csv", 
                    covid19_raw_url + "/" + file + ".csv"] 
                    for file in covid_files_list]

  # Download
  for file in covid_url_list:
    if not os.path.isfile(file[0]):
      download_as_csv(file[1], file[0])

  # 2. Create an unique CSV with an unique dataframe
