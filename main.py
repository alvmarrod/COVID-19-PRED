import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt

from covid19_feature import gen_covid19_feat
from poprisk_feature import gen_poprisk_feat

# from popden_feature import get_locations_df, latest_locations_with_popden

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
  popden_feat_csv = "./data/features/popden/popden.csv"
  # gen_popden_feat(popden_raw_file,
  #                output_csv=popden_feat_csv)

  # Mean Age
  meanage_raw_file = "./data/raw/meanage/meanage.csv"
  meanage_feat_csv = "./data/features/meanage/meanage.csv"
  # gen_meanage_feat(meanage_raw_file,
  #                 output_csv=meanage_feat_csv)

  # Mean Temperature
  temp_raw_file = "./data/raw/temp/temp.csv"
  temp_feat_csv = "./data/features/temp/temp.csv"
  # gen_temp_feat(temp_raw_file,
  #              output_csv=temp_feat_csv)

  # Population Risk - DONE
  poprisk_raw_file = "./data/raw/poprisk/poprisk.csv"
  poprisk_feat_png = "./data/features/poprisk/poprisk.png"
  poprisk_feat_csv = "./data/features/poprisk/poprisk.csv"
  #gen_poprisk_feat(poprisk_raw_file,
  #                 output_png=poprisk_feat_png,
  #                 output_csv=poprisk_feat_csv)

  # Gov. Measures 1 - Lockdown
  lockdown_raw_file = "./data/raw/govme/lockdown.csv"
  lockdown_feat_csv = "./data/features/govme/lockdown.csv"
  # gen_lockdown_feat(lockdown_raw_file,
  #                 output_csv=lockdown_feat_csv)

  # Gov. Measures 2 - Borders Closed
  borcls_raw_file = "./data/raw/govme/borcls.csv"
  borcls_feat_csv = "./data/features/govme/borcls.csv"
  # gen_borcls_feat(lockdown_raw_file,
  #                output_csv=lockdown_feat_csv)

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