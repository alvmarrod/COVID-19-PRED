"""This file aims to create the input for the model
"""

import numpy as np
import pandas as pd

def expand_cases_to_vector(data):
  """Receives the cases dataframe and expands it, creating a row per each day
  time lapse.
  """

  dates_num = len(data.columns) - 1
  dates = data.columns[1:].tolist()
  countries = data["Country"].unique().tolist()

  #print(f"Total countries: {len(countries)}")
  #print(f"Total Dates: {dates_num}")

  # Empty template to fill with data
  rows = (dates_num-1) * len(countries)

  countries_array = []
  for country in countries:
    countries_array += [country] * (dates_num-1)

  countries_nparray = np.array(countries_array)

  output = pd.DataFrame( \
                      columns=[
                      "Country",
                      "Date",
                      "Cases",
                      "Popden",
                      "Masks",
                      "Poprisk",
                      "Lockdown",
                      "Borders",
                      "NextDay"
                    ])

  for j in range(len(countries)):

    # Cases data from the country row
    country_data = data[ data["Country"] == countries[j] ].values.tolist()
    country_data = country_data[0][1:]

    for i in range(dates_num-1):

      output.loc[len(output)] = [countries_array[j*7+i],
                                 dates[i],
                                 country_data[i], 
                                 0, 0, 0, 0, 0, 
                                 country_data[i+1]]

  # print(f"Temp: {output}")

  return output