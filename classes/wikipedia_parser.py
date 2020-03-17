import re
import datetime as dt
from html.parser import HTMLParser

class wikipedia_parser(HTMLParser):
  """ This class aims to parse the wikipedia HTML to retrieve both population
  and population density for a given country page

  Look up for structure like:
  tr
    th
      "Population"
  tr
    th
      Estimate/Census/Density
    td
      Value[/km, in case of density]
  Based on the Estimae/Census/Density we're extracting population or density.

  Returns
  --
  (people/km2, total people amount)

  """

  task_finished = False

  population_sideblock = False
  # Structure: tr, th
  steps = [0, 0]

  data_steps = [0, 0]

  # Aux var to store data to be found
  data_2_find = "None"

  population = 0
  density = 0

  def handle_starttag(self, tag, attrs):

    if not self.task_finished:

      if not self.population_sideblock:

        if (tag == "tr"):
          self.steps[0] = 1
        elif (tag == "th") and (self.steps[0] == 1):
          self.steps[1] = 1

      else:

        if (tag == "tr"):
          self.data_steps[0] = 1
        elif (tag == "th") and (self.data_steps[0] == 1):
          self.data_steps[1] = 1


  def handle_endtag(self, tag):

    if not self.task_finished:

      if self.population_sideblock:

        if (tag == "tr"):
          self.data_steps[0] = 0
        elif (tag == "th"):
          self.data_steps[1] = 0

      else:

        if (tag == "tr"):
          self.data_steps[0] = 1
        elif (tag == "th"):
          self.data_steps[1] = 1



  def handle_data(self, data):

    if not self.task_finished:

      if ("Population" in data) and (sum(self.steps) == 2) and not self.population_sideblock:
        self.population_sideblock = True

      elif self.population_sideblock and (sum(self.steps) == 2) and (self.data_2_find == "None"):
        if ("census" in data.lower()) or ("estimate" in data.lower()):
          self.data_2_find = "Population"
        elif ("density" in data.lower()):
          self.data_2_find = "Density"

        # print(f"Structure: {self.steps}\nFound: {self.data_2_find}\nData: {data}\n----")

      else:

        if self.data_2_find == "Population":

          # Preprocess
          data = data.strip()
          # print(f"Data: \"{data}\"")

          # If data is empty, it can be because there's a list inside it. We avoid
          # doing anything and wait for the next element content
          if data != "":

            data = data.replace(',', '')
            if ' ' in data:
              # print(f"Space found in data: \"{data}\"")
              data_array = data.split(' ')
              if "million" in data_array[1]:
                data = float(data_array[0]) * 1000000
              else:
                data = data_array[0]

            self.population = int(data)
            # Don't get new data until new data row
            self.data_2_find = "None"

        elif self.data_2_find == "Density" and "km" in data:
          # We avoid undesired html elements between density and its value on right side

          print(f"Data: \"{data}\"")

          # Preprocess
          if "sq" in data:
            data_array = data.split(' ')
            for part in data_array:
              data = part.replace('(', '') if "km" in part else data

          # data = data.replace('/km', '')
          data = data.split('/')[0]
          data = data.replace(',', '')
          data = int(round(float(data)))
            
          self.density = data
          self.task_finished = True

        # print(f"Structure: {self.steps}\nFound: {self.data_2_find}\nData: {data}\n----")

      