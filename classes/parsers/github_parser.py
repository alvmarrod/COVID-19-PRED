import re
import datetime as dt
from html.parser import HTMLParser

class github_parser(HTMLParser):
  """ This class aims to parse the github folder HTML to retrieve the latest file
  taking into account their name

  Look up for structure like:
  td.content
    span
      a
        file_name.csv
  Where file_name has the format: MM-DD-YYYY.csv
  """

  structure = [0, 0, 0]

  files_dict = []

  def handle_starttag(self, tag, attrs):
    if (tag == "td") and (self.structure[0] == 0):
      for attr in attrs:
        if attr[0] == "class" and attr[1] == "content":
          self.structure[0] = 1
    elif (tag == "span") and (self.structure[1] == 0):
      self.structure[1] = 1
    elif (tag == "a") and (self.structure[2] == 0):
      self.structure[2] = 1

  def handle_endtag(self, tag):
    if (tag == "td") and (self.structure[0] == 1) and (sum(self.structure) == 1):
      self.structure[0] = 0
    elif (tag == "span") and (self.structure[1] == 1) and (sum(self.structure) == 2):
      self.structure[1] = 0
    elif (tag == "a") and (self.structure[2] == 1) and (sum(self.structure) == 3):
      self.structure[2] = 0

  def handle_data(self, data):
    if sum(self.structure) == 3:
      
      # Is the file a data file?
      regex = r"(time_series_covid19_)(confirmed|deaths|recovered)(_global.csv)"
      pattern = re.compile(regex)

      if pattern.match(data):
        file_name = data.split(".")[0]
        self.files_dict.append(file_name)