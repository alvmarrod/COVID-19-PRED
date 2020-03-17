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
  
  date_format = r'%m-%d-%Y'

  structure = [0, 0, 0]
  latest_file = dt.datetime.strptime("01-01-1990", date_format)

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
      
      # 1. Is the file a data file?
      pattern = re.compile(r"^((0|1)\d{1})-((0|1|2)\d{1})-((19|20)\d{2})")
      if pattern.match(data):

        # 2. Is it newer than the latest recognized?
        file_name = data.split(".")[0]

        if dt.datetime.strptime(file_name, self.date_format).date() > self.latest_file.date():
          # Keep it
          self.latest_file = dt.datetime.strptime(file_name, self.date_format)
          # print(file_name)