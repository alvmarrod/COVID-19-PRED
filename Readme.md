# COVID-19 Prediction

This repo aims to do an analysis on COVID-19 spreading patterns based on worldwide data.

## Raw - Data Sources

1. As base data for `COVID-19` in our analysis we're using [Johns Hopkins repository](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports). This data is stored into the [Covid Folder](./data/raw/covid/), where we replicate the content from the dialy reports. Don't panic, the data itself is downloaded by the script [main.py](./main.py) whenever it's run. We just maintain a copy here in our repo.

2. As `population density` data we're fetching the latest data available from [World Population Review](https://worldpopulationreview.com/countries/). Since this data is annual, we're keeping it stored in the repo without any change in [Population Density Folder](./data/raw/popden/).

3. As `temperature` data, we're fetching the data from ...

4. As `governments countermeasures` data, we've collected them from the following sources. Since this is the most subjective part, there has not been any data transformation, but directly features that will be in the next section.

* [Japan](https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Japan)
* [United Kingdom](https://www.telegraph.co.uk/politics/2020/03/16/uks-coronavirus-lockdown-has-already-begun-unofficially-northern/)
* [Finland](https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Finland)

## Features - Model Data

1. `COVID-19` data has been processed and saved from [Raw Covid Folder](./data/raw/covid/) to [Feature Covid Folder](./data/features/covid/).

2. The `population density` data has been processed and saved from [Raw Population Density Folder](./data/raw/popden/) to [Feature Population Density Folder](./data/features/popden/).

3. ...

4. The `governments measures` that we have collected and directly put into a feature, under our own criteria (that you can find in our Medium article), has been placed directly in the [Features Governments Measures Folder](./data/features/govme/).

## Dependencies

* Pandas
* Numpy
* Matplotlib
* [tqdm](https://github.com/tqdm/tqdm)