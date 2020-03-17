# COVID-19 Prediction

This repo aims to do an analysis on COVID-19 spreading patterns based on worldwide data.

## Data Source

* As base data for `COVID-19` in our analysis we're using [Johns Hopkins repository](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports).

* As `population density` data we're fetching the latest data available from [World Population Review](https://worldpopulationreview.com/countries/). Towards this purpose, we've created the script [`get_population_density.py`](./utilities/get_population_density.py) which is feed with a `CSV` file exported from this website.

* As `temperature` data, we're fetching the data from ...

* As `governments countermeasures` data, we've collected them from ...

## Utilities

### Population Density

As `population density` data we're fetching the latest data available from the file [Population CSV](./../data/population.csv) provided in [World Population Review](https://worldpopulationreview.com/countries/) to feed our models. Towards this purpose, we've created the script [`get_population_density.py`](./get_population_density.py).

This script:

* Provides the function to read this `CSV`
* Returns the data filtered by needed countries as a Pandas Dataframe

## Dependencies

* Pandas
* Numpy
* Matplotlib
* [tqdm](https://github.com/tqdm/tqdm)