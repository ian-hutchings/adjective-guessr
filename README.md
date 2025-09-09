## Background

For information regarding this project's research questions, findings, etc. read report.pdf.

## Download information
To run this project, you will need to install:
* wikipediaapi: a Python wrapper for Wikipedia's API
  * Link: https://pypi.org/project/Wikipedia-API/
* Several Python libraries, including:
  * pandas
  * geopandas
  * numpy
  * matplotlib
  * pycountry_convert
  * spacy
  * shapely
  * ast
  * scipy
  * sklearn
  * pytest
  * re


## File information
`cse_163_final_project.py`: This file contains the code that directly
answers this project's 3 research questions.

`load_data.py`: This file loads, cleans, and stores the data sources for
this project in a csv file called `wiki_cache.csv`.

`testing.py`: This file tests the functions in `cse_163_final_project.py`
and `load_data.py`

`ivalid_adjectives.py`: This file contains a list of adjectives considered
"invalid" for this project by the metrics given in the file.

`wiki_cache.csv`: The updated and cleaned data file that is analyzed in
`cse_163_final_project.py` (contains Wikipedia data).

`worldcities.csv`: The original csv file containing world city data
(does not include Wikipedia data).
Link: https://simplemaps.com/data/world-cities

`test_data.csv`: Small data file used for testing.

`test_data_wiki.csv`: Small data file used for testing (includes Wiki data).

`ne_110m_admin_0_countries`: This folder stores the shape data necessary to
plot a map of the world.
Link: https://www.naturalearthdata.com/downloads/110m-cultural-vectors/

`report.pdf`: File summarizing project research questions, methods, findings,
and impacts and limitations.

## Running instructions
Running `load_data.py` populates the file `wiki_cache.csv` with the data that
it currently stores. To verify it's working, delete `wiki_cache.csv`'s current
contents and run the `load_data.py` file. The process is slow (~3.5 hours)
because it fetches the Wikipedia page of and generates an adjective dictionary
for every city.

Running `cse_163_final_project.py` executes the code that answers the project
research questions. All the functions are relatively quick
(runtime < 1 minute), except for `regression_analysis` which takes about 8
minutes to run.
