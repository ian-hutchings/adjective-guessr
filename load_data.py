"""
Ian Hutchings | Section AA
This file loads, cleans, and stores the data for this project in a csv
file called "wiki_cache.csv".
"""


import pandas as pd
from pandas import DataFrame
import numpy as np
import wikipediaapi
import pycountry_convert as pc
import spacy
from invalid_adjectives import INVALID_ADJECTIVES


def main():
    cities = pd.read_csv('./worldcities.csv')
    cities = clean_data(cities)
    cities['adjectives'] = cities.apply(
        lambda row: get_page_adj_dictionary(row['city'], row['country']),
        axis=1
    )
    cities = cities[cities['adjectives'].apply(lambda x: len(x) >= 5)]
    cities.to_csv("wiki_cache.csv", index=False)
    # Source: https://pandas.pydata.org/docs/reference
    # /api/pandas.DataFrame.to_csv.html


def clean_data(cities: DataFrame) -> DataFrame:
    """
    Given the original cities dataframe (not including Wikipedia data),
    Returns a cleaned and formatted version of the data by removing
    irrelevant columns, adding a continent column representing the
    continent that the country each city is in belongs to, removing rows
    that do not have a continent, and randomly selecting 20,000 cities from
    the dataframe to be the new dataset, as to reduce future runtimes.

    Sources:
    https://www.geeksforgeeks.org/python
    /how-to-randomly-select-rows-from-pandas-dataframe/
    https://stackoverflow.com/questions/29314033
    /drop-rows-containing-empty-cells-from-a-pandas-dataframe
    """
    cities = cities.loc[:, ['city', 'lat', 'lng', 'country', 'iso2']]
    cities['continent'] = cities['iso2'].apply(country_to_continent)
    cities['continent'].replace('', np.nan, inplace=True)
    cities = cities.dropna(subset=['continent'])
    cities = cities.sample(n=min(len(cities), 20000), random_state=00)
    return cities


def get_page_adj_dictionary(city_name: str,
                            country_name: str) -> dict[str, int]:
    """
    Given the name of city,
    Returns a dictionary mapping adjectives in the city's Wikipedia page to
    the number of times they appear in the page.
    """
    freq_dict = {}
    text = get_wiki_text(city_name, country_name)
    if not text:
        return freq_dict

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "ADJ":
            token = token.text.lower()
            if token not in INVALID_ADJECTIVES:
                if token not in freq_dict:
                    freq_dict[token] = 0
                freq_dict[token] += 1

    sorted_items = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:50]
    return dict(top_items)


def country_to_continent(country_code: str) -> str:
    """
    Given a country code,
    Returns the continent that country is a part of.
    """
    try:
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        continent_name = pc.convert_continent_code_to_continent_name(
                                                            continent_code)
        return continent_name
    except (KeyError, Exception):
        return ""


def get_wiki_text(city_name: str, country_name: str) -> str:
    """
    Given the name of city,
    Returns the Wikipedia page text of the city.
    Returns an empty String if the page does not meet the page requirements.
    """
    wiki = wikipediaapi.Wikipedia(language='en',
                                  user_agent=('CityAdjectiveCollector/1.0' +
                                              '(ianhu@uw.edu)'))
    page = wiki.page(city_name)
    page_requirements = (page.exists and len(page.text) >= 1000 and
                         country_name in page.text[:200] and
                         "disambiguation" not in page.text[:200] and
                         "may refer to" not in page.text[:200] and
                         ("city" in page.text[:200] or "town" in
                          page.text[:200] or "municipality" in
                          page.text[:200]))
    if page_requirements:
        print(city_name, "exists! Starts with:", page.text[:20])
        return page.text
    print(city_name, "does not exist.")
    return ""


if __name__ == '__main__':
    main()