"""
Ian Hutchings | Section AA
This file contains the testing functions for the functions in the
load_data.py and cse_163_final_project.py files.
"""


import ast
import re
import pandas as pd
from pandas import DataFrame
import pytest
from _pytest.capture import CaptureFixture
from matplotlib import pyplot as plt


from load_data import (
    clean_data,
    get_page_adj_dictionary,
    country_to_continent,
    get_wiki_text
)
from cse_163_final_project import (
    plot_world_cities,
    create_overall_adjective_chart,
    compute_cos_similarity,
    regression_analysis,
    classification_analysis
)


def main():
    test_clean_data()
    test_get_page_adj_dictionary()
    test_country_to_continent()
    test_get_wiki_text()
    test_plot_world_cities()
    test_create_overall_adjective_chart()
    test_compute_cos_similarity()
    test_regression_analysis_and_plot()
    test_classification_analysis()
    print("All tests passed!")


def test_clean_data() -> None:
    """
    Given a test dataframe,
    Tests the clean_data function in load_data.py.

    Sources:
    https://pandas.pydata.org/docs/reference/api
    /pandas.testing.assert_frame_equal.html
    """
    # Test case 1: Valid entries, 5
    test_df = pd.read_csv("test_data.csv")
    result_df = clean_data(test_df)
    correct_result = {"city": ["Tokyo", "Jakarta", "Delhi", "Guangzhou",
                               "Mumbai"],
                      "lat": [35.6870, -6.1750, 28.6100, 23.1300, 19.0761],
                      "lng": [139.7495, 106.8275, 77.2300, 113.2600, 72.8775],
                      "country": ["Japan", "Indonesia", "India", "China",
                                  "India"],
                      "iso2": ["JP", "ID", "IN", "CN", "IN"],
                      "continent": ["Asia", "Asia", "Asia", "Asia", "Asia"]}
    correct_df = pd.DataFrame(correct_result).sample(n=5, random_state=00)
    pd.testing.assert_frame_equal(result_df, correct_df)

    # Test case 2: Invalid entry, 2
    test_data = {"city": ["Tokyo", "Made up city"],
                 "lat": [35.6870, -70.8860],
                 "lng": [139.7495, 613.2275],
                 "country": ["Japan", "Made up country"],
                 "iso2": ["JP", "MCC"],
                 "other col": ["OOO", "LLL"]}
    test_df = pd.DataFrame(test_data)
    result_df = clean_data(test_df)
    correct_result = {"city": ["Tokyo"],
                      "lat": [35.6870],
                      "lng": [139.7495],
                      "country": ["Japan"],
                      "iso2": ["JP"],
                      "continent": ["Asia"]}
    correct_df = pd.DataFrame(correct_result).sample(n=1, random_state=00)
    pd.testing.assert_frame_equal(result_df, correct_df)


def test_get_page_adj_dictionary() -> None:
    """
    Tests the get_page_dictionary function in load_data.py.
    Tests using the Wikipedia page for Reykjavík:
    https://en.wikipedia.org/wiki/Reykjav%C3%ADk
    """
    result = get_page_adj_dictionary("Reykjavík", "Iceland")
    assert isinstance(result, dict)
    assert "urban" in result
    assert result["economic"] == 5
    assert "arid" not in result

    result = get_page_adj_dictionary("HHHHHHHHHHH", "HHH")
    assert result == {}


def test_country_to_continent() -> None:
    """
    Tests the country_to_continent function in load_data.py.
    """
    assert country_to_continent("US") == "North America"
    assert country_to_continent("FR") == "Europe"
    assert country_to_continent("BR") == "South America"
    assert country_to_continent("XX") == ""


def test_get_wiki_text() -> None:
    """
    Tests the get_wiki_text function in load_data.py.
    """
    text = get_wiki_text("Paris", "France")
    assert isinstance(text, str)
    assert "Luteciam" in text

    text = get_wiki_text("Lebanon", "United States")
    assert text == ""
    # Directs to Lebanon country page, not the city in US. The page doesn't
    # meet the requirements, so returns an empty string.


@pytest.fixture
def sample_df() -> DataFrame:
    """
    Fixture: loads in a small csv file for testing.
    Returns a dataframe containing the file data.

    Sources:
    https://docs.pytest.org/en/stable/explanation/fixtures.html
    """
    result = pd.read_csv("./test_data_wiki.csv")
    result["adjectives"] = result["adjectives"].apply(ast.literal_eval)
    return result


@pytest.fixture
def real_df() -> DataFrame:
    """
    Fixture: loads in the full csv file for testing ML functions.
    Returns a dataframe containing the full city data.
    """
    result = pd.read_csv("./wiki_cache.csv")
    result["adjectives"] = result["adjectives"].apply(ast.literal_eval)
    return result


def test_plot_world_cities(sample_df: DataFrame) -> None:
    """
    Tests the plot_world_cities function in cse_163_final_project.py.
    Ensures that every city in the dataframe is plotted.

    Sources:
    https://stackoverflow.com/questions/27948126
    /how-can-i-write-unit-tests-against-code-that-uses-matplotlib
    https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html
    https://matplotlib.org/stable/api/collections_api.html
    """
    fig, ax = plt.subplots()
    plot_world_cities(sample_df, ax=ax)

    points_plotted = len(ax.collections[1].get_offsets())
    assert points_plotted == len(sample_df)


def test_create_overall_adjective_chart(sample_df: DataFrame,
                                        capsys: CaptureFixture[str]) -> None:
    """
    Tests the create_overall_adjective_chart function in
    cse_163_final_project.py.
    Checks the first most common adjective is printed correctly.

    Sources:
    https://docs.pytest.org/en/stable/how-to/capture-stdout-stderr.html
    """
    create_overall_adjective_chart(sample_df)
    captured = capsys.readouterr()
    assert "20 most common adjectives: ['new'," in captured.out
    assert "arid" not in captured.out


def test_compute_cos_similarity(sample_df: DataFrame,
                                capsys: CaptureFixture[str]) -> None:
    """
    Tests the compute_cosine_similarity function in cse_163_final_project.py.
    Checks that correct statement is printed for p-value greater or less than
    0.05.

    Sources:
    https://docs.python.org/3/library/re.html
    https://www.geeksforgeeks.org/python/re-search-in-python/
    """
    compute_cos_similarity(sample_df)
    captured = capsys.readouterr()
    assert "Average cosine similarity" in captured.out
    assert "p-value" in captured.out

    match = re.search(r"p-value:\s*([0-9.eE+-]+)", captured.out)
    assert match

    p_value = float(match.group(1))
    if p_value < 0.05:
        assert "p-value is less than 0.05!" in captured.out
    else:
        assert "p-value is not less than 0.05." in captured.out


def test_regression_analysis_and_plot(real_df: DataFrame,
                                      capsys: CaptureFixture[str]) -> None:
    """
    Tests the regression_analysis and plot_regression_predictions functions
    in cse_163_final_project.py. Ensures that all testing cities (min
    between 20 or 1/5 of data) doubled for predicted and actual coordinates
    are plotted. Ensures latitudinal error is less than 30 degrees.
    """
    fig, ax2 = plt.subplots()
    regression_analysis(real_df, ax2)
    captured = capsys.readouterr()
    assert "Longitude" in captured.out

    match = re.search(r"Mean Absolute Error - Latitude:\s*([0-9.eE+-]+)",
                      captured.out)
    assert match

    error = float(match.group(1))
    assert error <= 30, "latitudinal error is greater than 30 degrees"

    points_plotted = (len(ax2.collections[1].get_offsets()) +
                      len(ax2.collections[2].get_offsets()))
    assert points_plotted == 2 * min((len(real_df) / 5), 20)


def test_classification_analysis(real_df: DataFrame,
                                 capsys: CaptureFixture[str]) -> None:
    """
    Tests the classification_analysis function in cse_163_final_project.py.
    Ensures the classification accuracy is at least 25%.
    """
    classification_analysis(real_df)
    captured = capsys.readouterr()
    assert 'classification accuracy' in captured.out
    assert 'North America' in captured.out
    assert 'Asia' in captured.out

    match = re.search(r"classification accuracy:\s*([0-9.eE+-]+)",
                      captured.out)
    assert match

    accuracy = float(match.group(1))
    assert accuracy >= 25, "accuracy is less than 25%"


if __name__ == "__main__":
    main()