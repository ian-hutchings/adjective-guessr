"""
Ian Hutchings | Section AA
This file contains functions that analyze the data in "wiki_cache.csv",
answering the 3 research questions of this project.
"""

import pandas as pd
from matplotlib.axes import Axes
from pandas import DataFrame
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
import ast
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (mean_absolute_error, accuracy_score,
                             classification_report)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder


def main():
    cities = pd.read_csv('./wiki_cache.csv')
    cities["adjectives"] = cities["adjectives"].apply(ast.literal_eval)

    fig, ax = plt.subplots(figsize=(15, 10))
    plot_world_cities(cities, ax)

    create_overall_adjective_chart(cities)

    compute_cos_similarity(cities)

    fig, ax2 = plt.subplots(figsize=(15, 10))
    regression_analysis(cities, ax2)

    classification_analysis(cities)


def plot_world_cities(cities: DataFrame, ax: Axes) -> None:
    """
    Given the cities dataframe and a plotting axis,
    Plots a world map of the location of all cities in the dataframe.
    """
    world = gpd.read_file('./ne_110m_admin_0_countries' +
                          '/ne_110m_admin_0_countries.shp')
    geometry = [Point(xy) for xy in zip(cities['lng'], cities['lat'])]
    geo_df = gpd.GeoDataFrame(cities, geometry=geometry)
    world.plot(ax=ax, color='lightgray', edgecolor='black')
    geo_df.plot(ax=ax, color='red', markersize=50)

    plt.title("All Cities in Dataset Plotted on World Map")
    plt.show()


def create_overall_adjective_chart(cities: DataFrame) -> None:
    """
    Given the cities dataframe,
    Plots a pie chart of the most common adjectives across all the cities'
    Wikipedia pages.
    """
    overall_dict = {}
    for col, row in cities.iterrows():
        page_dict = row['adjectives']
        for adjective in page_dict.keys():
            if adjective not in overall_dict:
                overall_dict[adjective] = 0
            overall_dict[adjective] += page_dict[adjective]

    sorted_items = sorted(overall_dict.items(),
                          key=lambda item: item[1], reverse=True)
    top_items = sorted_items[:20]
    labels = [item[0] for item in top_items]
    sizes = [item[1] for item in top_items]

    print("20 most common adjectives:", labels)

    plt.pie(sizes, labels=labels,
            textprops={'fontsize': 10}, labeldistance=1.2)
    plt.title("20 Most Common Adjectives Across Entire Dataset")
    plt.show()


def compute_cos_similarity(cities: DataFrame) -> None:
    """
    Given the cities dataframe,
    Computes the similarity of adjectives used in the Wikipedia pages of
    cities on the same continent and different continents, measured using
    cosine similarity. A higher number means cities in that group have
    more similar use of adjectives in their Wiki pages.
    Prints the similarity scores for both groups and the results of a
    two-sample t-test. If the p-value is below 0.05, there is strong evidence
    that the two group's scores are significantly different.

    Sources:
    https://scikit-learn.org/stable/modules/generated
    /sklearn.feature_extraction.DictVectorizer.html
    https://www.geeksforgeeks.org/python
    /how-to-calculate-cosine-similarity-in-python/
    https://www.geeksforgeeks.org/python
    /difference-between-eval-and-ast-literal-eval-in-python/
    https://www.technologynetworks.com/informatics
    /articles/mann-whitney-u-test-assumptions-and-example-363425
    """
    vectorizer = DictVectorizer(sparse=False)
    count_matrix = vectorizer.fit_transform(cities["adjectives"])
    similarity_matrix = cosine_similarity(count_matrix)

    continents = cities["continent"].values
    same_continent_sims = []
    diff_continent_sims = []
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            sim = similarity_matrix[i, j]
            if continents[i] == continents[j]:
                same_continent_sims.append(sim)
            else:
                diff_continent_sims.append(sim)

    print("Average cosine similarity (all cities):",
          np.mean(similarity_matrix))
    print("Average cosine similarity (same continent):",
          np.mean(same_continent_sims))
    print("Average similarity (different continents):",
          np.mean(diff_continent_sims), "\n")

    t_stat, p_value = ttest_ind(same_continent_sims,
                                diff_continent_sims, equal_var=False)
    print("T-statistic:", t_stat, "p-value:", p_value)
    if p_value < 0.05:
        print("p-value is less than 0.05! There is strong evidence that",
              "there is a significant difference between countries on the",
              "same and different continents.\n")
    else:
        print("p-value is not less than 0.05. There is not strong evidence",
              "that there is a significant difference between countries on",
              "the same and different continents.\n")


def regression_analysis(cities: DataFrame, ax2: Axes) -> None:
    """
    Given the cities dataframe and a plotting axis,
    Trains and tests a machine learning regression model to predict the
    lat, lng coordinates of a city based on the adjectives and their counts
    in the city's Wikipedia page.
    Prints the mean absolute error of the model's predictions for both
    latitude and longitude.
    Plots a world map visualizing the difference between the predicted
    coordinates and actual coordinates of a random set of 20 city predictions.

    Sources:
    https://www.reddit.com/r/statistics/comments/17innzv
    /q_when_would_a_decision_tree_underperform_a/
    https://www.geeksforgeeks.org/machine-learning
    /random-forest-regression-in-python/
    https://stackoverflow.com/questions/19155718
    /select-pandas-rows-based-on-list-index
    """
    X = pd.DataFrame(list(cities['adjectives'])).fillna(0).astype(int)
    y = cities[['lat', 'lng']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=00)

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200,
                                                       random_state=00))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    print("Mean Absolute Error - Latitude:", mae[0],
          "Longitude:", mae[1], "\n")

    sample_indices = np.random.choice(len(y_test), size=min(len(y_test), 20),
                                      replace=False)
    plot_regression_predictions(y_pred[sample_indices],
                                y_test.iloc[sample_indices], ax2)


def plot_regression_predictions(y_pred: list[list[float]],
                                y_test: DataFrame, ax2: Axes) -> None:
    """
    Given a list of predicted coordinates (y_pred) and a dataframe of actual
    coordinates (y_test) for a random set of 20 city predictions,
    Plots a world map visualizing the difference between the predicted
    coordinates and actual coordinates.
    (Helper method for regression_analysis).

    Sources:
    https://www.reddit.com/r/learnpython/comments/mgvd2e
    /geopanadas_plotting_a_line_between_two_coordinates/
    """
    predictions = pd.DataFrame({
        'pred_lat': y_pred[:, 0],
        'pred_lng': y_pred[:, 1]
    })
    actual = pd.DataFrame({
        'actual_lat': y_test.loc[:, 'lat'],
        'actual_lng': y_test.loc[:, 'lng'],
    })
    world = gpd.read_file('./ne_110m_admin_0_countries' +
                          '/ne_110m_admin_0_countries.shp')
    actual_points = [Point(xy) for xy in zip(actual['actual_lng'],
                                             actual['actual_lat'])]
    pred_points = [Point(xy) for xy in zip(predictions['pred_lng'],
                                           predictions['pred_lat'])]
    geo_actual = gpd.GeoDataFrame(actual, geometry=actual_points)
    geo_pred = gpd.GeoDataFrame(predictions, geometry=pred_points)

    world.plot(ax=ax2, color='lightgray', edgecolor='black')
    geo_actual.plot(ax=ax2, color='green', marker='o', markersize=50,
                    label='Actual', legend=True)
    geo_pred.plot(ax=ax2, color='red', marker='o', markersize=50,
                  label='Predicted', legend=True)
    for (_, actual_row), predicted in zip(y_test.iterrows(), y_pred):
        plt.plot(
            [actual_row['lng'], predicted[1]],
            [actual_row['lat'], predicted[0]],
            color='gray', alpha=0.3, linewidth=0.8
        )

    ax2.legend(loc='upper left', frameon=True)
    plt.title("Regression Model City Predictions")
    plt.show()


def classification_analysis(cities: DataFrame) -> None:
    """
    Given the cities dataframe,
    Trains and tests a machine learning classification model to predict the
    continent that a city's country is in based on the adjectives and their
    counts in the city's Wikipedia page.
    Prints the model's accuracy score, as well as a classification report
    detailing the model's accuracy for each continent.

    Sources:
    https://scikit-learn.org/stable/modules/generated
    /sklearn.ensemble.RandomForestClassifier.html
    https://scikit-learn.org/stable/modules
    /generated/sklearn.preprocessing.LabelEncoder.html
    """
    X = pd.DataFrame(list(cities['adjectives'])).fillna(0).astype(int)
    y = cities['continent']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=00
    )

    model = RandomForestClassifier(n_estimators=200, random_state=00)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Overall classification accuracy: " + str(acc*100) + "%")

    print("Classification report:")
    print(classification_report(y_test, y_pred,
                                labels=np.arange(len(label_encoder.classes_)),
                                target_names=label_encoder.classes_,
                                zero_division=0))


if __name__ == '__main__':
    main()