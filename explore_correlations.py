import pandas as pd
import matplotlib.pyplot as plt

import ashrae_constants as const


def produce_correlation_df(train_df, y_variable='meter_reading', buildings=None, correlators=const.CORRELATORS):
    if buildings == None:
        buildings = list(set(train_df.index.get_level_values('building_id')))
    correlations_df = pd.DataFrame()
    for building_id in buildings:
        df = train_df.loc[building_id]
        correlations = df.corr()[y_variable]
        correlations = correlations[correlators]
        correlations_df[building_id] = correlations
    correlations_df['mean'] = correlations_df.mean(axis=1)
    correlations_df['st_dev'] = correlations_df.std(axis=1)
    correlations_df = correlations_df.T
    correlations_df.sort_values(by='mean', axis=1, ascending=False)

    return correlations_df


def plot_correlation_distribution(correlations_df):
    for variable in correlations_df.columns:
        sns.distplot(list(correlations_df[variable].dropna()), label=variable)
    plt.legend()
    plt.show()


def plot_correlation_values(correlations_df):
    correlations_df.drop(['mean', 'st_dev'], axis=0).plot()
    plt.show()


def plot_correlations(buildings_correlations):
    plot_correlation_values(buildings_correlations)
    plot_correlation_distribution(buildings_correlations)
    plot_correlation_distribution(buildings_correlations[['air_temperature', 'dew_temperature']])
