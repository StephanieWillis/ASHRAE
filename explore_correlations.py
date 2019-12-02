import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def produce_weekly_average_data(df, weekly_attributes=const.WEEKLY_ATTRIBUTES):
    df_weekly_means = df[weekly_attributes].groupby(['building_id', 'week_number']).mean()

    #add building id back in as a column for later operations
    df_weekly_means['building_id'] = list(df_weekly_means.index.get_level_values('building_id'))
    return df_weekly_means


def produce_weekly_degree_hours(df, set_point_temp):
    df_temp = df[['meter_reading','air_temperature','week_number']].copy()
    df_temp['delta_temp'] = df_temp['air_temperature'] - set_point_temp
    df_temp = df_temp.drop(columns='air_temperature')
    df_weekly_dh = df_temp.groupby(['building_id', 'week_number']).sum()
    df_weekly_dh['building_id'] = list(df_weekly_dh.index.get_level_values('building_id'))
    return df_weekly_dh


def loop_through_meters_apply_correlation_operation(correlation_function, all_meters_dict, correlators):
    dict_of_returns = {}
    for meter_type, all_df in all_meters_dict.items():
        print("Meter type: ", meter_type)
        dict_of_returns[meter_type] = correlation_function(all_df, correlators=correlators)
        plot_correlations(dict_of_returns[meter_type])
    return dict_of_returns