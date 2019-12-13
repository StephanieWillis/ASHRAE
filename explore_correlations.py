import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ashrae_constants as const


# processing correlations
def produce_correlation_df(train_df, y_variable='meter_reading', buildings=None, correlators=const.CORRELATORS):
    """
    Returns df showing correlations between correlators and y_variable for buildings in train df
    """
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


def produce_weekly_average_data(df, weekly_attributes=const.WEEKLY_ATTRIBUTES):
    """
    Averages the data for each building for each week
    """
    df_weekly_means = df[weekly_attributes].groupby(['building_id', 'week_number']).mean()
    #add building id back in as a column for later operations
    df_weekly_means['building_id'] = list(df_weekly_means.index.get_level_values('building_id'))
    return df_weekly_means


def produce_weekly_degree_hours(df, set_point_temp):
    """
    Calculates degree hours at set_point_temp and integrates over week
    """
    df_temp = df[['meter_reading','air_temperature','week_number']].copy()
    df_temp['delta_temp'] = df_temp['air_temperature'] - set_point_temp
    df_temp = df_temp.drop(columns='air_temperature')
    df_weekly_dh = df_temp.groupby(['building_id', 'week_number']).sum()
    df_weekly_dh['building_id'] = list(df_weekly_dh.index.get_level_values('building_id'))
    return df_weekly_dh


def loop_through_meters_apply_operation(operation, all_meters_dict, *args, **kwargs):
    """
    apply function to each meter specific df and return dict of returned values.
    """
    dict_of_returns = {}
    for meter_type, all_df in all_meters_dict.items():
        print("Meter type: ", meter_type)
        dict_of_returns[meter_type] = operation(all_df, *args, **kwargs)
    return dict_of_returns


# plotting correlations
def plot_correlation_distribution(correlations_df):
    """
    Plots the distribution of the correlations coefficients for each variable across the buildings in correlations df
    """
    for variable in correlations_df.columns:
        sns.distplot(list(correlations_df[variable].dropna()), label=variable)
    plt.legend()
    plt.show()


def plot_correlation_values(correlations_df):
    """
    Plots the actual correlation coefficients for each variable with the buildings on the x axis
    """
    correlations_df.drop(['mean', 'st_dev'], axis=0).plot()
    plt.show()


def plot_correlations(buildings_correlations):
    """
    Plots actual correlations, correlation distributions, and then correlation distributions for only air temp, dew temp
    """
    plot_correlation_values(buildings_correlations)
    plot_correlation_distribution(buildings_correlations)
    plot_correlation_distribution(buildings_correlations[['air_temperature', 'dew_temperature']])


def compare_distributions(df_dict, title):
    """
    df_dict is of format {variable_name: df_column}
    where variable name is label and df_column is the column to plot the distribution of.
    """
    for variable_name, df_column in df_dict.items():
        sns.distplot(list(df_column.dropna()), label=variable_name)
    plt.title(title)
    plt.legend()
    plt.show()


def compare_variable_across_building_uses(df, buildings_df, variable, title):
    """
    Plot the distribution of a variable in df for all the uses in buildings_df
    """
    for use in set(buildings_df['primary_use']):
        sns.distplot(list(df[df['primary_use'] == use][variable].dropna()), label=use)
        plt.title(title)
        # Put the legend out of the figure
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
