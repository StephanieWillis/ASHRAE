# from sklearn.linear_model import LinearRegression
# from sklearn.impute import SimpleImputer


import ashrae_constants as const
import prepare_data as prep
import explore_correlations as exp

# train_all_dict = prep.get_joined_data(dataset_names=['train'])

small_train_all_dict = prep.produce_and_cache_small_dataset_dict('train')
small_test_all_dict = prep.produce_and_cache_small_dataset_dict('test')

elec_train = small_train_all_dict['electricity']
elec_train_correlations = exp.produce_correlation_df(elec_train, buildings=None)
exp.plot_correlations(elec_train_correlations)

hourly_correlations = exp.loop_through_meters_apply_operation(exp.produce_correlation_df,
                                                              small_train_all_dict,
                                                              correlators=const.CORRELATORS)

elec_weekly_means = exp.produce_weekly_average_data(elec_train)
elec_weekly_dh = exp.produce_weekly_degree_hours(elec_train, set_point_temp=12)

elec_weekly_means_correlations = exp.produce_correlation_df(elec_weekly_means)
elec_weekly_dh_correlations = exp.produce_correlation_df(elec_weekly_dh, correlators=['delta_temp'])

weekly_means_dict = {meter_type: exp.produce_weekly_average_data(df, weekly_attributes=const.WEEKLY_ATTRIBUTES)
                     for meter_type, df in small_train_all_dict.items()}
weekly_mean_correlations = exp.loop_through_meters_apply_operation(exp.produce_correlation_df,
                                                                   weekly_means_dict,
                                                                   correlators=const.WEEKLY_CORRELATORS)

exp.loop_through_meters_apply_operation(exp.plot_correlation_distribution, weekly_mean_correlations)

exp.compare_distributions({'weekly average air temp': elec_weekly_means_correlations['air_temperature'],
                           'degree hours': elec_weekly_dh_correlations['delta_temp']},
                          'comparing degree hours with weekly averages')


