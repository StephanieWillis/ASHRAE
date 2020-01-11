import ashrae_constants as const
import prepare_data as prep
import explore_correlations as exp
import models as mod


# train_all_dict = prep.get_joined_data(dataset_names=['train'])

small_train_all_dict = prep.produce_and_cache_small_dataset_dict('train')

# Assume that form of equation is meter_reading = A.T(t) + B + f(t) where B = A.T_setpoint.

# hourly temperatures
hourly_temps_coeffs = exp.loop_through_meters_apply_operation(mod.fit_per_building_linear_regression,
                                                              small_train_all_dict,
                                                              feature_name='air_temperature')
#weekly average temperatures
weekly_means_dict = exp.loop_through_meters_apply_operation(exp.produce_weekly_average_data,
                                                            small_train_all_dict,
                                                            weekly_attributes=const.WEEKLY_ATTRIBUTES)
weekly_temps_coeffs = exp.loop_through_meters_apply_operation(mod.fit_per_building_linear_regression,
                                                              weekly_means_dict,
                                                              feature_name='air_temperature')
