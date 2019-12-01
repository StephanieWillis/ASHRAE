import prepare_data as prep

#raw data
raw_data= prep.get_data('ashrae-energy-prediction', cache_file='store_raw.h5')
prep.print_nan_counts_all_dfs(raw_data)
thresholds = [1_00_000, 1_000_000, 10_000_000]
higher_meter_buildings = exp.get_buildings_with_high_meter(raw_data['train'], thresholds)
prep.count_missing_timestamps(raw_data['weather_train'])
prep.count_missing_timestamps(raw_data['weather_test'])

# clean data
cleaned_data = prep.clean_data(raw_data)
prep.print_nan_counts_all_dfs(cleaned_data)
prep.count_missing_timestamps(cleaned_data['weather_train'])
prep.count_missing_timestamps(cleaned_data['weather_test'])

