import pathlib
import pandas as pd
import ashrae_constants as const
import itertools
from IPython.display import display


def import_data(ashrae_dir, filenames=const.NAMES):
    """
   Import ASHRAE data from a directory containing the .csv files.

   Return a {'thing': pd.Dataframe} dictionary.
   """
    ashrae_dir = pathlib.Path(ashrae_dir)
    data = {name: pd.read_csv((ashrae_dir / name).with_suffix('.csv')) for name in filenames}

    return data


def _cache_data(data, filename):
    """
   Given a data as a {str: pd.DataFrame} dictionary, save it to a .h5 file.
   """
    filename = pathlib.Path(filename)
    assert filename.suffix == '.h5'
    with pd.HDFStore(filename) as f:
        for name, df in data.items():
            f[name] = df


def get_raw_data(ashrae_dir, cache_file=None, filenames=const.NAMES):
    """
   Import ASHRAE data with optional caching mechanism.

   Return a {'thing': pd.Dataframe} dictionary.
   """
    cache_file = pathlib.Path(cache_file)

    if cache_file is not None and cache_file.exists():
        print(f'Importing data from {cache_file}')
        with pd.HDFStore(cache_file) as f:
            data = {name: f[name] for name in filenames}
    else:
        print('Importing data from csv')
        data = import_data(ashrae_dir)
        _cache_data(data, cache_file)

    # Sanity check: the set of building ids should be the same in the train and test sets.
    assert set(data['train'].building_id) == set(data['test'].building_id)

    return data


def print_nan_counts_all_dfs(data, names=const.NAMES):
    for name in names:
        print(f'NaNs for {name}')
        display(data[name].isna().sum())


def get_buildings_with_high_meter(df, thresholds):
    """
   Return a dict of {threshold: set of buildings with a meter reading above that threshold}
   """
    high_meter_buildings = {threshold: set(df.building_id[df.meter_reading > threshold]) for threshold in thresholds}
    for threshold, buildings in high_meter_buildings.items():
        print(f'There are {len(buildings)} buildings with meter readings above {threshold // 1_000_000}M: {buildings}')

    return high_meter_buildings


def count_missing_timestamps(df):
    """
    Return the number of timestamps missing
    """
    no_of_timestamps = len(df.timestamp)
    no_of_sites = len(set(df.site_id))
    full_date_range = pd.date_range(start=min(df.timestamp), end=max(df.timestamp), freq='H')
    no_of_missing_timestamps = no_of_sites * len(full_date_range) - no_of_timestamps
    print(f'There are {no_of_timestamps} timestamps in the data. The full date range is {len(full_date_range)} long and'
          f' there are {no_of_sites} sites so there should be {no_of_sites * len(full_date_range)} '
          f'timestamps in the data. There are therefore {no_of_missing_timestamps} missing. ')

    return no_of_missing_timestamps


def add_missing_weather_data(df):
    """ Add missing timestamps to weather data and interpolate to fill in the data
    return df with missing times and weather data filled in
    """

    full_date_range = pd.date_range(start=min(df.timestamp), end=max(df.timestamp), freq='H')
    sites = list(set(df.site_id))
    full_data_site_range = pd.DataFrame(itertools.product(sites, full_date_range),
                                        columns=['site_id', 'timestamp'])
    df_all_dates = full_data_site_range.merge(df, on=['site_id', 'timestamp'], how='left')
    df_all_dates = df_all_dates.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))

    return df_all_dates


def clean_data(raw_data, names=const.NAMES, meter_map=const.METER_MAP):
    """
    Convert timestamps to timestamp objects, fill in blanks in weather data, add names of meter types
    """

    cleaned_data = {}
    local_names = names.copy()
    if 'building_metadata' in local_names:
        local_names.remove('building_metadata')

    for name in local_names:
        print(f'Cleaning {name} dataset')
        df = raw_data[name]
        df.timestamp = pd.to_datetime(df.timestamp)
        if name.startswith('weather'):
            df = add_missing_weather_data(df)
        elif name in ['train', 'test']:
            df['meter_type'] = df['meter'].map(meter_map)
        cleaned_data[name] = df

    cleaned_data['building_metadata'] = raw_data['building_metadata']

    return cleaned_data


def join_input_data_and_multi_index(data, dataset_name):
    """Join together the meter data, weather data and building metadata into one df

    data = dict of df's (keys are'building_metadata', 'weather_train', 'weather_test', 'train','test')
    dataset_name = 'train' or 'test'
                    """

    meter_df = data[dataset_name]
    building_df = data['building_metadata']
    weather_df = data['weather_' + dataset_name]

    # join meter and weather data
    building_n_meter = meter_df.merge(building_df, on='building_id', how='left')
    joined_data = building_n_meter.merge(weather_df, on=['site_id', 'timestamp'], how='left')

    # Add time related columns
    joined_data['hour'] = joined_data['timestamp'].dt.hour
    joined_data['weekday'] = joined_data['timestamp'].dt.dayofweek
    joined_data['week_number'] = joined_data['timestamp'].dt.week
    joined_data['month'] = joined_data['timestamp'].dt.month

    joined_data['is_weekend'] = joined_data['weekday'].apply(lambda x: 1 if x in [0, 6] else 0)

    # multi index on building id and timestamp
    joined_data = joined_data.set_index(['building_id', 'timestamp']).sort_index()

    return joined_data


def split_on_meter_type(joined_data, meter_types):
    """ Split the joined data into a dict with a df for each meter type"""
    joined_data_dict = {meter_type: joined_data[joined_data['meter_type'] == meter_type]
                        for meter_type in meter_types}

    return joined_data_dict


def import_dict_from_cached(cache_file, key_list):
    print(f'Importing data from {cache_file}')
    with pd.HDFStore(cache_file) as f:
        data_dict = {key: f[key] for key in key_list}
    return data_dict


def get_joined_data(dataset_names=['train', 'test'],
                    joined_cache_filename_end='_store_joined.h5',
                    meter_types=const.METER_MAP.values(),
                    ashrae_dir='ashrae-energy-prediction',
                    raw_cache_file='store_raw.h5',
                    filenames=const.NAMES):
    """
    Return a dict of {meter_type: df} dictionary for either the train (default) or test datasets

   """

    joined_cache_filenames = {dataset: pathlib.Path(dataset + joined_cache_filename_end) for dataset in dataset_names}
    joined_data_dict = {}
    all_files_stored = [file.exists() for file in joined_cache_filenames.values()]

    if joined_cache_filename_end is not None and all(all_files_stored):
        for dataset in dataset_names:
            joined_data_dict[dataset] = import_dict_from_cached(joined_cache_filenames[dataset], meter_types)
    else:
        print('Reading, cleaning and joining data')
        raw_data = get_raw_data(ashrae_dir, cache_file=raw_cache_file, filenames=filenames)
        print('cleaning data')
        cleaned_data = clean_data(raw_data)
        for dataset in dataset_names:
            print('joining datasets - ' + dataset)
            joined_data = join_input_data_and_multi_index(cleaned_data, dataset)
            print('splitting on meter type - ' + dataset)
            joined_data_dict[dataset] = split_on_meter_type(joined_data, meter_types)
            print('caching resultant dataset - ' + dataset)
            _cache_data(joined_data_dict[dataset], joined_cache_filenames[dataset])

    return joined_data_dict


def produce_and_cache_small_dataset_dict(dataset_name,
                                         n=500000,
                                         meter_types=const.METER_MAP.values()):
    """
    dataset_name should be 'train' or 'test'
    """
    small_dataset_cache_file = pathlib.Path(dataset_name + '_small_store_joined.h5')

    if small_dataset_cache_file.exists():
        small_dataset_dict = import_dict_from_cached(small_dataset_cache_file, meter_types)
    else:
        big_dataset_cache_file = pathlib.Path(dataset_name + '_store_joined.h5')
        big_dataset_dict = import_dict_from_cached(big_dataset_cache_file, meter_types)
        small_dataset_dict = {key: big_dataset.head(n) for key, big_dataset in big_dataset_dict.items()}
        _cache_data(small_dataset_dict, small_dataset_cache_file)

    return small_dataset_dict
