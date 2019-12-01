import pathlib
import pandas as pd
import ashrae_constants as const
import itertools


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


def get_data(ashrae_dir, cache_file=None, filenames=const.NAMES):
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


def clean_data(raw_data, names=const.NAMES, meter_map=const.METER_MAP):
    """
    Convert timestamps to timestamp objects, fill in blanks in weather data, add names of meter types
    """

    cleaned_data = raw_data.copy()
    if 'building_metadata' in names:
        names.remove('building_metadata')

    for name in names:
        print(f'Cleaning {name} dataset')
        df = cleaned_data[name]
        df.timestamp = pd.to_datetime(df.timestamp)
        if name == 'weather_*':
            df = add_missing_weather_data(df)

        elif name in ['train', 'test']:
            df['meter_type'] = df['meter'].map(meter_map)
            pass

    return cleaned_data


def add_missing_weather_data(df):
    """ Add missing timestamps to weather data and interpolate to fill in the data
    return df with missing times and weather data filled in
    """

    full_date_range = pd.date_range(start=min(df.timestamp), end=max(df.timestamp), freq='H')
    sites = list(set(df.site_id))
    full_data_site_range = pd.DataFrame(itertools.product(sites, full_date_range),
                                        columns=['site_id', 'timestamp'])
    df_all_dates = full_data_site_range.merge(df, on = ['site_id', 'timestamp'], how='left')
    df_all_dates = df_all_dates.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))

    return df_all_dates


