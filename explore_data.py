import pandas as pd


def get_buildings_with_high_meter(df, thresholds):
    """
   Return a dict of {threshold: set of buildings with a meter reading above that threshold}
   """
    high_meter_buildings = {threshold: set(df.building_id[df.meter_reading > threshold]) for threshold in thresholds}
    for threshold, buildings in high_meter_buildings .items():
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

