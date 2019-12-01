import pandas as pd


def get_buildings_with_high_meter(df, threshold):
    """
   Return the set of building ids with a count of meter readings above a certain threshold.
   """
    return set(df.building_id[df.meter_reading > threshold])


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

