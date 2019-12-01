import pathlib
import pandas as pd
import ashrae_constants as const


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

