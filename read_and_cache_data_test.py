import read_and_cache_data as rcd
from IPython.display import display
import ashrae_constants as const

#raw data
data_raw = rcd.get_data('ashrae-energy-prediction', cache_file='store_raw.h5')
for name in const.NAMES:
    print(f'NaNs for {name}')
    display(data_raw[name].isna().sum())


