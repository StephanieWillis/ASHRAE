import read_and_cache_data as rcd
from IPython.display import display
import ashrae_constants as const

#raw data
raw_data= rcd.get_data('ashrae-energy-prediction', cache_file='store_raw.h5')
for name in const.NAMES:
    print(f'NaNs for {name}')
    display(raw_data[name].isna().sum())

# clean data
cleaned_data = rcd.clean_data(raw_data)


