# Filenames (without the extension) we are expected to find in `ashrae_dir`.
NAMES = [
    'building_metadata',
    'weather_train',
    'weather_test',
    'train',
    'test',
]

METER_MAP = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}

CORRELATORS = ['air_temperature', 'cloud_coverage', 'dew_temperature','precip_depth_1_hr', 'sea_level_pressure',
               'wind_direction','wind_speed', 'hour', 'weekday', 'month', 'is_weekend']
