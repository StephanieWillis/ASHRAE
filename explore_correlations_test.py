import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas_profiling
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


import ashrae_constants as const
import prepare_data as prep
import explore_correlations as exp

train_all_dict, test_all_dict = prep.get_joined_data()

elec_train = train_all_dict['electricity']
building_subset = list(range(30))
elec_train_subset = elec_train[elec_train.index.get_level_values('building_id').isin(building_subset)]
elec_buildings_subset_correlations = exp.produce_correlation_df(elec_train_subset, buildings=building_subset)
exp.explore_correlations(elec_buildings_subset_correlations)

