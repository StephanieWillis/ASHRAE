# from sklearn.linear_model import LinearRegression
# from sklearn.impute import SimpleImputer


import ashrae_constants as const
import prepare_data as prep
import explore_correlations as exp

# train_all_dict = prep.get_joined_data(dataset_names=['train'])

small_train_all_dict = prep.produce_and_cache_small_dataset_dict('train')
small_test_all_dict = prep.produce_and_cache_small_dataset_dict('test')

elec_train = small_train_all_dict['electricity']
elec_train_correlations = exp.produce_correlation_df(elec_train, buildings=None)
exp.plot_correlations(elec_train_correlations)

