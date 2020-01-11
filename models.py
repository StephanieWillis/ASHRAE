import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer


def fit_per_building_linear_regression(df, feature_name='air_temperature'):
    """ df is a training df which includes columns 'meter_reading and feature_name and which has 'building_id'
     in the index'"""
    
    building_ids = list(set(df.index.get_level_values('building_id')))
    df_coefficients = pd.DataFrame(index=building_ids, columns=['r_sq', 'intercept', 'temp_coef', 'set_point_temp'])
    df_coefficients.index.rename('building_id', inplace=True)

    for building_id in building_ids:
        building_df = df.loc[building_id]
        y = building_df['meter_reading'].to_numpy()
        x = building_df[feature_name].to_numpy().reshape(-1, 1)
        
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        x = imp_mean.fit_transform(x)
        
        model = LinearRegression(n_jobs=-1).fit(x, y)
        r_sq = model.score(x, y)
        df_coefficients.loc[building_id, 'r_sq'] = r_sq
        df_coefficients.loc[building_id, 'intercept'] = model.intercept_
        df_coefficients.loc[building_id, 'temp_coef'] = model.coef_[0]
        df_coefficients.loc[building_id, 'set_point_temp'] = -model.intercept_ / model.coef_[0]

    for col in list(df_coefficients):
        df_coefficients[col] = pd.to_numeric(df_coefficients[col])

    return df_coefficients

