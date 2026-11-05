import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

import pickle

df = pd.read_csv('laptops.csv')
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.lower()
df = df.rename(columns={'_storage': 'storage'})
df['operating_system_version'] = df['operating_system_version'].fillna("Unknown")

df['log_price'] = np.log1p(df['price'])
del df['price']

if 'ram' in df.columns and df['ram'].dtype == 'object':
    df['ram'] = df['ram'].str.extract(r'([\d\.]+)').astype(float)

if 'weight' in df.columns and df['weight'].dtype == 'object':
    df['weight'] = df['weight'].str.extract(r'([\d\.]+)').astype(float)

if 'screen_size' in df.columns and df['screen_size'].dtype == 'object':
    df['screen_size'] = df['screen_size'].str.extract(r'([\d\.]+)').astype(float)

del df['model_name']


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

y_test = df_test.log_price.values
del df_test['log_price']
y_full_train = df_full_train.log_price.values
del df_full_train['log_price']

dv = DictVectorizer(sparse=False)


full_train_dict = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)

dfull_train = xgb.DMatrix(X_full_train, label=y_full_train)
dtest = xgb.DMatrix(X_test, label=y_test)

final_params = {
    'eta': 0.3,
    'max_depth': 3,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'nthread': 8,
    'seed': 1,
}

final_model = xgb.train(final_params, dfull_train, num_boost_round=486)

y_pred_test = final_model.predict(dtest)
final_test_rmse = root_mean_squared_error(y_test, y_pred_test)

print(f'Final Test Log RMSE: {final_test_rmse:.5f}')


output_file = f'model.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, final_model), f_out)
    