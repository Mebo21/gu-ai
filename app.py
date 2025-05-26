import pandas as pd
import numpy as np
from datetime import datetime
from joblib import load
cluster_num_list=[0,1,2]

# 사용자 정의 함수들
def is_working(hour):
    return 1 if 9 <= hour <= 18 else 0

test = pd.read_csv('csv/test.csv')
building_info = pd.read_csv('csv/building_info.csv')
cluster_df = pd.read_csv('csv/cluster_info.csv')

test.columns = ['num_date_time', 'num', 'date_time', 'temperature', 'precipitation', 'windspeed', 'humidity']
building_info.columns = ['num', 'type', 'area', 'cooling_area', 'solar', 'ESS', 'PCS']

building_info.replace('-', 0, inplace=True)
building_info[building_info.columns.drop('type')] = building_info.drop(['type'], axis=1).apply(lambda x:x.astype('float'))

test = pd.merge(test, building_info, on='num', how='inner')

test['date_time'] = pd.to_datetime(test['date_time'])

test['precipitation'].fillna(0, inplace=True)
test['windspeed'].interpolate(method='linear', inplace=True)
test['humidity'].interpolate(method='linear', inplace=True)

test = pd.merge(test, cluster_df, on='num', how='left')

test['month'] = test['date_time'].dt.month
test['dow'] = test['date_time'].dt.day_of_week
test['day'] = test['date_time'].dt.day
test['is_working'] = test['date_time'].dt.hour.apply(is_working)

test = pd.get_dummies(data=test, columns=['type'])

test.to_csv('test_final.csv', index=False)

test_idx = []
preds = []
for c_num in cluster_num_list:
    X = test[test.cluster == c_num].drop(columns=['num_date_time', 'date_time', 'cluster'])
    model = load(f'best_model_cluster{c_num}.pkl')
    pred = model.predict(X)
    X['target'] = pred
    test_idx.append(X.index)
    preds.append(pred)

submission = pd.read_csv('csv/sample_submission.csv')
submission.loc[test_idx[0], 'answer'] = preds[0]
submission.loc[test_idx[1], 'answer'] = preds[1]
submission.loc[test_idx[2], 'answer'] = preds[2]
submission.to_csv("static/result.csv", index=False)
