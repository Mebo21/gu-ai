import os
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file
from joblib import load

model = Blueprint('model_predict', __name__)

cluster_num_list = [0, 1, 2]

def is_working(hour):
    return 1 if 9 <= hour <= 18 else 0

@model.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400

    try:
        test = pd.read_csv(file)
        building_info = pd.read_csv('csv/building_info.csv')
        cluster_df = pd.read_csv('csv/cluster_info.csv')
        submission = pd.read_csv('csv/sample_submission.csv')

        test.columns = ['num_date_time', 'num', 'date_time', 'temperature', 'precipitation', 'windspeed', 'humidity']
        building_info.columns = ['num', 'type', 'area', 'cooling_area', 'solar', 'ESS', 'PCS']
        building_info.replace('-', 0, inplace=True)
        building_info[building_info.columns.drop('type')] = building_info.drop(['type'], axis=1).astype(float)

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

        test_idx = []
        preds = []
        for c_num in cluster_num_list:
            X = test[test.cluster == c_num].drop(columns=['num_date_time', 'date_time', 'cluster'])
            model_path = f'best_model_cluster{c_num}.pkl'
            model = load(model_path)
            pred = model.predict(X)
            test_idx.append(test[test.cluster == c_num].index)
            preds.append(pred)

        for i in range(len(cluster_num_list)):
            submission.loc[test_idx[i], 'answer'] = preds[i]

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../app/model_predict
        STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'static'))  # .../static
        os.makedirs(STATIC_DIR, exist_ok=True)

        result_path = os.path.join(STATIC_DIR, 'result.csv')
        submission.to_csv(result_path, index=False)

        return send_file(
            result_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name='result.csv'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500
