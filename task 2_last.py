import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

train_path = './fraudTrain.csv'
test_path = './fraudTest.csv'

# Loading Data
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Preprocessing Data
target_column = 'is_fraud'
train_data['age'] = pd.to_datetime('today').year - pd.to_datetime(train_data['dob']).dt.year
train_data['hour'] = pd.to_datetime(train_data['trans_date_trans_time']).dt.hour
train_data['day'] = pd.to_datetime(train_data['trans_date_trans_time']).dt.dayofweek
train_data['month'] = pd.to_datetime(train_data['trans_date_trans_time']).dt.month

test_data['age'] = pd.to_datetime('today').year - pd.to_datetime(test_data['dob']).dt.year
test_data['hour'] = pd.to_datetime(test_data['trans_date_trans_time']).dt.hour
test_data['day'] = pd.to_datetime(test_data['trans_date_trans_time']).dt.dayofweek
test_data['month'] = pd.to_datetime(test_data['trans_date_trans_time']).dt.month

columns_list = ['category', 'amt', 'zip', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'age', 'hour', 'day',
                'month', target_column]

train_data = train_data[columns_list]
test_data = test_data[columns_list]

train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

y_train = train_data[target_column].values
X_train = train_data.drop(target_column, axis='columns').values

y_test = test_data[target_column].values
X_test = test_data.drop(target_column, axis='columns').values

# SMOTE Oversampling
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Building and evaluating the model
LR_model = LogisticRegression()
LR_model.fit(X_train_resampled, y_train_resampled)

LR_prediction = LR_model.predict(X_test)

LR_classification_report = classification_report(y_test, LR_prediction)
LR_confusion_matrix = confusion_matrix(y_test, LR_prediction)
LR_accuracy = accuracy_score(y_test, LR_prediction)

print(f'Logistic Regression Classification Report:\n{LR_classification_report}')
print(f'Logistic Regression Confusion matrix:\n{LR_confusion_matrix}')
print(f'Logistic Regression Accuracy:\n{LR_accuracy}')
