import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

#處理 outliers
train = train[train['GarageArea'] < 1200]

#處理 missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna() 

train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

data = train.select_dtypes(include=[np.number]).interpolate().dropna() 

y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

from sklearn import svm
lr = svm.SVR()
model = lr.fit(X_train, y_train)
predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error
print ('RMSE is: \t', mean_squared_error(y_test, predictions))

feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
# print(train.columns)
# print(X.columns)
# print(X_test.columns)
# print(feats.columns)
predictions = model.predict(feats)
final_predictions = np.exp(predictions)

submission = pd.DataFrame()
submission['Id'] = test.Id
submission['SalePrice'] = final_predictions
submission.to_csv('output.csv', index=False)

