import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

#處理 outliers
train = train[train['GarageArea'] < 1200]
train = train[train['TotalBsmtSF'] < 4000]
train = train[train['GrLivArea'] < 4000]

#處理 missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna() 

train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(test.Street, drop_first=True)

train['GrLivArea'] = np.log(train['GrLivArea']) 
test['GrLivArea'] = np.log(test['GrLivArea']) 

data = train.select_dtypes(include=[np.number]).interpolate().dropna()


y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=55, test_size=.2)

from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error
print ('RMSE is: \t', mean_squared_error(y_test, predictions))

feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = model.predict(feats)
final_predictions = np.exp(predictions)

submission = pd.DataFrame()
submission['Id'] = test.Id
submission['SalePrice'] = final_predictions
submission.to_csv('output.csv', index=False)

