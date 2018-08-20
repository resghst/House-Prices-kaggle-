
import pandas as pd 

train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

# 挑选特征值
selected_features = ['Foundation', 'Heating', 'Electrical', 'SaleType', 'SaleCondition', 'GarageArea','YearRemodAdd','YearBuilt','1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'BsmtUnfSF', 'CentralAir']

X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['SalePrice']

# 补充特征缺失值
X_train['Electrical'].fillna('SBrkr', inplace=True)
X_train['SaleType'].fillna('WD', inplace=True)
X_train['GarageArea'].fillna(X_train['GarageArea'].mean(), inplace=True)
X_train['TotalBsmtSF'].fillna(X_train['TotalBsmtSF'].mean(), inplace=True)
X_train['BsmtUnfSF'].fillna(X_train['BsmtUnfSF'].mean(), inplace=True)
X_test['Electrical'].fillna('SBrkr', inplace=True)
X_test['SaleType'].fillna('WD', inplace=True)
X_test['GarageArea'].fillna(X_test['GarageArea'].mean(), inplace=True)
X_test['TotalBsmtSF'].fillna(X_test['TotalBsmtSF'].mean(), inplace=True)
X_test['BsmtUnfSF'].fillna(X_test['BsmtUnfSF'].mean(), inplace=True)

print (X_train.info())
print (X_test.info())

# 采用DictVectorizer进行特征向量化
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)

X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

# 使用随机森林回归模型进行 回归预测
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
#rfr = RandomForestRegressor()
rfr = GradientBoostingRegressor()
rfr.fit(X_train, y_train)
rfr_y_predict = rfr.predict(X_test)

# 输出结果
rfr_submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': rfr_y_predict})
rfr_submission.to_csv('rfr_submission.csv', index=False)
