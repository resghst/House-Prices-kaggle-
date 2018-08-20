#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#bring in the six packs
df_train = pd.read_csv('./input/train.csv')
df_test = pd.read_csv('./input/test.csv')
print('---------------------------------------------------')
#check the decoration
print(df_train.columns)
print('---------------------------------------------------')
#descriptive statistics summary
print(df_train['SalePrice'].describe())
print('---------------------------------------------------')
#histogram
print('draw Image...')
filename = 'distplot_SalePrice.png'
sns.distplot(df_train['SalePrice'])
plt.gcf().savefig(filename, dpi=1000)
plt.close()
print('save Image finish filename = ' + filename )
print('---------------------------------------------------')
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
print('---------------------------------------------------')
#scatter plot grlivarea/saleprice
print('draw Image...')
var = 'GrLivArea'
filename = 'GrLivArea_SalePrice.png'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
plt.gcf().savefig(filename, dpi=1000)
plt.close()
print('save Image finish filename = ' + filename )
print('---------------------------------------------------')
#scatter plot totalbsmtsf/saleprice
print('draw Image...')
var = 'TotalBsmtSF'
filename = 'TotalBsmtSF_SalePrice.png'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
plt.gcf().savefig(filename, dpi=1000)
plt.close()
print('save Image finish filename = ' + filename)
print('---------------------------------------------------')
#box plot overallqual/saleprice
print('draw Image...')
var = 'OverallQual'
filename = 'OverallQual_SalePrice.png'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.gcf().savefig(filename, dpi=1000)
plt.close()
print('save Image finish filename = ' + filename)
print('---------------------------------------------------')
#box plot YearBuilt/saleprice
print('draw Image...')
var = 'YearBuilt'
filename = 'YearBuilt_SalePrice.png'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
plt.gcf().savefig(filename, dpi=500)
plt.close()
print('save Image finish filename = ' + filename)
print('---------------------------------------------------')
#correlation matrix
print('draw Image...')
corrmat = df_train.corr()
filename = 'correlation_matrix.png'
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.gcf().savefig(filename, dpi=500)
plt.close()
print('save Image finish filename = ' + filename)
print('---------------------------------------------------')
#saleprice correlation matrix
print('draw Image...')
k = 10 #number of variables for heatmap
filename = 'correlation_matrix_number.png'
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()
plt.gcf().savefig(filename, dpi=500)
plt.close()
print('save Image finish filename = ' + filename)
print('---------------------------------------------------')
#scatterplot
filename = 'SalePrice_correlated_variables.png'
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
# plt.show()
# plt.gcf().savefig(filename, dpi=600)
plt.close()
print('save Image finish filename = ' + filename)
print('---------------------------------------------------')
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
print('---------------------------------------------------')
#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
print(df_train.isnull().sum().max()) #just checking that there's no missing data missing...
print('---------------------------------------------------')
#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
print('---------------------------------------------------')
#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
print('---------------------------------------------------')
#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
print('---------------------------------------------------')
#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
print('---------------------------------------------------')
#histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
print('---------------------------------------------------')
#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])
#transformed histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
print('---------------------------------------------------')
#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#transformed histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
print('---------------------------------------------------')
#convert categorical variable into dummy
df_train = pd.get_dummies(df_train)
print(df_train)
