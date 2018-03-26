import pandas as pd 
from io import StringIO
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np


csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)

print(df.isnull().sum())

print(df.values)

print('Drop rows with NaN values')
print(df.dropna(axis=0))

print('Drop columns with NaN values')
print(df.dropna(axis=1))

print('Drop columns where all values as NaN')
print(df.dropna(how='all'))

print('Drop rows with less than 4 real values')
print(df.dropna(thresh=4))

print('Drop rows with NaN values in C column')
print(df.dropna(subset=['C']))
print()

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print('Imputed Missing values using mean of columns')
print(imputed_data)
print()


df = pd.DataFrame([
			['green', 'M', '10.1', 'class1'],
			['red', 'L', '13.5', 'class2'],
			['blue', 'XL', '15.3', 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']
print(df)
print()

size_mapping = {
				'XL': 3,
				'L': 2,
				'M': 1}

df['size'] = df['size'].map(size_mapping)
print(df)
print()

inv_size_mapping = {v: k for k, v in size_mapping.items()}
print(df['size'].map(inv_size_mapping))
print()

class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)
print()

df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)
print()

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)
print()

print('One-hot encoding of nominal features')
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:, 0])
print(X)

ohe = OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X).toarray())

print(pd.get_dummies(df[['price', 'color', 'size']]))

print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))

