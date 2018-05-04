from LinearRegressionGD import *
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor


df = pd.read_csv('./housing.data.txt',header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
			'NOX', 'RM', 'AGE', 'DIS', 'RAD',
			'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(df.head())

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
				cbar=True,
				annot=True,
				square=True,
				fmt='.2f',
				annot_kws={'size': 15},
				yticklabels=cols,
				xticklabels=cols)
plt.show()

X = df[['RM']].values
y = df['MEDV'].values
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

sns.reset_orig()
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

def lin_regplot(X, y, model):
	plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
	plt.plot(X, model.predict(X), color='black', lw=2)
	return None

lin_regplot(X_std, y_std, lr)
plt.title('Manual LinReg Implementation')
plt.xlabel('Average # of Rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()

num_rooms_std = sc_x.transform([[5.0]])
price_std = lr.predict(num_rooms_std)
print("Price in $1000s: %.3f" % sc_y.inverse_transform(price_std))


#SKLearn Linear Regression Implementation
slr = LinearRegression()
slr.fit(X, y)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

lin_regplot(X, y, slr)
plt.title('SKLearn Implementation')
plt.xlabel('Average # of Rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()


#Using RANSAC to deal with outliers
ransac = RANSACRegressor(LinearRegression(),
						max_trials=100,
						min_samples=50,
						loss='absolute_loss',
						residual_threshold=5.0,
						random_state=0)

ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
			c='steelblue', edgecolor='white',
			marker='o', label='Outliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
			c='limegreen', edgecolor='white',
			marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Average # of Rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')
plt.show()

print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)



#Evaluating Performance of Linear Regression Models
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=0.3, random_state=0)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

plt.scatter(y_train_pred, y_train_pred - y_train,
			c='steelblue', marker='o', edgecolor='white',
			label='Training Data')

plt.scatter(y_test_pred, y_test_pred - y_test,
			c='limegreen', marker='s', edgecolor='white',
			label='Test Data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

#metrics
print('MSE train:{0:.3f}, test: {1:.3f}'.format(
		mean_squared_error(y_train, y_train_pred),
		mean_squared_error(y_test, y_test_pred)))

print('R^2 train: {0:.3f}, test: {1:.3f}'.format(
		r2_score(y_train, y_train_pred),
		r2_score(y_test, y_test_pred)))


#Using Polynomial and Cubic for fit
X = df[['LSTAT']].values
y = df['MEDV'].values

regr = LinearRegression()

quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

plt.scatter(X, y, label='training points', color='lightgray')

plt.plot(X_fit, y_lin_fit,
		label='linear d=1, $R^2=% .2f$' % linear_r2,
		color='blue',
		lw=2,
		linestyle=':')

plt.plot(X_fit, y_quad_fit,
		label='quadratic d=2, $R^2=% .2f$' % quadratic_r2,
		color='red',
		lw=2,
		linestyle='-')

plt.plot(X_fit, y_cubic_fit,
		label='cubic d=3, $R^2=% .2f$' % cubic_r2,
		color='green',
		lw=2,
		linestyle='--')

plt.xlabel('% lower status of population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper right')
plt.show()


#Perform log transformation
X_log = np.log(X)
y_sqrt = np.sqrt(y)

X_fit = np.arange(X_log.min()-1,
				X_log.max()+1, 1)[:,np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

plt.plot(X_log, y_sqrt,
		label='training points',
		color='lightgray')

plt.plot(X_fit, y_lin_fit,
		label='linear d=1, $R^2=% .2f$' % linear_r2,
		color='blue',
		lw=2)

plt.xlabel('% lower status of population [LSTAT]')
plt.ylabel('$\sqrt{Price \; in \; \$1000s \; [MEDV]}$')
plt.legend(loc='upper right')
plt.show()


#Decision tree regression
X = df[['LSTAT']].values
y = df['MEDV'].values
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X,y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% Lower Status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()


