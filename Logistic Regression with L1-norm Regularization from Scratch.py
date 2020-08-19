import numpy as np # Linear algebra.
import pandas as pd # Data processing.
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#%%
data_df = pd.read_csv('/Users/carrotkr/Dropbox/UCI_Credit_Card.csv')
print(data_df.head())
print(data_df.info())
print(data_df.describe())
print(data_df.isnull().sum())
print(data_df.columns)

data = data_df.copy()
data = data.drop(['ID'], axis=1)
print(data.info())

#%% Targer variable.
print(data_df['default.payment.next.month'].unique())
print(data_df['default.payment.next.month'].value_counts())

Y = data['default.payment.next.month'].copy()
del data['default.payment.next.month']
Y = np.array(Y)

num_features = len(data.columns)

#%% Standardization.
data_std = data.astype('int')
print(data_std.info())

from sklearn.preprocessing import StandardScaler

data_std = StandardScaler().fit_transform(data_std)

#%%
X = data_std.copy()
X = np.insert(X, 0, 1, axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=33)

#%%
# Reference.
#   Implementing a Lasso solver from scratch in Python by Daniel O'Connor.
def clip(beta, alpha):
    clipped = np.minimum(beta, alpha)
    clipped = np.maximum(clipped, -alpha)
    
    return clipped

def prox_l1_norm(beta_hat, alpha, penalizeAll=True):
    output = beta_hat - clip(beta_hat, alpha)
    
    if not penalizeAll:
        output[0] = beta_hat[0]
        
    return output

def lasso_prox_grad(X, Y, Lambda):
    max_iter = 300
    alpha = 0.005
    
    beta = np.zeros(num_features+1)
    cost_function_vals = np.zeros(max_iter)
    
    for i in range(max_iter):
        gradient = X.T @ (X @ beta - Y)
        beta = prox_l1_norm(beta - alpha*gradient, alpha*Lambda)
        
        cost_function_vals[i] = 0.5*np.linalg.norm(X @ beta - Y)**2 + Lambda*np.sum(np.abs(beta))
        
        print('Iteration: ', i, 'Objective function value: ', cost_function_vals[i])
        
    return beta, cost_function_vals

#%%
num_non_zero_beta = 3 

prm = np.random.permutation(num_features+1)
beta_true = np.zeros(num_features+1)
beta_true[prm[0:num_non_zero_beta]] = 5*np.random.randn(num_non_zero_beta)

#%% Logistic regression.
def sigmoid_function(x):
    return 1 / (1+np.exp(-x))

#%%
theta = np.zeros(X_train.shape[1])
num_epochs = 10

Lambda = 10

for epoch in range(num_epochs):
    hypothesis = sigmoid_function(X_train @ theta)
    
    beta, cost_function_vals = lasso_prox_grad(X_train, Y_train, Lambda)
    
plt.figure()
plt.stem(beta, markerfmt='C1o')
plt.stem(beta_true)

#%%
from sklearn.linear_model import Lasso

lambda_values = (0.1, 0.05, 0.01, 0.001, 0.0001)

for index, Lambda in enumerate(lambda_values):
    lasso = Lasso(alpha=Lambda).fit(X_train, Y_train)
    
    print('\nLambda: ', Lambda)
    print('Training set accuracy: ', lasso.score(X_train, Y_train))
    print('Test set accuracy: ', lasso.score(X_test, Y_test))
    print('Lasso - Number of features: ', np.sum(lasso.coef_ != 0))
    print(lasso.coef_)
    for index_feature in range(num_features):
        if (lasso.coef_[index_feature+1] != 0):
            print('\'Index\'', index_feature, 'Feature: ', data.columns[index_feature])