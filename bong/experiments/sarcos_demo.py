
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import jax.numpy as jnp
import jax

from datasets import get_sarcos_data
from models import fit_linreg_baseline, fit_gauss_baseline, nll_gauss, nll_linreg

# We follow sec 2.5 of https://gaussianprocess.org/gpml/chapters/RW2.pdf
# The inputs were linearly rescaled to have zero mean and unit variance on the training set.
# The outputs were centered so as to have zero mean on the training set.
# SMSE is the MSE / var(ytrain)


def calc_mse(prediction, Y):
    mse = jnp.mean(jnp.square(prediction - Y))
    return mse

def add_col_ones(X):
    ones_column = jnp.ones((X.shape[0], 1))
    return jnp.hstack((ones_column, X))

data = get_sarcos_data(raw=True)

scaler = preprocessing.StandardScaler().fit(data['X_train_raw'])
Xtrain = add_col_ones(scaler.transform(data['X_train_raw']))
Xtest = add_col_ones(scaler.transform(data['X_test_raw']))
ytrain, ytest = data['Y_train_raw'], data['Y_test_raw']
mu_y, v_y = jnp.mean(ytrain), jnp.var(ytrain)
ytrain, ytest = ytrain - mu_y, ytest - mu_y

print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)

params, residuals, rank, s = jnp.linalg.lstsq(Xtrain, ytrain, rcond=None)
prediction = Xtest @ params 
mse = calc_mse(prediction, ytest)
print(f'Raw Jax. MSE(jax)={mse:.3f}, SMSE={mse/v_y:.3f}. Book SME=0.075')

model = LinearRegression()
model.fit(Xtrain, ytrain)
prediction = model.predict(Xtest)
msen = calc_mse(prediction, ytest)
print(f'Sklearn. MSE(sklearn)={mse:.3f}, SMSE={mse/v_y:.3f}. Book SME=0.075')

#Compare to library code 
for ntrain in [0]:
     # We add column of 1s by default to match linreg results in sarcos_demo.py
    # But if passing to a neural net with a bias term, this is unnecessary
    data = get_sarcos_data(ntrain, add_ones=True) # ntrain=0 means use all the data

    mu_y, v_y = fit_gauss_baseline(data['X_tr'], data['Y_tr'])
    w, sigma2 = fit_linreg_baseline(data['X_tr'], data['Y_tr'], method='lstsq')
    w_sgd, sigma2_sgd = fit_linreg_baseline(data['X_tr'], data['Y_tr'], method='sgd')

    print(f'Estimated noise variance. Gauss {v_y}, Linreg {sigma2}, Linreg-sd {sigma2_sgd}')

    prediction = data['X_te'] @ w 
    mse = calc_mse(prediction, data['Y_te'])
    print(f'n={ntrain}, lstsq. MSE={mse:.3f}, SMSE={mse/v_y:.3f}. Book SME=0.075')

    prediction = data['X_te'] @ w_sgd 
    mse = calc_mse(prediction, data['Y_te'])
    print(f'n={ntrain}, SGD. MSE={mse:.3f}, SMSE={mse/v_y:.3f}. Book SME=0.075')

    nll_te_gauss = jnp.mean(jax.vmap(nll_gauss, (None, None, 0, 0))(mu_y, v_y, data['X_te'], data['Y_te']))
    nll_te_linreg = jnp.mean(jax.vmap(nll_linreg, (None, None, 0, 0))(w, sigma2, data['X_te'], data['Y_te']))
    msll = nll_te_linreg - nll_te_gauss
    print(f'n={ntrain}, NLPD={nll_te_linreg:.3f}, MSLL={msll:.3f}. Book MSLL=-1.29')