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

scaler = preprocessing.StandardScaler().fit(data['X_train'])
Xtrain = add_col_ones(scaler.transform(data['X_train']))
Xtest = add_col_ones(scaler.transform(data['X_test']))
ytrain, ytest = data['Y_train'], data['Y_test']
mu_y, v_y = jnp.mean(ytrain), jnp.var(ytrain)
ytrain, ytest = ytrain - mu_y, ytest - mu_y

params, residuals, rank, s = np.linalg.lstsq(Xtrain, ytrain, rcond=None)
prediction = Xtest @ params 
mse = calc_mse(prediction, ytest)

model = sklearn.linear_model.LinearRegression()
model.fit(Xtrain, ytrain)
prediction = model.predict(Xtest)
mse_sklearn = calc_mse(prediction, ytest)

print(f'Test set. MSE(jax)={mse:.2f}, MSE(sklearn)={mse_sklearn:.2f}, SMSE={mse/v_y:.2f}')

gauss_log_likelihood = lambda mean, cov, y: \
    jax.scipy.stats.norm.logpdf(y, mean, jnp.sqrt(jnp.diag(cov))).sum()

def nll_gauss(params, x, y):
    mu_y, v_t, w = params
    m = mu_y * jnp.eye(1)
    c = v_y * jnp.eye(1)
    return -gauss_log_likelihood(m, c, y)

def nll_linreg(params, x, y):
    mu_y, v_t, w = params
    m = jnp.dot(w, x) * jnp.eye(1)
    c = v_y * jnp.eye(1)
    return -gauss_log_likelihood(m, c, y)


def compute_regression_baselines(Xtrain, ytrain, Xtest, ytest):
    mu_y, v_y = jnp.mean(ytrain), jnp.var(ytrain)
    #  model = sklearn.linear_model.LinearRegression() 
    w, residuals, rank, s = np.linalg.lstsq(Xtrain, ytrain, rcond=None) # model.fit(Xtrain, ytrain)
    #prediction = Xtest @ w # prediction = model.predict(Xtest)
    params = (mu_y, v_y, w)

    nll_te_gauss = jnp.mean(jax.vmap(nll_baseline, (None, 0, 0))(params, Xtest, ytest))
    nll_te_linreg = jnp.mean(jax.vmap(nll_linreg, (None, 0, 0))(params, Xtest, ytest))
    return nll_te_gauss, nll_te_linreg



nll_te_gauss, nll_te_linreg = compute_regression_baselines(Xtrain, ytrain, Xtest, ytest)
msll = nll_te_linreg - nll_te_gauss
print(f'Test set. NLPD={nll_te_linreg:.2f}, MSLL={msll:.2f})
