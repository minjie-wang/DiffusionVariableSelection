import numpy as np

def EBIC_LinearRegression(X,y,beta,beta0,gamma=0.5):
    
    n = X.shape[0]; p = X.shape[1]; s = sum(( beta !=0 ))
    beta = np.reshape(beta,(p,1))
    
    y = np.reshape(y,(n,))

    linear_pred = np.reshape(np.asarray(np.matmul(X, beta) + beta0),(n,))
    
    EBIC = n * np.log(np.sum(np.square(y - linear_pred))) + s * np.log(n) + 2 * gamma * s * np.log(p)
    
    if s >= n:
        EBIC = float("inf") 
        
    return(EBIC)



from sklearn.linear_model import LinearRegression
def refit_LinearRegression(X,y,beta_hat,fit_intercept=True):
    beta_refit = np.zeros(len(beta_hat))
    
    act_set = np.where( beta_hat !=0 )
    act_set = np.asanyarray(act_set).flatten()
    
    if len(act_set) > 0:
        X_subset = X[:,act_set]
        reg = LinearRegression(fit_intercept=fit_intercept).fit(X_subset, y)
        beta_refit[act_set] = reg.coef_
        beta0_refit = reg.intercept_
    elif fit_intercept:
        beta0_refit = np.mean(y)
    else:
        beta0_refit = 0    
    return(beta_refit,beta0_refit)



from sklearn import linear_model
def choose_para_EBIC(X,y,alphas=None,fit_intercept=True,gamma=0.5):
    
    n = X.shape[0]
    if alphas is None:        
        lambda_max = np.max(abs( np.matmul(X.T,y) / n ))
        if fit_intercept:
            lambda_max = np.max(abs( np.matmul(X.T,y-np.mean(y)) / n ))
    
        alphas = np.linspace(0.01, lambda_max + 0.001, num=100)
    
    EBIC_criterion = []
    for alpha in alphas:
        clf = linear_model.Lasso(alpha=alpha,fit_intercept=fit_intercept)
        clf.fit(X,y)

        beta_refit,beta0_refit = refit_LinearRegression(X,y,clf.coef_,fit_intercept=fit_intercept)
        EBIC = EBIC_LinearRegression(X,y,beta_refit,beta0_refit,gamma=gamma)
        EBIC_criterion.append(EBIC)
    
    
    alpha_optimal = np.min(alphas[np.where(EBIC_criterion == min(EBIC_criterion))])
    return(alpha_optimal)





###########################################################################################################################
### Logistic regression

from sklearn.linear_model import LogisticRegression
def refit_LogisticRegression(X,y,beta_hat,fit_intercept=True):
    beta_refit = np.zeros(len(beta_hat))
    
    act_set = np.where( beta_hat !=0 )
    act_set = np.asanyarray(act_set).flatten()
    
    if len(act_set) > 0:
        X_subset = X[:,act_set]
        reg = LogisticRegression(penalty="none",fit_intercept=fit_intercept).fit(X_subset, y)
        beta_refit[act_set] = reg.coef_
        beta0_refit = reg.intercept_
    elif fit_intercept:
        beta0_refit = np.log(np.mean(y) / (1 - np.mean(y)))
    else:
        beta0_refit = 0
    return(beta_refit,beta0_refit)



def EBIC_LogisticRegression(X,y,beta,beta0,gamma=0.5):
    
    n = X.shape[0]; p = X.shape[1]; s = sum(( beta !=0 ))
    beta = np.reshape(beta,(p,1))
    
    y = np.reshape(y,(n,))
    
    linear_pred = np.reshape(np.asarray(np.matmul(X, beta) + beta0),(n,))
    
    EBIC = 2 * np.sum( np.log(1 + np.exp(linear_pred)) - np.multiply(y , linear_pred) ) +  s * np.log(n) + 2 * gamma * s * np.log(p)
    
    if s >= n:
        EBIC = float("inf") 

    return(EBIC)



def choose_para_EBIC_LogisticRegression(X,y,alphas=None,fit_intercept=True,gamma=0.5):
    
    if alphas is None:        
        lambda_max = np.max(abs( np.matmul(X.T, y-np.mean(y) ) ))
        C_min = 1 / (lambda_max + 0.001)
        alphas = np.linspace(C_min, 1, num=100)
    
    EBIC_criterion = []
    for alpha in alphas:
        clf = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=alpha,
            fit_intercept=fit_intercept)
        clf.fit(X,y)

        beta_refit,beta0_refit = refit_LogisticRegression(X,y,np.concatenate(clf.coef_,axis=0),fit_intercept=fit_intercept)
        EBIC = EBIC_LogisticRegression(X,y,beta_refit,beta0_refit,gamma=gamma)
        EBIC_criterion.append(EBIC)
    
    
    alpha_optimal = np.max(alphas[np.where(EBIC_criterion == min(EBIC_criterion))])
    return(alpha_optimal)



###########################################################################################################################
### Neighborhood selection

def neighb_select(X,alpha=None,rule='AND_rule',fit_intercept=False,gamma=0.5):
    
    [n,p] = np.shape(X)

    Sigma = np.zeros([p,p])
    
    for j in range(p):
        
        Xj = X[:,np.arange(p)!=j]
        yj = X[:,j]
        
        if alpha == None:
            alpha_optimal = choose_para_EBIC(Xj,yj,fit_intercept=fit_intercept,gamma=gamma)
        else:
            alpha_optimal = alpha

        clf = linear_model.Lasso(alpha=alpha_optimal,fit_intercept=fit_intercept)
        clf.fit(Xj,yj)
        beta_hat_optimal = clf.coef_
        beta0_hat_optimal = clf.intercept_
        
        Sigma[np.arange(p)!=j,j] = beta_hat_optimal
        
    
    if rule == "AND_rule":
        Z_U = np.triu(Sigma);
        Z_L = np.tril(Sigma);
        Z_L_T = Z_L.T
        indice = (Z_U == 0) & (Z_L_T == 0)
        Z_U[indice] = 0
        Z_L_T[indice] = 0
        Z_sym = Z_U + Z_L_T.T
        Sigma = Z_sym
        Sigma = (Sigma + Sigma.T)/2
        Sigma[np.diag_indices(p)] = 0
    elif rule == "OR_rule":
        Z_U = np.triu(Sigma);
        Z_L = np.tril(Sigma);
        Z_L_T = Z_L.T
        indice = (Z_U == 0) | (Z_L_T == 0)
        Z_U[indice] = 0
        Z_L_T[indice] = 0
        Z_sym = Z_U + Z_L_T.T
        Sigma = Z_sym           
        Sigma = (Sigma + Sigma.T)/2
        Sigma[np.diag_indices(p)] = 0
    return(Sigma)

