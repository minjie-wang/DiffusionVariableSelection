sim_setup = "new_data_sim_blockdiag_p200"
real_data_path = 'data/{}'.format(sim_setup)
config_path = "exp/{}/ddpm_tune_best".format(sim_setup)


##############################################################################################################################
##############################################################################################################################
### I. Generate data
import numpy as np
import zero
from sklearn.metrics import f1_score

seed = 0
zero.improve_reproducibility(seed)

N = 500
p = 200
n_runs = 10

X_train_data = np.zeros([N,p,n_runs]); X_val_data = np.zeros([N,p,n_runs]); X_test_data = np.zeros([N,p,n_runs])
y_train_data = np.zeros([N,n_runs]); y_val_data = np.zeros([N,n_runs]); y_test_data = np.zeros([N,n_runs])

beta_store = np.zeros([p,n_runs])

### simulate and store data
# design = "i.i.d"
# design = "auto-regressive"
design = "block_diagonal"
rho = 0.9
from scipy.linalg import toeplitz

from random import sample
no_non_zero = 5

true_coef_idx = "first_k"
coef_value = "pos_neg"



for i in range(n_runs):
    
    mean = np.zeros(p)
    if design == "i.i.d":
        cov = np.eye(p)
    elif design == "auto-regressive":
        a = np.power(rho,list(range(p)))
        cov = toeplitz(a)
    elif design == "block_diagonal":
        p_sub = 5 # number of p in a sub-matrix/block
        Sigma_sub = np.ones([p_sub,p_sub])
        Sigma_sub = rho * Sigma_sub
        np.fill_diagonal(Sigma_sub, 1)
        cov = np.kron(np.eye(int(p/p_sub),dtype=int),Sigma_sub)

    X = np.random.multivariate_normal(mean, cov, N)

    # set coef beta
    beta = np.zeros([p,1])
    if design == "i.i.d":
        if true_coef_idx == "first_k":
            
            beta[:no_non_zero] = (2 + 0.1 * np.random.normal(0, 1, size=(no_non_zero, 1)))
            if coef_value == "pos_neg":
                beta[:no_non_zero] = (beta[:no_non_zero].T * np.random.choice([-1,1], no_non_zero ,replace=True)).T
        elif true_coef_idx == "random":
            idx = sample(range(p),no_non_zero)
    
            beta[idx] = (2 + 0.1 * np.random.normal(0, 1, size=(no_non_zero, 1)))
            if coef_value == "pos_neg":
                beta[idx] = (beta[idx].T * np.random.choice([-1,1], no_non_zero ,replace=True)).T
    elif design == "auto-regressive":
        no_non_zero = 10
        beta[:no_non_zero] = np.reshape([2,2,-2,-2,-2,2,2,-2,-2,-2],(10,1))
    elif design == "block_diagonal":
        no_non_zero = 5
        beta[:no_non_zero] = np.reshape([2,2,2,-2,-2],(5,1))

    
    y = np.matmul(X, beta) + np.random.normal(0, 1, size=(N, 1))
    y = np.reshape(y, [N,])
    
    
    X_train = X; y_train = y;
    
    X_val = np.zeros([N,p]); y_val = np.zeros([N,])
    X_test = np.zeros([N,p]); y_test = np.zeros([N,])
    
    
    ### store data
    X_train_data[:,:,i] = X_train; X_val_data[:,:,i] = X_val; X_test_data[:,:,i] = X_test;
    y_train_data[:,i] = y_train; y_val_data[:,i] = y_val; y_test_data[:,i] = y_test;
    
    beta_store[:,i] = np.reshape(beta,[p,])



import os
np.save(os.path.join(real_data_path, 'X_train_data_all'), X_train_data)
np.save(os.path.join(real_data_path, 'y_train_data_all'), y_train_data)
np.save(os.path.join(real_data_path, 'beta_all'), beta_store)




from diffusion_sampling import *
from code_base import *



######################################################################################################################
######################################################################################################################
######################################################################################################################
### Run replicates for simulation

num_samples0 = np.shape(X_train_data)[0]

Fscore_lasso = []
Fscore_lasso2 = np.zeros(n_runs)

seed = 0
zero.improve_reproducibility(seed)


import glob
n_configs = len(glob.glob1(config_path,"*.toml"))
print("Number of configs for tuning:",n_configs)

thres_list = [0.05,0.1,0.25,0.5,0.75,0.9]



MC_samples = 20
synthetic_X_data_saved = np.zeros([N,p,MC_samples,n_runs])
synthetic_y_data_saved = np.zeros([N,MC_samples,n_runs])




for i in range(n_runs):


    ########################################################################################################################
    ###

    X_train = X_train_data[:,:,i]; X_val = X_val_data[:,:,i]; X_test = X_test_data[:,:,i]
    y_train = y_train_data[:,i]; y_val = y_val_data[:,i]; y_test = y_test_data[:,i]
    
    ###

    np.save(os.path.join(real_data_path, 'X_num_train'), X_train)
    np.save(os.path.join(real_data_path, 'y_train'), y_train)

    np.save(os.path.join(real_data_path, 'X_num_val'), X_val)
    np.save(os.path.join(real_data_path, 'y_val'), y_val)

    np.save(os.path.join(real_data_path, 'X_num_test'), X_test)
    np.save(os.path.join(real_data_path, 'y_test'), y_test)

    
    beta = beta_store[:,i]
    active_set = np.where(np.array(beta.T) !=0 ,1,0)

    ###
    synthetic_X_data_saved_optimal,synthetic_y_data_saved_optimal,est_active_set_diffusion_optimal = LinearRegression_variable_selection(X_train,y_train,sim_setup,n_configs=n_configs,thres_list = thres_list,MC_samples=MC_samples)
              
    Fscore_best_config = f1_score(active_set, est_active_set_diffusion_optimal)
    
    Fscore_lasso.append(Fscore_best_config)
    Fscore_lasso2[i] = Fscore_best_config
        
    synthetic_X_data_saved[:,:,:,i] = synthetic_X_data_saved_optimal
    synthetic_y_data_saved[:,:,i] = synthetic_y_data_saved_optimal
    
    
    print("Trial %d" % i)



Fscore_lasso_diffusion = np.mean(Fscore_lasso)
Fscore_lasso_diffusion_se = np.std(Fscore_lasso,ddof=1) / np.sqrt(n_runs)

print("Diffusion:",[Fscore_lasso_diffusion,Fscore_lasso_diffusion_se])

Fscore_lasso_diffusion_runs = Fscore_lasso


np.save(os.path.join(config_path, 'synthetic_X_data_saved'), synthetic_X_data_saved)
np.save(os.path.join(config_path, 'synthetic_y_data_saved'), synthetic_y_data_saved)



# Lasso
Fscore_lasso = []
Fscore_lasso2 = np.zeros(n_runs)

seed = 0
zero.improve_reproducibility(seed)


for i in range(n_runs):
        
    X_train = X_train_data[:,:,i]; X_val = X_val_data[:,:,i]; X_test = X_test_data[:,:,i]
    y_train = y_train_data[:,i]; y_val = y_val_data[:,i]; y_test = y_test_data[:,i]
    
    beta = beta_store[:,i]
    active_set = np.where(np.array(beta.T) !=0 ,1,0)
    
    fit_intercept=False
    alpha_optimal = choose_para_EBIC(X_train,y_train,fit_intercept=fit_intercept)
    clf = linear_model.Lasso(alpha=alpha_optimal,fit_intercept=fit_intercept)
    clf.fit(X_train,y_train)
    beta_hat_optimal = clf.coef_
    beta0_hat_optimal = clf.intercept_

    est_active_set_Lasso = np.where(beta_hat_optimal !=0,1,0)
    
    Fscore_lasso.append(f1_score(active_set, est_active_set_Lasso))
    Fscore_lasso2[i] = f1_score(active_set, est_active_set_Lasso)
    
    
Fscore_lasso_existing = np.mean(Fscore_lasso)
Fscore_lasso_existing_se = np.std(Fscore_lasso,ddof=1) / np.sqrt(n_runs)

print("Existing:",[Fscore_lasso_existing,Fscore_lasso_existing_se])

Fscore_lasso_existing_runs = Fscore_lasso



np.save('Fscore_diffusion_{}.npy'.format(sim_setup), Fscore_lasso_diffusion_runs)
np.save('Fscore_existing_{}.npy'.format(sim_setup), Fscore_lasso_existing_runs)