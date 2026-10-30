from code_base import *


### From pipeline.py
import tomli
import shutil
import os
import argparse
from train import train
from sample import sample
from eval_catboost import train_catboost
from eval_mlp import train_mlp
from eval_simple import train_simple
import pandas as pd
import matplotlib.pyplot as plt
import zero
import lib
import torch


### From sample.py
import torch
import numpy as np
import zero
import os
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from tab_ddpm.utils import FoundNANsError
from utils_train import get_model, make_dataset
from lib import round_columns
import lib


def sample_MC( # same input as sample function except adding B
    parent_dir,
    real_data_path = 'data/higgs-small',
    batch_size = 2000,
    num_samples = 0,
    model_type = 'mlp',
    model_params = None,
    model_path = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    disbalance = None,
    device = torch.device('cuda:1'),
    seed = 0,
    change_val = False,
    B = 50
):
    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)
    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(D.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes('train')
    )

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model, num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
    )

    diffusion.to(device)
    diffusion.eval()

    _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)


    ####################################################################################### MC codes start from here, previous code same as sample.py
    
    if os.path.isfile(os.path.join(real_data_path, "X_num_train.npy")):
        num_numerical_features = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True).shape[1]
    else:
        num_numerical_features = 0
    
    num_numerical_features = num_numerical_features + int(D.is_regression and not model_params["is_y_cond"])

    for b in range(B):

        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)
        
        ######################################################################################################    
        X_gen, y_gen = x_gen.numpy(), y_gen.numpy()
        
        X_num_ = X_gen
        if num_numerical_features < X_gen.shape[1]:
            
            if T_dict['cat_encoding'] == 'one-hot':
                X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
            X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])


        if num_numerical_features_ != 0:
            
            X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
            X_num = X_num_[:, :num_numerical_features]

            X_num_real = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
            disc_cols = []
            for col in range(X_num_real.shape[1]):
                uniq_vals = np.unique(X_num_real[:, col])
                if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                    disc_cols.append(col)
            print("Discrete cols:", disc_cols)
            if model_params['num_classes'] == 0:
                y_gen = X_num[:, 0]
                X_num = X_num[:, 1:]
            if len(disc_cols):
                X_num = round_columns(X_num_real, X_num, disc_cols)

        ######################################################################################################
        
        if os.path.isfile(os.path.join(real_data_path, "X_num_train.npy")):
            X_num_train = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
            
            row_max = np.max(abs(X_num),axis=1)
            large_obs_idx = np.where(row_max > np.max(abs(X_num_train)))
            scaling_factor = np.max(abs(X_num)) / np.max(abs(X_num_train))   
            print(len(large_obs_idx[0])); print(T_dict['normalization']); print(D.is_regression)
            
            if T_dict['normalization'] == 'standard':
                X_num[large_obs_idx,:] = X_num[large_obs_idx,:] / scaling_factor
                if D.is_regression:
                    y_gen[large_obs_idx] = y_gen[large_obs_idx] / scaling_factor
        
        #####################################################################################################################
        #####################################################################################################################

        if num_numerical_features_ != 0: 
            if b == 0:
                N,p_num = X_num.shape
                X_num_gen_store = np.zeros([N,p_num,B]); 
                X_num_gen_store[:,:,b] = X_num; 
            else:        
                X_num_gen_store[:,:,b] = X_num;
        else:
            X_num_gen_store = [];
        if num_numerical_features < X_gen.shape[1]:
            if b == 0:
                N,p_cat = X_cat.shape
                X_cat_gen_store = np.zeros([N,p_cat,B]); 
                X_cat_gen_store[:,:,b] = X_cat; 
            else:        
                X_cat_gen_store[:,:,b] = X_cat;
        else:
            X_cat_gen_store = [];
    
        if b == 0:
            y_gen_store = np.zeros([N,B])
            y_gen_store[:,b] = y_gen
        else:
            y_gen_store[:,b] = y_gen

    
    np.save(os.path.join(parent_dir, 'X_num_gen_storeII'), X_num_gen_store)
    np.save(os.path.join(parent_dir, 'y_gen_storeII'), y_gen_store)
    
    return(X_num_gen_store,X_cat_gen_store,y_gen_store)


    



##########################################################################################################################################            

def LinearRegression_variable_selection(        
    X_train = None,
    y_train = None,
    sim_setup = None,
    n_configs = None,
    fit_intercept=False,
    change_val = False,
    thres_list = [0.05,0.1,0.25,0.5,0.75,0.9],
    MC_samples = 20,
    EBIC_gamma = 0.5
):
    
    [N,p] = np.shape(X_train)
    num_samples0 = N
    
    crit_configs = np.zeros([n_configs,])
    EBIC_train = np.zeros([n_configs,len(thres_list)])
    est_active_set_diffusion_configs = np.zeros([p,n_configs,len(thres_list)])

    synthetic_X_data_configs = np.zeros([N,p,MC_samples,n_configs])
    synthetic_y_data_configs = np.zeros([N,MC_samples,n_configs])


    ########################################################################################################################
    
    for l in range(n_configs):
        
        config_path = "exp/{}/ddpm_tune_best/config{}.toml".format(sim_setup,l)
        
        raw_config = lib.load_config(config_path)
        if 'device' in raw_config:
            device = torch.device(raw_config['device'])
        else:
            device = torch.device('cuda:1')
        
        # Train: from train part in pipeline.py
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=change_val
        )
        
        # Sample: from sample part in pipeline.py
        X_num_gen_store,X_cat_gen_store,y_gen_store = sample_MC(
            num_samples=num_samples0,        
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=change_val,
            B = MC_samples
        )
        
        X_gen_store = X_num_gen_store
        y_gen_store = y_gen_store
        
        synthetic_X_data_configs[:,:,:,l] = X_num_gen_store
        synthetic_y_data_configs[:,:,l] = y_gen_store
        
        ########################################################################################################################
        B = y_gen_store.shape[1]
        p = X_gen_store.shape[1]
        
        count = np.zeros(p)
        
        crit_store = []
            
        for b in range(B):
            
            X = X_gen_store[:,:,b]; y = y_gen_store[:,b]
            
            alpha_optimal = choose_para_EBIC(X,y,fit_intercept=fit_intercept,gamma=EBIC_gamma)
            clf = linear_model.Lasso(alpha=alpha_optimal,fit_intercept=fit_intercept)
            clf.fit(X,y)
            beta_hat_optimal = clf.coef_
            beta0_hat_optimal = clf.intercept_
        
            count = count + np.where(beta_hat_optimal !=0,1,0)
            
            Xy_train = np.concatenate([X_train,np.reshape(y_train,[len(y_train),1])],axis=1)
            Xy = np.concatenate([X,np.reshape(y,[len(y),1])],axis=1)
            crit_b = calculate_fid(Xy_train,Xy)
            
            crit_store.append(crit_b)
        
        crit = np.mean(crit_store)
        crit_configs[l] = crit
        
        pvalue_lasso = 1-count/B
        pvalue_lasso = np.round(pvalue_lasso,5)
        
        
        for r in range(len(thres_list)):
            
            thres = thres_list[r]
            
            est_active_set_diffusion = np.where(np.array(pvalue_lasso) <= thres ,1,0)
            est_active_set_diffusion_configs[:,l,r] = est_active_set_diffusion
            
            beta_train_refit,beta0_train_refit = refit_LinearRegression(X_train,y_train,est_active_set_diffusion,fit_intercept=fit_intercept)
            
            EBIC_train[l,r] = EBIC_LinearRegression(X_train,y_train,beta_train_refit,beta0_train_refit,gamma=EBIC_gamma)
    
    
    best_config_all = np.where(EBIC_train == np.min(EBIC_train))
    
    best_config = [i[0] for i in best_config_all]
    best_config = tuple(best_config)
    
    synthetic_X_data_saved = synthetic_X_data_configs[:,:,:,best_config[0]]
    synthetic_y_data_saved = synthetic_y_data_configs[:,:,best_config[0]]
    
    est_active_set_diffusion_optimal = est_active_set_diffusion_configs[:,best_config[0],best_config[1]]
    
    return(synthetic_X_data_saved,synthetic_y_data_saved,est_active_set_diffusion_optimal)


















############################################################################################################################################
### FID
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

# https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

############################################################################################################################################



def LinearRegression_variable_selection_FID(        
    X_train = None,
    y_train = None,
    sim_setup = None,
    n_configs = None,
    fit_intercept=False,
    change_val = False,
    thres_list = [0.05,0.1,0.25,0.5,0.75,0.9],
    MC_samples = 20,
    EBIC_gamma = 0.5
):
    
    [N,p] = np.shape(X_train)
    num_samples0 = N
    
    crit_configs = np.zeros([n_configs,])
    EBIC_train = np.zeros([n_configs,len(thres_list)])
    est_active_set_diffusion_configs = np.zeros([p,n_configs,len(thres_list)])

    synthetic_X_data_configs = np.zeros([N,p,MC_samples,n_configs])
    synthetic_y_data_configs = np.zeros([N,MC_samples,n_configs])


    ########################################################################################################################
    
    for l in range(n_configs):
        
        config_path = "exp/{}/ddpm_tune_best/config{}.toml".format(sim_setup,l)
        
        raw_config = lib.load_config(config_path)
        if 'device' in raw_config:
            device = torch.device(raw_config['device'])
        else:
            device = torch.device('cuda:1')
        
        # Train: from train part in pipeline.py
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=change_val
        )
        
        # Sample: from sample part in pipeline.py
        X_num_gen_store,X_cat_gen_store,y_gen_store = sample_MC(
            num_samples=num_samples0,        
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=change_val,
            B = MC_samples
        )
        
        X_gen_store = X_num_gen_store
        y_gen_store = y_gen_store
        
        synthetic_X_data_configs[:,:,:,l] = X_num_gen_store
        synthetic_y_data_configs[:,:,l] = y_gen_store
        
        ########################################################################################################################
        B = y_gen_store.shape[1]
        p = X_gen_store.shape[1]
        
        count = np.zeros(p)
        
        crit_store = []
            
        for b in range(B):
            
            X = X_gen_store[:,:,b]; y = y_gen_store[:,b]
            
            alpha_optimal = choose_para_EBIC(X,y,fit_intercept=fit_intercept,gamma=EBIC_gamma)
            clf = linear_model.Lasso(alpha=alpha_optimal,fit_intercept=fit_intercept)
            clf.fit(X,y)
            beta_hat_optimal = clf.coef_
            beta0_hat_optimal = clf.intercept_
        
            count = count + np.where(beta_hat_optimal !=0,1,0)
            
            Xy_train = np.concatenate([X_train,np.reshape(y_train,[len(y_train),1])],axis=1)
            Xy = np.concatenate([X,np.reshape(y,[len(y),1])],axis=1)
            crit_b = calculate_fid(Xy_train,Xy)
            
            crit_store.append(crit_b)
        
        crit = np.mean(crit_store)
        crit_configs[l] = crit
        
        pvalue_lasso = 1-count/B
        pvalue_lasso = np.round(pvalue_lasso,5)
        
        
        for r in range(len(thres_list)):
            
            thres = thres_list[r]
            
            est_active_set_diffusion = np.where(np.array(pvalue_lasso) <= thres ,1,0)
            est_active_set_diffusion_configs[:,l,r] = est_active_set_diffusion
            
            beta_train_refit,beta0_train_refit = refit_LinearRegression(X_train,y_train,est_active_set_diffusion,fit_intercept=fit_intercept)
            
            EBIC_train[l,r] = EBIC_LinearRegression(X_train,y_train,beta_train_refit,beta0_train_refit,gamma=EBIC_gamma)
    
    
    best_config_idx = [i[0] for i in np.where(crit_configs == np.min(crit_configs))]
    best_config_idx = best_config_idx[0]
    best_thres_idx = [i[0] for i in np.where(EBIC_train[best_config_idx,:] == np.min(EBIC_train[best_config_idx,:]))]
    best_thres_idx = best_thres_idx[0]
    
    synthetic_X_data_saved = synthetic_X_data_configs[:,:,:,best_config_idx]
    synthetic_y_data_saved = synthetic_y_data_configs[:,:,best_config_idx]
    
    est_active_set_diffusion_optimal = est_active_set_diffusion_configs[:,best_config_idx,best_thres_idx]
    
    return(synthetic_X_data_saved,synthetic_y_data_saved,est_active_set_diffusion_optimal)





















############################################################################################################################################
### Logistic Regression

def LogisticRegression_variable_selection(        
    X_train = None,
    y_train = None,
    sim_setup = None,
    n_configs = None,
    fit_intercept=True,
    change_val = False,
    thres_list = [0.05,0.1,0.25,0.5,0.75,0.9],
    MC_samples = 20,
    EBIC_gamma = 0.5
):
    
    [N,p] = np.shape(X_train)
    num_samples0 = N
    
    crit_configs = np.zeros([n_configs,])
    EBIC_train = np.zeros([n_configs,len(thres_list)])
    est_active_set_diffusion_configs = np.zeros([p,n_configs,len(thres_list)])

    synthetic_X_data_configs = np.zeros([N,p,MC_samples,n_configs])
    synthetic_y_data_configs = np.zeros([N,MC_samples,n_configs])


    ########################################################################################################################
    
    for l in range(n_configs):
        
        config_path = "exp/{}/ddpm_tune_best/config{}.toml".format(sim_setup,l)
        
        raw_config = lib.load_config(config_path)
        if 'device' in raw_config:
            device = torch.device(raw_config['device'])
        else:
            device = torch.device('cuda:1')
        
        # Train: from train part in pipeline.py
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=change_val
        )
        
        # Sample: from sample part in pipeline.py
        X_num_gen_store,X_cat_gen_store,y_gen_store = sample_MC(
            num_samples=num_samples0,        
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=change_val,
            B = MC_samples
        )
        
        X_gen_store = X_num_gen_store
        y_gen_store = y_gen_store
        
        synthetic_X_data_configs[:,:,:,l] = X_num_gen_store
        synthetic_y_data_configs[:,:,l] = y_gen_store
        
        ########################################################################################################################
        B = y_gen_store.shape[1]
        p = X_gen_store.shape[1]
        
        count = np.zeros(p)
        
        crit_store = []
            
        for b in range(B):
            
            X = X_gen_store[:,:,b]; y = y_gen_store[:,b]
            
            alpha_optimal = choose_para_EBIC_LogisticRegression(X,y,fit_intercept=fit_intercept,gamma=EBIC_gamma)
            clf = LogisticRegression(penalty='l1',solver='liblinear',C=alpha_optimal,fit_intercept=fit_intercept)
            clf.fit(X,y)
            beta_hat_optimal = clf.coef_
            beta0_hat_optimal = clf.intercept_
        
            beta_hat_optimal = np.reshape(beta_hat_optimal, [p,])
            
            count = count + np.where(beta_hat_optimal !=0,1,0)
            
            Xy_train = np.concatenate([X_train,np.reshape(y_train,[len(y_train),1])],axis=1)
            Xy = np.concatenate([X,np.reshape(y,[len(y),1])],axis=1)
            crit_b = calculate_fid(Xy_train,Xy)
            
            crit_store.append(crit_b)
        
        crit = np.mean(crit_store)
        crit_configs[l] = crit
        
        pvalue_lasso = 1-count/B
        pvalue_lasso = np.round(pvalue_lasso,5)
        
        
        for r in range(len(thres_list)):
            
            thres = thres_list[r]
            
            est_active_set_diffusion = np.where(np.array(pvalue_lasso) <= thres ,1,0)
            est_active_set_diffusion_configs[:,l,r] = est_active_set_diffusion
            
            beta_train_refit,beta0_train_refit = refit_LogisticRegression(X_train,y_train,est_active_set_diffusion,fit_intercept=fit_intercept)
            
            EBIC_train[l,r] = EBIC_LogisticRegression(X_train,y_train,beta_train_refit,beta0_train_refit,gamma=EBIC_gamma)
    
    
    best_config_all = np.where(EBIC_train == np.min(EBIC_train))
    
    best_config = [i[0] for i in best_config_all]
    best_config = tuple(best_config)
    
    synthetic_X_data_saved = synthetic_X_data_configs[:,:,:,best_config[0]]
    synthetic_y_data_saved = synthetic_y_data_configs[:,:,best_config[0]]
    
    est_active_set_diffusion_optimal = est_active_set_diffusion_configs[:,best_config[0],best_config[1]]
    
    return(synthetic_X_data_saved,synthetic_y_data_saved,est_active_set_diffusion_optimal)










def LogisticRegression_variable_selection_FID(        
    X_train = None,
    y_train = None,
    sim_setup = None,
    n_configs = None,
    fit_intercept=True,
    change_val = False,
    thres_list = [0.05,0.1,0.25,0.5,0.75,0.9],
    MC_samples = 20,
    EBIC_gamma = 0.5
):
    
    [N,p] = np.shape(X_train)
    num_samples0 = N
    
    crit_configs = np.zeros([n_configs,])
    EBIC_train = np.zeros([n_configs,len(thres_list)])
    est_active_set_diffusion_configs = np.zeros([p,n_configs,len(thres_list)])

    synthetic_X_data_configs = np.zeros([N,p,MC_samples,n_configs])
    synthetic_y_data_configs = np.zeros([N,MC_samples,n_configs])


    ########################################################################################################################
    
    for l in range(n_configs):
        
        config_path = "exp/{}/ddpm_tune_best/config{}.toml".format(sim_setup,l)
        
        raw_config = lib.load_config(config_path)
        if 'device' in raw_config:
            device = torch.device(raw_config['device'])
        else:
            device = torch.device('cuda:1')
        
        # Train: from train part in pipeline.py
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=change_val
        )
        
        # Sample: from sample part in pipeline.py
        X_num_gen_store,X_cat_gen_store,y_gen_store = sample_MC(
            num_samples=num_samples0,        
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=change_val,
            B = MC_samples
        )
        
        X_gen_store = X_num_gen_store
        y_gen_store = y_gen_store
        
        synthetic_X_data_configs[:,:,:,l] = X_num_gen_store
        synthetic_y_data_configs[:,:,l] = y_gen_store
        
        ########################################################################################################################
        B = y_gen_store.shape[1]
        p = X_gen_store.shape[1]
        
        count = np.zeros(p)
        
        crit_store = []
            
        for b in range(B):
            
            X = X_gen_store[:,:,b]; y = y_gen_store[:,b]
            
            alpha_optimal = choose_para_EBIC_LogisticRegression(X,y,fit_intercept=fit_intercept,gamma=EBIC_gamma)
            clf = LogisticRegression(penalty='l1',solver='liblinear',C=alpha_optimal,fit_intercept=fit_intercept)
            clf.fit(X,y)
            beta_hat_optimal = clf.coef_
            beta0_hat_optimal = clf.intercept_
        
            beta_hat_optimal = np.reshape(beta_hat_optimal, [p,])
            
            count = count + np.where(beta_hat_optimal !=0,1,0)
            
            Xy_train = np.concatenate([X_train,np.reshape(y_train,[len(y_train),1])],axis=1)
            Xy = np.concatenate([X,np.reshape(y,[len(y),1])],axis=1)
            crit_b = calculate_fid(Xy_train,Xy)
            
            crit_store.append(crit_b)
        
        crit = np.mean(crit_store)
        crit_configs[l] = crit
        
        pvalue_lasso = 1-count/B
        pvalue_lasso = np.round(pvalue_lasso,5)
        
        
        for r in range(len(thres_list)):
            
            thres = thres_list[r]
            
            est_active_set_diffusion = np.where(np.array(pvalue_lasso) <= thres ,1,0)
            est_active_set_diffusion_configs[:,l,r] = est_active_set_diffusion
            
            beta_train_refit,beta0_train_refit = refit_LogisticRegression(X_train,y_train,est_active_set_diffusion,fit_intercept=fit_intercept)
            
            EBIC_train[l,r] = EBIC_LogisticRegression(X_train,y_train,beta_train_refit,beta0_train_refit,gamma=EBIC_gamma)
    
    
    best_config_idx = [i[0] for i in np.where(crit_configs == np.min(crit_configs))]
    best_config_idx = best_config_idx[0]
    best_thres_idx = [i[0] for i in np.where(EBIC_train[best_config_idx,:] == np.min(EBIC_train[best_config_idx,:]))]
    best_thres_idx = best_thres_idx[0]
    
    synthetic_X_data_saved = synthetic_X_data_configs[:,:,:,best_config_idx]
    synthetic_y_data_saved = synthetic_y_data_configs[:,:,best_config_idx]
    
    est_active_set_diffusion_optimal = est_active_set_diffusion_configs[:,best_config_idx,best_thres_idx]
    
    return(synthetic_X_data_saved,synthetic_y_data_saved,est_active_set_diffusion_optimal)













############################################################################################################################################
### Neighborhood Selection
from neighborhood_selection_code_base import *

def NeighborhoodSelection_graph_selection(        
    X_train = None,
    y_train = None,
    sim_setup = None,
    n_configs = None,
    change_val = False,
    thres_list = [0.05,0.1,0.25,0.5,0.75,0.9],
    MC_samples = 20,
    EBIC_gamma = 0.5
):
    
    
    data_train = np.concatenate([X_train,np.reshape(y_train,[len(y_train),1])],axis=1)
    
    n = np.shape(data_train)[0]
    
    data_train_centered = data_train - data_train.mean(axis=0)
    
    empirical_cov = 1/n * data_train_centered.T @ data_train_centered
    
    [N,p] = np.shape(data_train)
    num_samples0 = N
    
    crit_configs = np.zeros([n_configs,])
    EBIC_train = np.zeros([n_configs,len(thres_list)])
    est_active_set_diffusion_configs = np.zeros([p*(p-1),n_configs,len(thres_list)])
    
    synthetic_X_data_configs = np.zeros([N,p,MC_samples,n_configs])


    ########################################################################################################################
    
    for l in range(n_configs):
        
        config_path = "exp/{}/ddpm_tune_best/config{}.toml".format(sim_setup,l)
        
        raw_config = lib.load_config(config_path)
        if 'device' in raw_config:
            device = torch.device(raw_config['device'])
        else:
            device = torch.device('cuda:1')
        
        # Train: from train part in pipeline.py
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=change_val
        )
        
        # Sample: from sample part in pipeline.py
        X_num_gen_store,X_cat_gen_store,y_gen_store = sample_MC(
            num_samples=num_samples0,        
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=change_val,
            B = MC_samples
        )
        
        X_gen_store = X_num_gen_store
        y_gen_store = y_gen_store
        
        for b in range(MC_samples):            
            synthetic_X_data_configs[:,:,b,l] = np.concatenate([X_num_gen_store[:,:,b],np.reshape(y_gen_store[:,b],[len(y_gen_store[:,b]),1])],axis=1)
        
        
        ########################################################################################################################
        B = y_gen_store.shape[1]
        p = X_gen_store.shape[1] + 1
        
        Sigma_est_store = []
        Sigma_all = 0
        
        crit_store = []
            
        for b in range(B):
            
            X = X_gen_store[:,:,b]; y = y_gen_store[:,b]
            
            y = np.reshape(y,[len(y),1])
            data = np.concatenate([X,y],axis=1)  
            
            Sigma_est = neighb_select(data,gamma=EBIC_gamma)
                    
            Sigma_est_store.append(Sigma_est)
            Sigma_all = Sigma_all + np.where(Sigma_est !=0,1,0)
            
            crit_b = calculate_fid(data_train,data)
            
            crit_store.append(crit_b)
        
        crit = np.mean(crit_store)
        crit_configs[l] = crit
        
        Pvalue_matrix = 1 - Sigma_all/B
        Pvalue_matrix = np.round(Pvalue_matrix,5)
        
        Sigma_est_synthetic_off_diag = np.extract(1 - np.eye(Pvalue_matrix.shape[0]), Pvalue_matrix)
        
        
        for r in range(len(thres_list)):
            
            thres = thres_list[r]
            
            est_active_set_diffusion = np.where(Sigma_est_synthetic_off_diag <= thres ,1,0)
            est_active_set_diffusion_configs[:,l,r] = est_active_set_diffusion
            
            Sigma_hat_active_est = np.where(Pvalue_matrix <= thres ,1,0)
            np.fill_diagonal(Sigma_hat_active_est,1)
            
            Sigma_refit = refit_GaussianGM(empirical_cov,Sigma_hat_active_est)
            
            EBIC_train[l,r] = EBIC_GaussianGM(empirical_cov,Sigma_refit,n,gamma=EBIC_gamma)
    
    
    best_config_all = np.where(EBIC_train == np.min(EBIC_train))
    
    best_config = [i[0] for i in best_config_all]
    best_config = tuple(best_config)
    
    synthetic_X_data_saved = synthetic_X_data_configs[:,:,:,best_config[0]]
    
    est_active_set_diffusion_optimal = est_active_set_diffusion_configs[:,best_config[0],best_config[1]]
    
    return(synthetic_X_data_saved,est_active_set_diffusion_optimal)







############################################################################################################################################

def NeighborhoodSelection_graph_selection_FID(        
    X_train = None,
    y_train = None,
    sim_setup = None,
    n_configs = None,
    change_val = False,
    thres_list = [0.05,0.1,0.25,0.5,0.75,0.9],
    MC_samples = 20,
    EBIC_gamma = 0.5
):
    
    
    data_train = np.concatenate([X_train,np.reshape(y_train,[len(y_train),1])],axis=1)
    
    n = np.shape(data_train)[0]
    
    data_train_centered = data_train - data_train.mean(axis=0)
    
    empirical_cov = 1/n * data_train_centered.T @ data_train_centered
    
    [N,p] = np.shape(data_train)
    num_samples0 = N
    
    crit_configs = np.zeros([n_configs,])
    EBIC_train = np.zeros([n_configs,len(thres_list)])
    est_active_set_diffusion_configs = np.zeros([p*(p-1),n_configs,len(thres_list)])
    
    synthetic_X_data_configs = np.zeros([N,p,MC_samples,n_configs])


    ########################################################################################################################
    
    for l in range(n_configs):
        
        config_path = "exp/{}/ddpm_tune_best/config{}.toml".format(sim_setup,l)
        
        raw_config = lib.load_config(config_path)
        if 'device' in raw_config:
            device = torch.device(raw_config['device'])
        else:
            device = torch.device('cuda:1')
        
        # Train: from train part in pipeline.py
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=change_val
        )
        
        # Sample: from sample part in pipeline.py
        X_num_gen_store,X_cat_gen_store,y_gen_store = sample_MC(
            num_samples=num_samples0,        
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=change_val,
            B = MC_samples
        )
        
        X_gen_store = X_num_gen_store
        y_gen_store = y_gen_store
        
        for b in range(MC_samples):            
            synthetic_X_data_configs[:,:,b,l] = np.concatenate([X_num_gen_store[:,:,b],np.reshape(y_gen_store[:,b],[len(y_gen_store[:,b]),1])],axis=1)
        
        
        ########################################################################################################################
        B = y_gen_store.shape[1]
        p = X_gen_store.shape[1] + 1
        
        Sigma_est_store = []
        Sigma_all = 0
        
        crit_store = []
            
        for b in range(B):
            
            X = X_gen_store[:,:,b]; y = y_gen_store[:,b]
            
            y = np.reshape(y,[len(y),1])
            data = np.concatenate([X,y],axis=1)  
            
            Sigma_est = neighb_select(data,gamma=EBIC_gamma)
                    
            Sigma_est_store.append(Sigma_est)
            Sigma_all = Sigma_all + np.where(Sigma_est !=0,1,0)
            
            crit_b = calculate_fid(data_train,data)
            
            crit_store.append(crit_b)
        
        crit = np.mean(crit_store)
        crit_configs[l] = crit
        
        Pvalue_matrix = 1 - Sigma_all/B
        Pvalue_matrix = np.round(Pvalue_matrix,5)
        
        Sigma_est_synthetic_off_diag = np.extract(1 - np.eye(Pvalue_matrix.shape[0]), Pvalue_matrix)
        
        
        for r in range(len(thres_list)):
            
            thres = thres_list[r]
            
            est_active_set_diffusion = np.where(Sigma_est_synthetic_off_diag <= thres ,1,0)
            est_active_set_diffusion_configs[:,l,r] = est_active_set_diffusion
            
            Sigma_hat_active_est = np.where(Pvalue_matrix <= thres ,1,0)
            np.fill_diagonal(Sigma_hat_active_est,1)
            
            Sigma_refit = refit_GaussianGM(empirical_cov,Sigma_hat_active_est)
            
            EBIC_train[l,r] = EBIC_GaussianGM(empirical_cov,Sigma_refit,n,gamma=EBIC_gamma)
    
    
    best_config_idx = [i[0] for i in np.where(crit_configs == np.min(crit_configs))]
    best_config_idx = best_config_idx[0]
    best_thres_idx = [i[0] for i in np.where(EBIC_train[best_config_idx,:] == np.min(EBIC_train[best_config_idx,:]))]
    best_thres_idx = best_thres_idx[0]
    
    synthetic_X_data_saved = synthetic_X_data_configs[:,:,:,best_config_idx]
    
    est_active_set_diffusion_optimal = est_active_set_diffusion_configs[:,best_config_idx,best_thres_idx]
    
    return(synthetic_X_data_saved,est_active_set_diffusion_optimal)



    

    
    return(synthetic_X_data_saved,est_active_set_diffusion_optimal)





### For configs that give the same EBIC/variable selection accuracy, choose the one that gives the minimum FID

def LinearRegression_variable_selection_break_EBIC_ties_with_min_FID(        
    X_train = None,
    y_train = None,
    sim_setup = None,
    n_configs = None,
    fit_intercept=False,
    change_val = False,
    thres_list = [0.05,0.1,0.25,0.5,0.75,0.9],
    MC_samples = 20,
    EBIC_gamma = 0.5
):
    
    [N,p] = np.shape(X_train)
    num_samples0 = N
    
    crit_configs = np.zeros([n_configs,])
    EBIC_train = np.zeros([n_configs,len(thres_list)])
    est_active_set_diffusion_configs = np.zeros([p,n_configs,len(thres_list)])

    synthetic_X_data_configs = np.zeros([N,p,MC_samples,n_configs])
    synthetic_y_data_configs = np.zeros([N,MC_samples,n_configs])


    ########################################################################################################################
    
    for l in range(n_configs):
        
        config_path = "exp/{}/ddpm_tune_best/config{}.toml".format(sim_setup,l)
        
        raw_config = lib.load_config(config_path)
        if 'device' in raw_config:
            device = torch.device(raw_config['device'])
        else:
            device = torch.device('cuda:1')
        
        # Train: from train part in pipeline.py
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=change_val
        )
        
        # Sample: from sample part in pipeline.py
        X_num_gen_store,X_cat_gen_store,y_gen_store = sample_MC(
            num_samples=num_samples0,        
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=change_val,
            B = MC_samples
        )
        
        X_gen_store = X_num_gen_store
        y_gen_store = y_gen_store
        
        synthetic_X_data_configs[:,:,:,l] = X_num_gen_store
        synthetic_y_data_configs[:,:,l] = y_gen_store
        
        ########################################################################################################################
        B = y_gen_store.shape[1]
        p = X_gen_store.shape[1]
        
        count = np.zeros(p)
        
        crit_store = []
            
        for b in range(B):
            
            X = X_gen_store[:,:,b]; y = y_gen_store[:,b]
            
            alpha_optimal = choose_para_EBIC(X,y,fit_intercept=fit_intercept,gamma=EBIC_gamma)
            clf = linear_model.Lasso(alpha=alpha_optimal,fit_intercept=fit_intercept)
            clf.fit(X,y)
            beta_hat_optimal = clf.coef_
            beta0_hat_optimal = clf.intercept_
        
            count = count + np.where(beta_hat_optimal !=0,1,0)
            
            Xy_train = np.concatenate([X_train,np.reshape(y_train,[len(y_train),1])],axis=1)
            Xy = np.concatenate([X,np.reshape(y,[len(y),1])],axis=1)
            crit_b = calculate_fid(Xy_train,Xy)
            
            crit_store.append(crit_b)
        
        crit = np.mean(crit_store)
        crit_configs[l] = crit
        
        pvalue_lasso = 1-count/B
        pvalue_lasso = np.round(pvalue_lasso,5)
        
        
        for r in range(len(thres_list)):
            
            thres = thres_list[r]
            
            est_active_set_diffusion = np.where(np.array(pvalue_lasso) <= thres ,1,0)
            est_active_set_diffusion_configs[:,l,r] = est_active_set_diffusion
            
            beta_train_refit,beta0_train_refit = refit_LinearRegression(X_train,y_train,est_active_set_diffusion,fit_intercept=fit_intercept)
            
            EBIC_train[l,r] = EBIC_LinearRegression(X_train,y_train,beta_train_refit,beta0_train_refit,gamma=EBIC_gamma)
    
    
    
    best_config_all = np.where(EBIC_train == np.min(EBIC_train))
    
    configs_idx_with_optimal_EBIC_ties = best_config_all[0]
    
    if (len(configs_idx_with_optimal_EBIC_ties) > 1):
        configs_idx_with_optimal_EBIC_ties_with_min_FID = [i[0] for i in np.where(crit_configs[configs_idx_with_optimal_EBIC_ties] == np.min(crit_configs[configs_idx_with_optimal_EBIC_ties]))]
        configs_idx_with_optimal_EBIC_ties_with_min_FID  = configs_idx_with_optimal_EBIC_ties_with_min_FID[0]
        best_config = [i[configs_idx_with_optimal_EBIC_ties_with_min_FID] for i in best_config_all]
        best_config = tuple(best_config)
    else:
        best_config = [i[0] for i in best_config_all]
        best_config = tuple(best_config)
    
    synthetic_X_data_saved = synthetic_X_data_configs[:,:,:,best_config[0]]
    synthetic_y_data_saved = synthetic_y_data_configs[:,:,best_config[0]]
    
    est_active_set_diffusion_optimal = est_active_set_diffusion_configs[:,best_config[0],best_config[1]]
    
    return(synthetic_X_data_saved,synthetic_y_data_saved,est_active_set_diffusion_optimal)





############################################################################################################################################
### Logistic Regression

def LogisticRegression_variable_selection_break_EBIC_ties_with_min_FID(        
    X_train = None,
    y_train = None,
    sim_setup = None,
    n_configs = None,
    fit_intercept=True,
    change_val = False,
    thres_list = [0.05,0.1,0.25,0.5,0.75,0.9],
    MC_samples = 20,
    EBIC_gamma = 0.5
):
    
    [N,p] = np.shape(X_train)
    num_samples0 = N
    
    crit_configs = np.zeros([n_configs,])
    EBIC_train = np.zeros([n_configs,len(thres_list)])
    est_active_set_diffusion_configs = np.zeros([p,n_configs,len(thres_list)])

    synthetic_X_data_configs = np.zeros([N,p,MC_samples,n_configs])
    synthetic_y_data_configs = np.zeros([N,MC_samples,n_configs])


    ########################################################################################################################
    
    for l in range(n_configs):
        
        config_path = "exp/{}/ddpm_tune_best/config{}.toml".format(sim_setup,l)
        
        raw_config = lib.load_config(config_path)
        if 'device' in raw_config:
            device = torch.device(raw_config['device'])
        else:
            device = torch.device('cuda:1')
        
        # Train: from train part in pipeline.py
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=change_val
        )
        
        # Sample: from sample part in pipeline.py
        X_num_gen_store,X_cat_gen_store,y_gen_store = sample_MC(
            num_samples=num_samples0,        
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=change_val,
            B = MC_samples
        )
        
        X_gen_store = X_num_gen_store
        y_gen_store = y_gen_store
        
        synthetic_X_data_configs[:,:,:,l] = X_num_gen_store
        synthetic_y_data_configs[:,:,l] = y_gen_store
        
        ########################################################################################################################
        B = y_gen_store.shape[1]
        p = X_gen_store.shape[1]
        
        count = np.zeros(p)
        
        crit_store = []
            
        for b in range(B):
            
            X = X_gen_store[:,:,b]; y = y_gen_store[:,b]
            
            alpha_optimal = choose_para_EBIC_LogisticRegression(X,y,fit_intercept=fit_intercept,gamma=EBIC_gamma)
            clf = LogisticRegression(penalty='l1',solver='liblinear',C=alpha_optimal,fit_intercept=fit_intercept)
            clf.fit(X,y)
            beta_hat_optimal = clf.coef_
            beta0_hat_optimal = clf.intercept_
        
            beta_hat_optimal = np.reshape(beta_hat_optimal, [p,])
            
            count = count + np.where(beta_hat_optimal !=0,1,0)
            
            Xy_train = np.concatenate([X_train,np.reshape(y_train,[len(y_train),1])],axis=1)
            Xy = np.concatenate([X,np.reshape(y,[len(y),1])],axis=1)
            crit_b = calculate_fid(Xy_train,Xy)
            
            crit_store.append(crit_b)
        
        crit = np.mean(crit_store)
        crit_configs[l] = crit
        
        pvalue_lasso = 1-count/B
        pvalue_lasso = np.round(pvalue_lasso,5)
        
        
        for r in range(len(thres_list)):
            
            thres = thres_list[r]
            
            est_active_set_diffusion = np.where(np.array(pvalue_lasso) <= thres ,1,0)
            est_active_set_diffusion_configs[:,l,r] = est_active_set_diffusion
            
            beta_train_refit,beta0_train_refit = refit_LogisticRegression(X_train,y_train,est_active_set_diffusion,fit_intercept=fit_intercept)
            
            EBIC_train[l,r] = EBIC_LogisticRegression(X_train,y_train,beta_train_refit,beta0_train_refit,gamma=EBIC_gamma)
    
    
    
    best_config_all = np.where(EBIC_train == np.min(EBIC_train))
    
    configs_idx_with_optimal_EBIC_ties = best_config_all[0]
    
    if (len(configs_idx_with_optimal_EBIC_ties) > 1):
        configs_idx_with_optimal_EBIC_ties_with_min_FID = [i[0] for i in np.where(crit_configs[configs_idx_with_optimal_EBIC_ties] == np.min(crit_configs[configs_idx_with_optimal_EBIC_ties]))]
        configs_idx_with_optimal_EBIC_ties_with_min_FID  = configs_idx_with_optimal_EBIC_ties_with_min_FID[0]
        best_config = [i[configs_idx_with_optimal_EBIC_ties_with_min_FID] for i in best_config_all]
        best_config = tuple(best_config)
    else:
        best_config = [i[0] for i in best_config_all]
        best_config = tuple(best_config)
    
    synthetic_X_data_saved = synthetic_X_data_configs[:,:,:,best_config[0]]
    synthetic_y_data_saved = synthetic_y_data_configs[:,:,best_config[0]]
    
    est_active_set_diffusion_optimal = est_active_set_diffusion_configs[:,best_config[0],best_config[1]]
    
    return(synthetic_X_data_saved,synthetic_y_data_saved,est_active_set_diffusion_optimal)






############################################################################################################################################
### Neighborhood Selection
from neighborhood_selection_code_base import *

def NeighborhoodSelection_graph_selection_break_EBIC_ties_with_min_FID(        
    X_train = None,
    y_train = None,
    sim_setup = None,
    n_configs = None,
    change_val = False,
    thres_list = [0.05,0.1,0.25,0.5,0.75,0.9],
    MC_samples = 20,
    EBIC_gamma = 0.5
):
    
    
    data_train = np.concatenate([X_train,np.reshape(y_train,[len(y_train),1])],axis=1)
    
    n = np.shape(data_train)[0]
    
    data_train_centered = data_train - data_train.mean(axis=0)
    
    empirical_cov = 1/n * data_train_centered.T @ data_train_centered
    
    [N,p] = np.shape(data_train)
    num_samples0 = N
    
    crit_configs = np.zeros([n_configs,])
    EBIC_train = np.zeros([n_configs,len(thres_list)])
    est_active_set_diffusion_configs = np.zeros([p*(p-1),n_configs,len(thres_list)])
    
    synthetic_X_data_configs = np.zeros([N,p,MC_samples,n_configs])


    ########################################################################################################################
    
    for l in range(n_configs):
        
        config_path = "exp/{}/ddpm_tune_best/config{}.toml".format(sim_setup,l)
        
        raw_config = lib.load_config(config_path)
        if 'device' in raw_config:
            device = torch.device(raw_config['device'])
        else:
            device = torch.device('cuda:1')
        
        # Train: from train part in pipeline.py
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=change_val
        )
        
        # Sample: from sample part in pipeline.py
        X_num_gen_store,X_cat_gen_store,y_gen_store = sample_MC(
            num_samples=num_samples0,        
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=change_val,
            B = MC_samples
        )
        
        X_gen_store = X_num_gen_store
        y_gen_store = y_gen_store
        
        for b in range(MC_samples):            
            synthetic_X_data_configs[:,:,b,l] = np.concatenate([X_num_gen_store[:,:,b],np.reshape(y_gen_store[:,b],[len(y_gen_store[:,b]),1])],axis=1)
        
        
        ########################################################################################################################
        B = y_gen_store.shape[1]
        p = X_gen_store.shape[1] + 1
        
        Sigma_est_store = []
        Sigma_all = 0
        
        crit_store = []
            
        for b in range(B):
            
            X = X_gen_store[:,:,b]; y = y_gen_store[:,b]
            
            y = np.reshape(y,[len(y),1])
            data = np.concatenate([X,y],axis=1)  
            
            Sigma_est = neighb_select(data,gamma=EBIC_gamma)
                    
            Sigma_est_store.append(Sigma_est)
            Sigma_all = Sigma_all + np.where(Sigma_est !=0,1,0)
            
            crit_b = calculate_fid(data_train,data)
            
            crit_store.append(crit_b)
        
        crit = np.mean(crit_store)
        crit_configs[l] = crit
        
        Pvalue_matrix = 1 - Sigma_all/B
        Pvalue_matrix = np.round(Pvalue_matrix,5)
        
        Sigma_est_synthetic_off_diag = np.extract(1 - np.eye(Pvalue_matrix.shape[0]), Pvalue_matrix)
        
        
        for r in range(len(thres_list)):
            
            thres = thres_list[r]
            
            est_active_set_diffusion = np.where(Sigma_est_synthetic_off_diag <= thres ,1,0)
            est_active_set_diffusion_configs[:,l,r] = est_active_set_diffusion
            
            Sigma_hat_active_est = np.where(Pvalue_matrix <= thres ,1,0)
            np.fill_diagonal(Sigma_hat_active_est,1)
            
            Sigma_refit = refit_GaussianGM(empirical_cov,Sigma_hat_active_est)
            
            EBIC_train[l,r] = EBIC_GaussianGM(empirical_cov,Sigma_refit,n,gamma=EBIC_gamma)
    
    
    
    best_config_all = np.where(EBIC_train == np.min(EBIC_train))
    
    configs_idx_with_optimal_EBIC_ties = best_config_all[0]
    
    if (len(configs_idx_with_optimal_EBIC_ties) > 1):
        configs_idx_with_optimal_EBIC_ties_with_min_FID = [i[0] for i in np.where(crit_configs[configs_idx_with_optimal_EBIC_ties] == np.min(crit_configs[configs_idx_with_optimal_EBIC_ties]))]
        configs_idx_with_optimal_EBIC_ties_with_min_FID  = configs_idx_with_optimal_EBIC_ties_with_min_FID[0]
        best_config = [i[configs_idx_with_optimal_EBIC_ties_with_min_FID] for i in best_config_all]
        best_config = tuple(best_config)
    else:
        best_config = [i[0] for i in best_config_all]
        best_config = tuple(best_config)
    
    synthetic_X_data_saved = synthetic_X_data_configs[:,:,:,best_config[0]]
    
    est_active_set_diffusion_optimal = est_active_set_diffusion_configs[:,best_config[0],best_config[1]]
    
    return(synthetic_X_data_saved,est_active_set_diffusion_optimal)















