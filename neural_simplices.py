# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:53:51 2021

@author: mmall
"""


CODE_DIR = 'C:/Users/mmall/Documents/github/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/assignment_errors/fits/'

import socket
import os
import sys
import pickle as pkl

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.font_manager as mfm
import matplotlib.tri as tri
from matplotlib import cm
import matplotlib.colors as clr
import scipy.linalg as la
import scipy.io as sio
import scipy.stats as sts
from sklearn import svm, manifold, linear_model
from sklearn.model_selection import cross_val_score as cv_score
import sklearn.kernel_approximation as kaprx
from tqdm import tqdm
import itertools as itt
import pystan as ps
import arviz as az
import umap

sys.path.append(CODE_DIR+'assignment_errors/')
sys.path.append(CODE_DIR+'assignment_errors/jeffcode/')
import general.data_io as gio
import general.utility as u
import swap_errors.auxiliary as swa
import swap_errors.analysis as swan
import swap_errors.visualization as swv
import general.neural_analysis as na
import general.plotting as gpl
import general.stan_utility as su

import helpers as hlp

sys.path.append(CODE_DIR+'repler/src/')
import util
import plotting as dicplt

special_font = mfm.FontProperties(fname='C:/Windows/Fonts/seguiemj.ttf')

#%%

dset_prm = {'session':list(range(13)), ## delay period 2
 					'regions':['all'],
 					'tzf':'WHEEL_ON_diode',
 					'tbeg':-0.5,
 					'twindow':0.5,
 					'tstep':0.5,
 					'num_bins':5,
 					'do_pca':'before', #'after'
 					'pca_thrs':0.95,
 					'min_trials':40,
 					'shuffle':False,
 					'impute_nan':True,
 					'shuffle_probs':False,
 					'which_block':['pro','retro'],
 					'impute_params':{'weights':'uniform','n_neighbors':5},
 					'color_weights': hlp.Splines(1) # 'softmax'
 					}
# model_type = 'ushh_t_inter'
model_type = 'ushh_inter'

# dset_prm = {'session':list(range(13,23)), # delay period 1
#  					'regions':['all'],
#  					'tzf':'CUE2_ON_diode',
#  					'tbeg':-0.5,
#  					'twindow':0.5,
#  					'tstep':0.5,
#  					'num_bins':5,
#  					'do_pca':'before', #'after'
#  					'pca_thrs':0.95,
#  					'min_trials':40,
#  					'shuffle':False,
#  					'impute_nan':True,
#  					'shuffle_probs':False,
#  					'which_block':'retro',
#  					'impute_params':{'weights':'uniform','n_neighbors':5},
#  					'color_weights': hlp.Splines(1) # 'softmax'
#  					}
# model_type = 'inter_tf_precue'

## funky way of iterating over all the parameters in the dictionary
variable_prms = {k:v for k,v in dset_prm.items() if type(v) is list and k!='session'}
fixed_prms = {k:v for k,v in dset_prm.items() if type(v) is not list and k!='session'}

these_sess = dset_prm['session']
var_k, var_v = zip(*variable_prms.items())

# simplx_basis = np.array([[1,-1,0],[-0.5,-0.5,1]])
simplx_basis = np.array([[1,0,-1],[-0.5,1,-0.5]])
simplx_basis /= la.norm(simplx_basis,axis=1,keepdims=True)

all_probs = []
for vals in list(itt.product(*var_v)):
    this_dset = dict(zip(var_k, vals), **fixed_prms)
    this_dset['tend'] = this_dset['tbeg'] + this_dset['twindow']

    sess_probs = [[] for _ in these_sess]
    for idx, which_sess in enumerate(these_sess):
        this_dset['session'] = which_sess
        dset_info = {**this_dset}
        folds = hlp.folder_hierarchy(dset_info) 
    
        with open(SAVE_DIR+folds+'/arviz_fit_%s_model.pkl'%model_type, 'rb') as f:
            az_fit = pkl.load(f)
            
        probs = az_fit.posterior['p_err'].to_numpy()
        sess_probs[idx] = probs@simplx_basis.T
        
    all_probs.append(sess_probs)

all_probs = np.array(all_probs)

#%%
cmap = 'tab20'

row_labs = var_k[1:]
row_lab_vals = var_v[1:]
col_labs = var_k[:1] 
col_lab_vals = var_v[:1]

# col_labs = var_k[1:]
# col_lab_vals = var_v[1:]
# row_labs = var_k[:1]
# row_lab_vals = var_v[:1]

# contours = True
contours = False

heatmap = True
# heatmap = False

y_ticks = False
# y_ticks = True

share_y_axis = False
# share_y_axis = True

xmin = -0.5*np.sqrt(2)
xmax = 0.5*np.sqrt(2)
ymin = np.sqrt(6)/3 - np.sqrt(1.5)
ymax = np.sqrt(6)/3

if contours:
    xx, yy = np.meshgrid(np.linspace(xmin,xmax,100),np.linspace(ymin,ymax,100))
    foo = (np.stack([xx.flatten(),yy.flatten()]).T@simplx_basis) + [1/3,1/3,1/3]
    support = la.norm(foo,1, axis=-1)<1.001
if heatmap:
    grid = tri.Triangulation(simplx_basis[0,:], simplx_basis[1,:])
    grid = tri.UniformTriRefiner(grid).refine_triangulation(subdiv=6)
    foo = np.stack([grid.x,grid.y])[:,grid.triangles].mean(-1).T@simplx_basis + [1/3,1/3,1/3]
    msk = la.norm(foo,1, axis=-1)>=1.001
    grid.set_mask(msk)

axs = dicplt.hierarchical_labels(row_lab_vals, col_lab_vals,    
                                 row_names=row_labs, col_names=col_labs,
                                 fontsize=13, wmarg=0.3, hmarg=0.1)

    
n_row_lab = np.flip(np.array([1,]+[len(v) for v in row_lab_vals[1:]]))
n_col_lab = np.flip(np.array([1,]+[len(v) for v in col_lab_vals[1:]]))
for k, this_prm in enumerate(itt.product(*var_v)):
    
    col_idx = np.array([np.where(np.isin(var_v[i],this_prm[i]))[0].item() \
                        for i in np.where(np.isin(var_k,col_labs))[0]])
    row_idx = np.array([np.where(np.isin(var_v[i],this_prm[i]))[0].item() \
                        for i in np.where(np.isin(var_k,row_labs))[0]])
    
    if len(col_labs)>=1:
        c = col_idx@n_col_lab
    else:
        c = 0
    if len(row_labs)>=1:
        r = row_idx@n_row_lab
    else:
        r = 0
    
    if contours:
        cols = getattr(cm, cmap)(np.arange(len(these_sess))/len(these_sess))
        for idx, sess in enumerate(these_sess):
            simp = all_probs[k, idx]
            # simp = all_probs[k,idx,:,0,:]
            
            kd_pdf = sts.gaussian_kde(simp.reshape((-1,2)).T)
            zz = np.where(support, kd_pdf(np.stack([xx.flatten(),yy.flatten()])), np.nan)
            
            if heatmap:
                axs[r,c].contour(xx,yy,zz.reshape(100,100,order='A'), 2,
                              colors=['#EC7063','#3498DB'][int(idx>12)], alpha=0.7,
                              linestyles=['solid','dotted'])
            else:
                axs[r,c].contour(xx,yy,zz.reshape(100,100,order='A'), 2,
                                  colors=clr.to_hex(cols[idx]),
                                  linestyles=['solid','dotted'])
            # axs[r,c].contourf(xx,yy,zz.reshape(100,100,order='A'), 2,
            #                  colors=clr.to_hex(cols[idx]),
            #                  alpha=0.7)
    if heatmap:
        # simp = all_probs[k,...,0,:]
        simp = all_probs[k]  
        
        kd_pdf = sts.gaussian_kde(simp.reshape((-1,2)).T)
        
        zz = kd_pdf(np.stack([grid.x,grid.y]))
        
        axs[r,c].tripcolor(grid, zz, rasterized=True, cmap='binary')
    
    axs[r,c].plot([xmin,xmax,0,xmin], [ymin, ymin, ymax, ymin],'k')
    # axs[r,c].plot([xmin,xmax,0,xmin], [ymin, ymin, ymax, ymin],'#A6ACAF')
        
    axs[r,c].set_ylim([ymin*1.1,ymax*1.1])
    axs[r,c].set_xlim([xmin*1.1,xmax*1.1])
    axs[r,c].set_aspect('equal')
    axs[r,c].set_axis_off()
    # dicplt.square_axis(axs[r,c])

#%%

dset_prm = {'session':list(range(13,23)), # delay period 1
					'regions':['all'],
					'tzf':'CUE2_ON_diode',
					'tbeg':-0.5,
					'twindow':0.5,
					'tstep':0.5,
					'num_bins':5,
					'do_pca':'before', #'after'
					'pca_thrs':0.95,
					'min_trials':40,
					'shuffle':False,
					'impute_nan':True,
					'shuffle_probs':False,
					'which_block':'retro',
					'impute_params':{'weights':'uniform','n_neighbors':5},
					'color_weights': hlp.Splines(1) # 'softmax'
					}

model_type = 'inter_t_precue'
# model_type = 'inter_tf_precue'
# model_type = 'null_t_precue'


## funky way of iterating over all the parameters in the dictionary
variable_prms = {k:v for k,v in dset_prm.items() if type(v) is list and k!='session'}
fixed_prms = {k:v for k,v in dset_prm.items() if type(v) is not list and k!='session'}

these_sess = dset_prm['session']
var_k, var_v = zip(*variable_prms.items())

all_probs = []
for vals in (itt.product(*var_v)):
    this_dset = dict(zip(var_k, vals), **fixed_prms)
    this_dset['tend'] = this_dset['tbeg'] + this_dset['twindow']

    sess_probs = [[] for _ in these_sess]
    for idx, which_sess in (enumerate(these_sess)):
        this_dset['session'] = which_sess
        dset_info = {**this_dset}
        folds = hlp.folder_hierarchy(dset_info) 
        
        with open(SAVE_DIR+folds+'/arviz_fit_%s_model.pkl'%model_type, 'rb') as f:
            az_fit = pkl.load(f)
        # with open(SAVE_DIR+folds+'/arviz_fit_inter_tf_precue_model.pkl', 'rb') as f:
        #     az_fit = pkl.load(f)
        # with open(SAVE_DIR+folds+'/arviz_fit_hybrid_error_precue_model.pkl', 'rb') as f:
        #     az_fit = pkl.load(f)
        
        # logits = az_fit.posterior['logits'].to_numpy()
        # sess_probs[idx] = np.exp(logits)/(1+np.exp(logits))
        sess_probs[idx] = az_fit.posterior['p_err'].to_numpy()[...,0]
        
    all_probs.append(sess_probs)

all_probs = np.array(all_probs)

#%%
cmap = 'tab20'

row_labs = var_k[1:]
row_lab_vals = var_v[1:]
col_labs = var_k[:1] 
col_lab_vals = var_v[:1]

# col_labs = var_k[1:]
# col_lab_vals = var_v[1:]
# row_labs = var_k[:1]
# row_lab_vals = var_v[:1]


y_ticks = False
# y_ticks = True

share_y_axis = False
# share_y_axis = True


# show_sessions = True
show_sessions = False

show_all = True
# show_all = False

axs = dicplt.hierarchical_labels(row_lab_vals, col_lab_vals,    
                                 row_names=row_labs, col_names=col_labs,
                                 fontsize=13, wmarg=0.3, hmarg=0.1)

    
n_row_lab = np.flip(np.array([1,]+[len(v) for v in row_lab_vals[1:]]))
n_col_lab = np.flip(np.array([1,]+[len(v) for v in col_lab_vals[1:]]))
for k, this_prm in enumerate(itt.product(*var_v)):
    
    col_idx = np.array([np.where(np.isin(var_v[i],this_prm[i]))[0].item() \
                        for i in np.where(np.isin(var_k,col_labs))[0]])
    row_idx = np.array([np.where(np.isin(var_v[i],this_prm[i]))[0].item() \
                        for i in np.where(np.isin(var_k,row_labs))[0]])
    
    if len(col_labs)>=1:
        c = col_idx@n_col_lab
    else:
        c = 0
    if len(row_labs)>=1:
        r = row_idx@n_row_lab
    else:
        r = 0   

    if show_sessions:
        cols = getattr(cm, cmap)(np.arange(len(these_sess))/len(these_sess))
        for idx, sess in enumerate(these_sess):
            simp = all_probs[k, idx]
            
            kd_pdf = sts.gaussian_kde(simp.flatten())
            zz = kd_pdf(np.linspace(0,1,100))
            
            if show_all:
                axs[r,c].plot(np.linspace(0,1,100), kd_pdf(np.linspace(0,1,100)), color=['#EC7063','#3498DB'][int(idx>12)])
            else:
                axs[r,c].plot(np.linspace(0,1,100), kd_pdf(np.linspace(0,1,100)), color=cols[idx])
        
    if show_all:
        simp = all_probs[k,:,...]
        axs[r,c].hist(simp.flatten(), bins=50, density=True)
        
    axs[r,c].set_xlabel(r'$p_{\rm{sptl}}$')
    axs[r,c].set_ylabel('Posterior density')
    axs[r,c].set_ylim([0, np.max(axs[r,c].get_ylim())])
    axs[r,c].set_xlim([0,1])  
    
    # dicplt.square_axis(axs[r,c])



#%% For the delay 2 forgetting model

dset_prm = {'session':list(range(13,23)), ## delay period 2
 					'regions':['all'],
 					'tzf':'WHEEL_ON_diode',
 					'tbeg':-0.5,
 					'twindow':0.5,
 					'tstep':0.5,
 					'num_bins':5,
 					'do_pca':'before', #'after'
 					'pca_thrs':0.95,
 					'min_trials':40,
 					'shuffle':False,
 					'impute_nan':True,
 					'shuffle_probs':False,
 					'which_block':['pro','retro'],
 					'impute_params':{'weights':'uniform','n_neighbors':5},
 					'color_weights': hlp.Splines(1) # 'softmax'
 					}
model_type = 'ushh_tf_inter'
    

## funky way of iterating over all the parameters in the dictionary
variable_prms = {k:v for k,v in dset_prm.items() if type(v) is list and k!='session'}
fixed_prms = {k:v for k,v in dset_prm.items() if type(v) is not list and k!='session'}

these_sess = dset_prm['session']
var_k, var_v = zip(*variable_prms.items())

# simplx_basis = np.array([[1,-1,0],[-0.5,-0.5,1]])
simplx_basis = np.array([[1,0,-1],[-0.5,1,-0.5]])
simplx_basis /= la.norm(simplx_basis,axis=1,keepdims=True)

all_probs = []
all_null = []
for vals in list(itt.product(*var_v)):
    this_dset = dict(zip(var_k, vals), **fixed_prms)
    this_dset['tend'] = this_dset['tbeg'] + this_dset['twindow']

    sess_probs = [[] for _ in these_sess]
    sess_null = [[] for _ in these_sess]
    for idx, which_sess in enumerate(these_sess):
        this_dset['session'] = which_sess
        dset_info = {**this_dset}
        folds = hlp.folder_hierarchy(dset_info) 
    
        with open(SAVE_DIR+folds+'/arviz_fit_%s_model.pkl'%model_type, 'rb') as f:
            az_fit = pkl.load(f)
            
        probs = az_fit.posterior['p_err'].to_numpy()
        sess_probs[idx] = probs[...,:3]@simplx_basis.T
        sess_null[idx] = probs[...,-1]
        
    all_probs.append(sess_probs)
    all_null.append(sess_null)

all_probs = np.array(all_probs)
all_null = np.array(all_null)

#%%
cmap = 'tab20'

row_labs = var_k[1:]
row_lab_vals = var_v[1:]
col_labs = var_k[:1] 
col_lab_vals = var_v[:1]

# col_labs = var_k[1:]
# col_lab_vals = var_v[1:]
# row_labs = var_k[:1]
# row_lab_vals = var_v[:1]

# contours = True
contours = False

heatmap = True
# heatmap = False

y_ticks = False
# y_ticks = True

share_y_axis = False
# share_y_axis = True

xmin = -0.5*np.sqrt(2)
xmax = 0.5*np.sqrt(2)
ymin = np.sqrt(6)/3 - np.sqrt(1.5)
ymax = np.sqrt(6)/3

if contours:
    xx, yy = np.meshgrid(np.linspace(xmin,xmax,100),np.linspace(ymin,ymax,100))
    foo = (np.stack([xx.flatten(),yy.flatten()]).T@simplx_basis) + [1/3,1/3,1/3]
    support = la.norm(foo,1, axis=-1)<1.001
if heatmap:
    grid = tri.Triangulation(simplx_basis[0,:], simplx_basis[1,:])
    grid = tri.UniformTriRefiner(grid).refine_triangulation(subdiv=6)
    foo = np.stack([grid.x,grid.y])[:,grid.triangles].mean(-1).T@simplx_basis + [1/3,1/3,1/3]
    msk = la.norm(foo,1, axis=-1)>=1.001
    grid.set_mask(msk)

axs = dicplt.hierarchical_labels(row_lab_vals, col_lab_vals,    
                                 row_names=row_labs, col_names=col_labs,
                                 fontsize=13, wmarg=0.3, hmarg=0.1)

    
n_row_lab = np.flip(np.array([1,]+[len(v) for v in row_lab_vals[1:]]))
n_col_lab = np.flip(np.array([1,]+[len(v) for v in col_lab_vals[1:]]))
for k, this_prm in enumerate(itt.product(*var_v)):
    
    col_idx = np.array([np.where(np.isin(var_v[i],this_prm[i]))[0].item() \
                        for i in np.where(np.isin(var_k,col_labs))[0]])
    row_idx = np.array([np.where(np.isin(var_v[i],this_prm[i]))[0].item() \
                        for i in np.where(np.isin(var_k,row_labs))[0]])
    
    if len(col_labs)>=1:
        c = col_idx@n_col_lab
    else:
        c = 0
    if len(row_labs)>=1:
        r = row_idx@n_row_lab
    else:
        r = 0
    
    if contours:
        cols = getattr(cm, cmap)(np.arange(len(these_sess))/len(these_sess))
        for idx, sess in enumerate(these_sess):
            simp = all_probs[k, idx]
            # simp = all_probs[k,idx,:,0,:]
            
            kd_pdf = sts.gaussian_kde(simp.reshape((-1,2)).T)
            zz = np.where(support, kd_pdf(np.stack([xx.flatten(),yy.flatten()])), np.nan)
            
            if heatmap:
                axs[r,c].contour(xx,yy,zz.reshape(100,100,order='A'), 2,
                              colors=['#EC7063','#3498DB'][int(idx>12)], alpha=0.7,
                              linestyles=['solid','dotted'])
            else:
                axs[r,c].contour(xx,yy,zz.reshape(100,100,order='A'), 2,
                                  colors=clr.to_hex(cols[idx]),
                                  linestyles=['solid','dotted'])
            # axs[r,c].contourf(xx,yy,zz.reshape(100,100,order='A'), 2,
            #                  colors=clr.to_hex(cols[idx]),
            #                  alpha=0.7)
    if heatmap:
        # simp = all_probs[k,...,0,:]
        simp = all_probs[k]  
        
        kd_pdf = sts.gaussian_kde(simp.reshape((-1,2)).T)
        
        zz = kd_pdf(np.stack([grid.x,grid.y]))
        
        axs[r,c].tripcolor(grid, zz, rasterized=True, cmap='binary')
    
    axs[r,c].plot([xmin,xmax,0,xmin], [ymin, ymin, ymax, ymin],'k')
    # axs[r,c].plot([xmin,xmax,0,xmin], [ymin, ymin, ymax, ymin],'#A6ACAF')
        
    axs[r,c].set_ylim([ymin*1.1,ymax*1.1])
    axs[r,c].set_xlim([xmin*1.1,xmax*1.1])
    axs[r,c].set_aspect('equal')
    axs[r,c].set_axis_off()
    # dicplt.square_axis(axs[r,c])

#%%
plt.figure()
cmap = 'tab20'

row_labs = var_k[1:]
row_lab_vals = var_v[1:]
col_labs = var_k[:1] 
col_lab_vals = var_v[:1]

# col_labs = var_k[1:]
# col_lab_vals = var_v[1:]
# row_labs = var_k[:1]
# row_lab_vals = var_v[:1]


y_ticks = False
# y_ticks = True

share_y_axis = False
# share_y_axis = True


# show_sessions = True
show_sessions = False

show_all = True
# show_all = False

axs = dicplt.hierarchical_labels(row_lab_vals, col_lab_vals,    
                                 row_names=row_labs, col_names=col_labs,
                                 fontsize=13, wmarg=0.3, hmarg=0.1)

    
n_row_lab = np.flip(np.array([1,]+[len(v) for v in row_lab_vals[1:]]))
n_col_lab = np.flip(np.array([1,]+[len(v) for v in col_lab_vals[1:]]))
for k, this_prm in enumerate(itt.product(*var_v)):
    
    col_idx = np.array([np.where(np.isin(var_v[i],this_prm[i]))[0].item() \
                        for i in np.where(np.isin(var_k,col_labs))[0]])
    row_idx = np.array([np.where(np.isin(var_v[i],this_prm[i]))[0].item() \
                        for i in np.where(np.isin(var_k,row_labs))[0]])
    
    if len(col_labs)>=1:
        c = col_idx@n_col_lab
    else:
        c = 0
    if len(row_labs)>=1:
        r = row_idx@n_row_lab
    else:
        r = 0   

    if show_sessions:
        cols = getattr(cm, cmap)(np.arange(len(these_sess))/len(these_sess))
        for idx, sess in enumerate(these_sess):
            simp = all_null[k, idx]
            
            kd_pdf = sts.gaussian_kde(simp.flatten())
            zz = kd_pdf(np.linspace(0,1,100))
            
            if show_all:
                axs[r,c].plot(np.linspace(0,1,100), kd_pdf(np.linspace(0,1,100)), color=['#EC7063','#3498DB'][int(idx>12)])
            else:
                axs[r,c].plot(np.linspace(0,1,100), kd_pdf(np.linspace(0,1,100)), color=cols[idx])
        
    if show_all:
        simp = all_null[k,:,...]
        axs[r,c].hist(simp.flatten(), bins=50, density=True)
        
    axs[r,c].set_xlabel(r'$p_{\rm{null}}$')
    axs[r,c].set_ylabel('Posterior density')
    axs[r,c].set_ylim([0, np.max(axs[r,c].get_ylim())])
    axs[r,c].set_xlim([0,1])  
    
    # dicplt.square_axis(axs[r,c])

#%% no-swap


# dset_prm = {'session':list(range(13)), # delay period 1
#  					'regions':'all',
#  					'tzf':'CUE2_ON_diode',
#  					'tbeg':-0.5,
#  					'twindow':0.5,
#  					'tstep':0.5,
#  					'num_bins':[5,6],
#  					'do_pca':'before', #'after'
#  					'pca_thrs':0.95,
#  					'min_trials':40,
#  					'shuffle':False,
#  					'impute_nan':True,
#  					'shuffle_probs':False,
#  					'which_block':'retro',
#  					'impute_params':{'weights':'uniform','n_neighbors':5},
#  					'color_weights': hlp.Splines(1) # 'softmax'
#  					}
# model_type = 'noswap_precue'


dset_prm = {'session':list(range(13)), ## delay period 2
 					'regions':'all',
 					'tzf':'WHEEL_ON_diode',
 					'tbeg':-0.5,
 					'twindow':0.5,
 					'tstep':0.5,
 					'num_bins':[5,6],
 					'do_pca':'before', #'after'
 					'pca_thrs':0.95,
 					'min_trials':40,
 					'shuffle':False,
 					'impute_nan':True,
 					'shuffle_probs':False,
 					'which_block':['pro','retro'],
 					'impute_params':{'weights':'uniform','n_neighbors':5},
 					'color_weights': hlp.Splines(1) # 'softmax'
 					}

model_type = 'noswap_delay2'


## funky way of iterating over all the parameters in the dictionary
variable_prms = {k:v for k,v in dset_prm.items() if type(v) is list and k!='session'}
fixed_prms = {k:v for k,v in dset_prm.items() if type(v) is not list and k!='session'}

these_sess = dset_prm['session']
var_k, var_v = zip(*variable_prms.items())

all_probs = []
for vals in (itt.product(*var_v)):
    this_dset = dict(zip(var_k, vals), **fixed_prms)
    this_dset['tend'] = this_dset['tbeg'] + this_dset['twindow']

    sess_probs = [[] for _ in these_sess]
    for idx, which_sess in (enumerate(these_sess)):
        this_dset['session'] = which_sess
        dset_info = {**this_dset}
        folds = hlp.folder_hierarchy(dset_info) 
        
        with open(SAVE_DIR+folds+'/arviz_fit_%s_model.pkl'%model_type, 'rb') as f:
            az_fit = pkl.load(f)
        # with open(SAVE_DIR+folds+'/arviz_fit_inter_tf_precue_model.pkl', 'rb') as f:
        #     az_fit = pkl.load(f)
        # with open(SAVE_DIR+folds+'/arviz_fit_hybrid_error_precue_model.pkl', 'rb') as f:
        #     az_fit = pkl.load(f)
        
        # logits = az_fit.posterior['logits'].to_numpy()
        # sess_probs[idx] = np.exp(logits)/(1+np.exp(logits))
        sess_probs[idx] = az_fit.posterior['p_err'].to_numpy()[...,1]
        
    all_probs.append(sess_probs)

all_probs = np.array(all_probs)

#%%

cmap = 'tab20'

row_labs = var_k[1:]
row_lab_vals = var_v[1:]
col_labs = var_k[:1] 
col_lab_vals = var_v[:1]

# col_labs = var_k[1:]
# col_lab_vals = var_v[1:]
# row_labs = var_k[:1]
# row_lab_vals = var_v[:1]


y_ticks = False
# y_ticks = True

share_y_axis = False
# share_y_axis = True


# show_sessions = True
show_sessions = False

show_all = True
# show_all = False

axs = dicplt.hierarchical_labels(row_lab_vals, col_lab_vals,    
                                 row_names=row_labs, col_names=col_labs,
                                 fontsize=13, wmarg=0.3, hmarg=0.1)

    
n_row_lab = np.flip(np.array([1,]+[len(v) for v in row_lab_vals[1:]]))
n_col_lab = np.flip(np.array([1,]+[len(v) for v in col_lab_vals[1:]]))
for k, this_prm in enumerate(itt.product(*var_v)):
    
    col_idx = np.array([np.where(np.isin(var_v[i],this_prm[i]))[0].item() \
                        for i in np.where(np.isin(var_k,col_labs))[0]])
    row_idx = np.array([np.where(np.isin(var_v[i],this_prm[i]))[0].item() \
                        for i in np.where(np.isin(var_k,row_labs))[0]])
    
    if len(col_labs)>=1:
        c = col_idx@n_col_lab
    else:
        c = 0
    if len(row_labs)>=1:
        r = row_idx@n_row_lab
    else:
        r = 0   

    if show_sessions:
        cols = getattr(cm, cmap)(np.arange(len(these_sess))/len(these_sess))
        for idx, sess in enumerate(these_sess):
            simp = all_probs[k, idx]
            
            kd_pdf = sts.gaussian_kde(simp.flatten())
            zz = kd_pdf(np.linspace(0,1,100))
            
            if show_all:
                axs[r,c].plot(np.linspace(0,1,100), kd_pdf(np.linspace(0,1,100)), color=['#EC7063','#3498DB'][int(idx>12)])
            else:
                axs[r,c].plot(np.linspace(0,1,100), kd_pdf(np.linspace(0,1,100)), color=cols[idx])
        
    if show_all:
        simp = all_probs[k,:,...]
        axs[r,c].hist(simp.flatten(), bins=50, density=True)
        
    axs[r,c].set_xlabel(r'$p_{\rm{null}}$')
    axs[r,c].set_ylabel('Posterior density')
    axs[r,c].set_ylim([0, np.max(axs[r,c].get_ylim())])
    axs[r,c].set_xlim([0,1])  
    
    # dicplt.square_axis(axs[r,c])

