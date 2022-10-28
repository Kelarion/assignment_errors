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
import assistants as ta

special_font = mfm.FontProperties(fname='C:/Windows/Fonts/seguiemj.ttf')

#%% delay 2

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
model_type = 'ushh_t_inter'
    
    

## funky way of iterating over all the parameters in the dictionary
variable_prms = {k:v for k,v in dset_prm.items() if type(v) is list and k!='session'}
fixed_prms = {k:v for k,v in dset_prm.items() if type(v) is not list and k!='session'}

these_sess = dset_prm['session']
var_k, var_v = zip(*variable_prms.items())

all_u = []
all_l = []
all_u_d = []
all_l_d = []
for vals in list(itt.product(*var_v)):
    this_dset = dict(zip(var_k, vals), **fixed_prms)
    this_dset['tend'] = this_dset['tbeg'] + this_dset['twindow']

    sess_u = [[] for _ in these_sess]
    sess_l = [[] for _ in these_sess]
    sess_u_d = [[] for _ in these_sess]
    sess_l_d = [[] for _ in these_sess]
    for idx, which_sess in enumerate(these_sess):
        this_dset['session'] = which_sess
        dset_info = {**this_dset}
        folds = hlp.folder_hierarchy(dset_info) 
    
        with open(SAVE_DIR+folds+'/arviz_fit_%s_model.pkl'%model_type, 'rb') as f:
            az_fit = pkl.load(f)
            
        sess_u[idx] = az_fit.posterior['mu_u'].mean(axis=(0,1)).data
        sess_l[idx] = az_fit.posterior['mu_l'].mean(axis=(0,1)).data
        sess_u_d[idx] = az_fit.posterior['mu_d_u'].mean(axis=(0,1)).data
        sess_l_d[idx] = az_fit.posterior['mu_d_l'].mean(axis=(0,1)).data
        
    all_u.append(sess_u)
    all_l.append(sess_l)
    all_l_d.append(sess_l_d)
    all_u_d.append(sess_u_d)

# all_probs = np.array(all_probs)


#%%

# use_iso = True
use_iso = False

# n_iso_comp = 72
# n_iso_neigh = 

n_col_plt = 200

u = np.concatenate(all_u[0],axis=0)
l = np.concatenate(all_l[0],axis=0)
d_u = np.concatenate(all_u_d[0],axis=0)
d_l = np.concatenate(all_l_d[0],axis=0)


these_cols = hlp.convexify(np.linspace(0,2*np.pi,n_col_plt), np.linspace(0,2*np.pi,7)[:-1])

if use_iso:
    iso = manifold.Isomap(n_components=5, n_neighbors=100)
    all_circs = iso.fit_transform(np.concatenate([u@these_cols,l@these_cols,d_l@these_cols,d_u@these_cols],axis=1).T)
else:
    all_circs = np.concatenate([u@these_cols,l@these_cols,d_l@these_cols,d_u@these_cols],axis=1).T


N = all_circs.shape[-1]

clf = svm.LinearSVC()
cued_clf = ta.VonMisesRegression(lr=1e-2)
unc_clf = ta.VonMisesRegression(lr=1e-2)

clf.fit(all_circs, np.repeat(np.mod(np.arange(4),2),n_col_plt)>0)
cued_clf.fit(all_circs[:2*n_col_plt,:], np.tile(np.linspace(0,2*np.pi,n_col_plt),2)) # fit on cued circles

basis1 = np.concatenate([cued_clf.coef_, clf.coef_.T/la.norm(clf.coef_)], axis=1)

# fit uncued color in complementary subspace
circ_comp = all_circs@(np.eye(N) - basis1@basis1.T)
unc_clf.fit(circ_comp[2*n_col_plt:,:], np.tile(np.linspace(0,2*np.pi,n_col_plt),2)) # fit on uncued circles

basis2 = np.concatenate([unc_clf.coef_, clf.coef_.T/la.norm(clf.coef_)], axis=1)

#%%

knots = np.tile(these_cols.argmax(1),2)+ np.repeat(np.arange(2),6)*n_col_plt

ax1 = plt.subplot(1,2,1, projection='3d')
dicplt.scatter3d(all_circs[:2*n_col_plt,:]@basis1, c=cm.hsv(np.tile(np.linspace(0,2*np.pi,n_col_plt),2)/(2*np.pi)), ax=ax1)
dicplt.scatter3d(all_circs[2*n_col_plt:,:]@basis1, c=[(0.5,0.5,0.5)], ax=ax1)
dicplt.scatter3d(all_circs[knots,:]@basis1, 
                 c=cm.hsv(np.tile(np.linspace(0,2*np.pi,7)[:-1],2)/(2*np.pi)), ax=ax1, s=100)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])


ax2 = plt.subplot(1,2,2,  projection='3d')
dicplt.scatter3d(all_circs[2*n_col_plt:,:]@basis2, c=cm.hsv(np.tile(np.linspace(0,2*np.pi,n_col_plt),2)/(2*np.pi)), ax=ax2)
dicplt.scatter3d(all_circs[:2*n_col_plt,:]@basis2, c=[(0.5,0.5,0.5)], ax=ax2)
dicplt.scatter3d(all_circs[knots+2*n_col_plt,:]@basis2, 
                 c=cm.hsv(np.tile(np.linspace(0,2*np.pi,7)[:-1],2)/(2*np.pi)), ax=ax2, s=100)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])

#%% delay 1

dset_prm = {'session':list(range(13)), # delay period 1
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
model_type = 'inter_tf_precue'


## funky way of iterating over all the parameters in the dictionary
variable_prms = {k:v for k,v in dset_prm.items() if type(v) is list and k!='session'}
fixed_prms = {k:v for k,v in dset_prm.items() if type(v) is not list and k!='session'}

these_sess = dset_prm['session']
var_k, var_v = zip(*variable_prms.items())

all_u = []
all_l = []
for vals in list(itt.product(*var_v)):
    this_dset = dict(zip(var_k, vals), **fixed_prms)
    this_dset['tend'] = this_dset['tbeg'] + this_dset['twindow']

    sess_u = [[] for _ in these_sess]
    sess_l = [[] for _ in these_sess]
    for idx, which_sess in enumerate(these_sess):
        this_dset['session'] = which_sess
        dset_info = {**this_dset}
        folds = hlp.folder_hierarchy(dset_info) 
    
        with open(SAVE_DIR+folds+'/arviz_fit_hybrid_error_precue_model.pkl', 'rb') as f:
            az_fit = pkl.load(f)
            
        sess_u[idx] = az_fit.posterior['mu_u'].mean(axis=(0,1)).data
        sess_l[idx] = az_fit.posterior['mu_l'].mean(axis=(0,1)).data
        
    all_u.append(sess_u)
    all_l.append(sess_l)
# all_probs = np.array(all_probs)


#%%

# use_iso = True
use_iso = False

# enforce_ortho = True
enforce_ortho = False

# n_iso_comp = 72
# n_iso_neigh = 

n_col_plt = 200
num_bin = dset_prm['num_bins']

u = np.concatenate(all_u[0],axis=0)
l = np.concatenate(all_l[0],axis=0)

these_cols = hlp.Convexify()(np.linspace(0,2*np.pi,n_col_plt), num_bin)

if use_iso:
    iso = manifold.Isomap(n_components=5, n_neighbors=100)
    all_circs = iso.fit_transform(np.concatenate([u@these_cols,l@these_cols],axis=1).T)
else:
    all_circs = np.concatenate([u@these_cols,l@these_cols],axis=1).T

N = all_circs.shape[-1]

up_clf = ta.VonMisesRegression(lr=1e-2)
down_clf = ta.VonMisesRegression(lr=1e-2)

up_clf.fit(all_circs[:n_col_plt,:], np.linspace(0,2*np.pi,n_col_plt)) # fit on cued circles

basis1 = up_clf.coef_

# fit uncued color in complementary subspace
if enforce_ortho:
    circ_comp = all_circs@(np.eye(N) - basis1@basis1.T)
else:
    circ_comp = all_circs
down_clf.fit(circ_comp[n_col_plt:,:], np.linspace(0,2*np.pi,n_col_plt)) # fit on uncued circles

basis2 = down_clf.coef_

#%%

knots = these_cols.argmax(1)

ax1 = plt.subplot(1,2,1)
plt.scatter(all_circs[:n_col_plt,:]@basis1[:,0],all_circs[:n_col_plt,:]@basis1[:,1], 
            c=cm.hsv(np.linspace(0,2*np.pi,n_col_plt)/(2*np.pi)))
plt.scatter(all_circs[n_col_plt:,:]@basis1[:,0],all_circs[n_col_plt:,:]@basis1[:,1], 
            c=[(0.5,0.5,0.5)])
# plt.scatter(all_circs[knots,:]@basis1[:,0],all_circs[knots,:]@basis1[:,1], 
#             c=cm.hsv(np.linspace(0,2*np.pi,num_bin+1)[:-1]/(2*np.pi)), s=200)
dicplt.square_axis()
plt.axis('off')

ax2 = plt.subplot(1,2,2)
plt.scatter(all_circs[n_col_plt:,:]@basis2[:,0],all_circs[n_col_plt:,:]@basis2[:,1], 
            c=cm.hsv(np.linspace(0,2*np.pi,n_col_plt)/(2*np.pi)))
plt.scatter(all_circs[:n_col_plt,:]@basis2[:,0],all_circs[:n_col_plt,:]@basis2[:,1], 
            c=[(0.5,0.5,0.5)])
# plt.scatter(all_circs[knots+n_col_plt,:]@basis2[:,0],all_circs[knots+n_col_plt,:]@basis2[:,1], 
#             c=cm.hsv(np.linspace(0,2*np.pi,num_bin+1)[:-1]/(2*np.pi)), s=200)
dicplt.square_axis()
plt.axis('off')

