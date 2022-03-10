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

dset_prm = {'session':list(range(13)),
 					'regions':['all'],
 					'tzf': 'WHEEL_ON_diode',
 					'tbeg':-0.5,
 					'twindow':0.5,
 					'tstep':0.5,
 					'num_bins':6,
 					'do_pca':'before', #'after'
 					'pca_thrs':0.95,
 					'min_trials':40,
 					'shuffle':False,
 					'impute_nan':True,
                    'shuffle_probs':False,
                    'which_block':'joint',
 					'impute_params':{'weights':'uniform','n_neighbors':5},
 					'color_weights':'interpolated' # 'softmax'
 					}
    
## funky way of iterating over all the parameters in the dictionary
variable_prms = {k:v for k,v in dset_prm.items() if type(v) is list and k!='session'}
fixed_prms = {k:v for k,v in dset_prm.items() if type(v) is not list and k!='session'}

these_sess = dset_prm['session']
var_k, var_v = zip(*variable_prms.items())

# simplx_basis = np.array([[1,-1,0],[-0.5,-0.5,1]])
simplx_basis = np.array([[1,0,-1],[-0.5,1,-0.5]])
simplx_basis /= la.norm(simplx_basis,axis=1,keepdims=True)

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
    
        with open(SAVE_DIR+folds+'/arviz_fit_ultra_super_hybrid_hierarchical_model.pkl', 'rb') as f:
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


n_col_plt = 200

u = np.concatenate(all_u[0],axis=0)
l = np.concatenate(all_l[0],axis=0)
d_u = np.concatenate(all_u_d[0],axis=0)
d_l = np.concatenate(all_l_d[0],axis=0)


these_cols = helpers.convexify(np.linspace(0,2*np.pi,n_col_plt), np.linspace(0,2*np.pi,7)[:-1])


