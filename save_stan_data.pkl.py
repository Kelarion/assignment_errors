import socket
import os
import sys

if socket.gethostname() == 'kelarion':
    CODE_DIR = '/home/kelarion/github/'
    SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/assignment_errors/'
    LOAD_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/assignment_errors/server_cache/'
    repler_DIR = 'repler/src'
else:    
    # CODE_DIR = '/rigel/home/ma3811/repler/'
    # SAVE_DIR = '/rigel/theory/users/ma3811/'  
    CODE_DIR = '/burg/home/ma3811/'
    SAVE_DIR = '/burg/theory/users/ma3811/assignment_errors/'
    LOAD_DIR = SAVE_DIR
    repler_DIR = 'repler/'
    openmind = False

import pickle as pkl

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.io as sio
from sklearn import svm, manifold, linear_model
from sklearn.model_selection import cross_val_score as cv_score
import sklearn.kernel_approximation as kaprx
import pystan as ps
import arviz as az

sys.path.append(CODE_DIR+'assignment_errors/')
sys.path.append(CODE_DIR+'assignment_errors/jeffcode/')
import general.data_io as gio
import general.utility as u
import swap_errors.auxiliary as swa
import swap_errors.analysis as swan
# import swap_errors.visualization as swv
import general.neural_analysis as na
import general.plotting as gpl
import general.stan_utility as su

import helpers as hlp

sys.path.append(CODE_DIR+repler_DIR)
import util

####  Load dataset and parameters  ######
##########################################################

# get the indices
allargs = sys.argv
idx = int(allargs[1])
ndat = int(allargs[2])
dset_idx = int(allargs[3])

data_idx = int(np.mod(idx,ndat))
param_idx = idx//ndat

data_dict = pkl.load(open(LOAD_DIR+'dataset_%d_%d.pkl'%(dset_idx, data_idx), 'rb'))
fit_params = pkl.load(open(LOAD_DIR+'params_%d_%d.pkl'%(dset_idx, param_idx), 'rb'))

print('Loaded data!')

pkl.dump(data_dict['stan_data'], open(SAVE_DIR+folds+'/stan_data.pkl', 'wb'))

print('Done !!!!!!!!!\n\n')
