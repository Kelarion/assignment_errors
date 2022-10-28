CODE_DIR = '/home/kelarion/github/'
SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/assignment_errors/'

REMOTE_SYNC_SERVER = 'ma3811@motion.rcs.columbia.edu' #must have ssh keys set up
REMOTE_CODE = '/burg/home/ma3811/assignment_errors/'
REMOTE_RESULTS = '/burg/theory/users/ma3811/assignment_errors/'
# REMOTE_SYNC_SERVER = 'kelarion@kelarion' #must have ssh keys set up
# REMOTE_CODE = '/home/kelarion/github/'
# REMOTE_RESULTS = '/mnt/c/Users/mmall/Documents/uni/columbia/assignment_errors/'

import socket
import os
import sys
import pickle as pkl
import subprocess

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.io as sio
from sklearn import svm, manifold, linear_model
from sklearn.model_selection import cross_val_score as cv_score
import sklearn.kernel_approximation as kaprx
from sklearn.impute import KNNImputer as knni
from tqdm import tqdm
import itertools as itt
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

import helpers

sys.path.append(CODE_DIR+'repler/src/')
import util


### Set experiment parameters
##############################
regions = {'frontal':['pfc','fef'],
		   'posterior':['v4pit','tpot','7ab'],
		   'decision':['pfc','fef','7ab'],
		   'sensory':['v4pit','tpot'],
		   'all': ['7ab', 'fef', 'motor', 'pfc', 'tpot', 'v4pit']}

these_dsets = []

# these_dsets.append({'these_sess':list(range(13)),
# 					'regions':['all'],
# 					'tzf': 'WHEEL_ON_diode',
# 					'tbeg':-0.5,
# 					'twindow':0.5,
# 					'tstep':0.5,
# 					'num_bins':6,
# 					'do_pca':'before', #'after'
# 					'pca_thrs':0.95,
# 					'min_trials':40,
# 					'shuffle':False,
# 					'impute_nan':True,
# 					'shuffle_probs':False,
# 					'impute_params':{'weights':'uniform','n_neighbors':5},
# 					'color_weights':'interpolated' # 'softmax'
# 					})

# these_dsets.append({'these_sess':[3,5],
# 					'tzf':'CUE2_ON_diode',
# 					'tbeg':0,
# 					'twindow':0.5,
# 					'tstep':0.5,
# 					'num_bins':[4,5,6,7,8],
# 					'do_pca':'before', #'after'
# 					'pca_thrs':0.95,
# 					'min_trials':40,
# 					'shuffle':False,
# 					'impute_nan':True,
# 					'impute_params':{'weights':'uniform','n_neighbors':5},
# 					'color_weights':'interpolated' # 'softmax'
# 					})

# these_dsets.append({'these_sess':list(range(23)), # delay period 1
# 					'regions':['all'],
# 					'tzf':'CUE2_ON_diode',
# 					'tbeg':-0.5,
# 					'twindow':0.5,
# 					'tstep':0.5,
# 					'num_bins':[5,6],
# 					'do_pca':'before', #'after'
# 					'pca_thrs':0.95,
# 					'min_trials':40,
# 					'shuffle':False,
# 					'impute_nan':True,
# 					'shuffle_probs':False,
# 					'which_block':'retro',
# 					'impute_params':{'weights':'uniform','n_neighbors':5},
# 					'color_weights': [helpers.Splines(1), helpers.Splines(2)] # 'softmax'
# 					})

these_dsets.append({'these_sess':list(range(23)), ## delay period 2
					'regions':['all'],
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
					'which_block':['pro','retro','joint'],
					'impute_params':{'weights':'uniform','n_neighbors':5},
					'color_weights': [helpers.Splines(1), helpers.Splines(2)] # 'softmax'
					})

these_models = []
# these_models.append(['null_hierarchical','spatial_error_hierarchical','cue_error_hierarchical',
# 				'hybrid_error_hierarchical', 'super_hybrid_error_hierarchical'])
# these_models.append(['null_precue','spatial_error_precue','hybrid_error_precue'])
# these_models.append(['super_hybrid_error_hierarchical','spatial_error_hierarchical','cue_error_hierarchical'])
# these_models.append(['hybrid_error_precue'])
these_models.append(['ushh_inter', 'ushh_t_inter'])

##### Make parameter dicts
##########################################

hmc_args = {'iter':1000, 
			'chains':4}

for i_dst, dset_models in enumerate(these_models):

	####### Run job array
	###########################################
	# n_sess = len(these_dsets[i_dst]['these_sess'])*len(list(itt.product(*var_v)))
	n_sess = np.prod([len(v) for k,v in these_dsets[i_dst].items() if type(v) is list])
	n_mod = len(dset_models)

	print('\nSending %d jobs to server ...\n'%(n_sess*n_mod))

	# update file to have correct array indices
	tmplt_file = open(CODE_DIR+'assignment_errors/job_save_template.sh','r')
	with open(SAVE_DIR+'server_cache/job_save_%d.sh'%i_dst,'w') as script_file:
		sbatch_text = tmplt_file.read().format(n_tot=n_sess*n_mod - 1, n_dat=n_sess, dset_idx=i_dst, file_dir=REMOTE_CODE)
		script_file.write(sbatch_text)
	tmplt_file.close()

	## run job
	if 'columbia' in REMOTE_SYNC_SERVER:
		cmd = "ssh ma3811@ginsburg.rcs.columbia.edu 'sbatch -s' < {}".format(SAVE_DIR+'server_cache/job_save_%d.sh'%i_dst)
		subprocess.call(cmd, shell=True)
