CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation as anime
from matplotlib import colors as mpc
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations, combinations
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
from sklearn import gaussian_process as gp
import sklearn.model_selection as skms
import sklearn.linear_model as sklm
import scipy.stats as sts
import scipy.linalg as la

# import umap
from cycler import cycler

# my code
import students
import assistants as ta
import experiments as exp
import util
import tasks
import plotting as tplt
import anime as ani

#%%

def vmr_cos_grad(w, x, theta, kap=2):
    x_nrm = x/la.norm(x, axis=0, keepdims=True)
    return kap*(np.cos(theta) - np.sin(theta)*(w@x_nrm)/np.sqrt(np.abs(1-(w@x_nrm)**2)))*x_nrm

def vmr_sin_grad(v, x, theta, kap=2):
    x_nrm = x/la.norm(x, axis=0, keepdims=True)
    return kap*(np.sin(theta) - np.cos(theta)*(v@x_nrm)/np.sqrt(np.abs(1-(v@x_nrm)**2)))*x_nrm

def vmr_logli(w, v , x, theta, kap=2):
    x_nrm = x/la.norm(x, axis=0, keepdims=True)
    return kap*np.cos(theta)*(w@x_nrm) + kap*np.sin(theta)*(v@x_nrm) - np.log(2*np.pi*spc.i0(kap))

def expmap(w, dw):
    dw_nrm = la.norm(dw)
    return np.cos(dw_nrm)*w + np.sin(dw_nrm)*(dw/dw_nrm)


basis = la.qr( np.random.randn(100,100))[0]
basis[:,1] += 0.5*basis[:,0]
basis[:,1] /= la.norm(basis[:,1])

w = np.random.randn(100)
w /= la.norm(w)

v = np.random.randn(100)
v /= la.norm(v)

x = basis[:,:2]@np.stack([np.sin(cuecol), np.cos(cuecol)]) + np.random.randn(100, 2000)*0.1
x -= x.mean(1, keepdims=True)
x_nrm = x/la.norm(x, axis=0, keepdims=True)

ls = []
ovlp = []
for _ in tqdm(range(100)):
    ls.append(vmr_logli(w, v, x, cuecol).mean())
    ovlp.append([basis[:,0]@v, basis[:,1]@w])
    
    w_grad = vmr_cos_grad(w, x, cuecol, kap=5)
    w = expmap(w, 0.1*((np.eye(100) - w[:,None]*w[None,:])@w_grad).mean(1))
    
    v_grad = vmr_sin_grad(v, x, cuecol, kap=5)
    v = expmap(v, 0.1*((np.eye(100) - v[:,None]*v[None,:])@v_grad).mean(1))


#%%

def stief_update(W, x, theta, kap=1, alpha=1e-2, n_cay=100):
    
    Pr = W@W.T 
    x_nrm = x/la.norm(Pr@x, axis=0, keepdims=True)
    
    th_pred = W.T@x_nrm
    
    # dLdW = kap*np.stack([np.cos(theta), np.sin(theta)])[None,:,:]*x_nrm[:,None,:]
    
    # P1 = np.einsum('ij...,kj...->ik...',dLdW, W) 
    # P2 = np.einsum('ij...,kj...->ik...', W, W)
    # P = P1 - 0.5*np.einsum('ij...,kj...->ik...', P2, P1)
    # P = np.mean(P - P.transpose((1,0,2)),-1)
    
    th = np.stack([np.cos(theta), np.sin(theta)])
    D = np.repeat(th_pred[:,None,:], 2, axis=1)*np.repeat(th_pred[None,:,:], 2, axis=0)*np.array([[1,-1],[-1,1]])[...,None]
    
    dLdW = np.mean(kap*np.einsum('ij...,j...->i...',D,th)[None,:,:]*x_nrm[:,None,:], -1)
    
    # dLdW = np.stack([np.cos(theta)*np.sin(th_pred)*2 - np.sin())*x_nrm
    
    # dLdW = np.mean(kap*np.stack([np.cos(theta), np.sin(theta)])[None,:,:]*x_nrm[:,None,:], -1)
    P1 = dLdW@W.T
    P = P1 - 0.5*(W@W.T)@P1
    P -= P.T
    
    proj_grad =  P@W
    
    Y = W + alpha*proj_grad
    for _ in range(n_cay):
        Y = W + (alpha/2)*(P@(W+Y))
    
    return Y


basis = la.qr( np.random.randn(100,100))[0]
basis[:,1] += 0.5*basis[:,0]
basis[:,1] /= la.norm(basis[:,1])

W = la.qr( np.random.randn(100,100))[0][:,:2]

x = basis[:,:2]@np.stack([np.sin(cuecol), np.cos(cuecol)]) + np.random.randn(100, 2000)*0.1
x -= x.mean(1, keepdims=True)

targ = np.stack([np.cos(cuecol), np.sin(cuecol)])

ls = []
ovlp = []
for _ in tqdm(range(200)):
    
    Pr = W@W.T
    x_nrm = x/la.norm(Pr@x, axis=0, keepdims=True)
    pred = W.T@x_nrm
    
    ls.append(np.sqrt(1-np.sum(targ*pred, 0)**2).mean())
    
    W = stief_update(W, x, cuecol, kap=5, alpha=1e-2, n_cay=100)

#%%

N = 50
noise_sigma = 0.1
# angle = np.pi/8
angle = 0
gp_sigma = 0.5
lin_mag = 1.0

num_test = 5

von_mis = True
# von_mis = False

# if von_mis:
vm_clf = ta.VonMisesRegression(lr=1e-2,max_iter=1000, max_cayley_iter=100, tol=1e-4)
# else:
sv_clf = ta.LinearDecoder(N, 2, svm.LinearSVR)
gp_clf = gp.GaussianProcessRegressor(gp.kernels.RBF(2*gp_sigma))


col_trn = np.random.choice(np.linspace(-np.pi, np.pi, 100), 2000)
col_tst = np.random.choice(np.linspace(-np.pi, np.pi, 100), 2000)

targ_trn = np.stack([np.cos(col_trn), np.sin(col_trn)])
targ_tst = np.stack([np.cos(col_tst), np.sin(col_tst)])

basis = la.qr( np.random.randn(N,N))[0]
basis[:,1] += np.tan(angle)*basis[:,0]
basis[:,1] /= la.norm(basis[:,1])

coords = gp.GaussianProcessRegressor(gp.kernels.RBF(gp_sigma))

vm_perf = []
sv_perf = []
vm_ovlp = []
sv_ovlp = []
gp_perf = []
for gp_mag in tqdm(np.linspace(0,0.2,50)):
    
    vm_prf = []
    vm_ov = []
    sv_prf = []
    sv_ov = []
    gp_prf = []
    for _ in range(num_test):
        
        ys = coords.sample_y(targ_trn.T, n_samples=N).T
        ys_tst = coords.sample_y(targ_tst.T, n_samples=N).T
    
        x = lin_mag*basis[:,:2]@targ_trn + ys*gp_mag + np.random.randn(N, 2000)*noise_sigma
        
        x_tst = lin_mag*basis[:,:2]@targ_tst + ys_tst*gp_mag + np.random.randn(N, 2000)*noise_sigma
        
        # if von_mis:
        vm_clf.fit(x.T, col_trn)
        
        vm_prf.append(vm_clf.score(x_tst.T, col_tst))
        
        vm_ov.append(la.norm(((vm_clf.coef_@vm_clf.coef_.T)@basis[:,:2]),axis=0))
        
    # else:
        sv_clf.fit(x.T, targ_trn.T)
                
        pred = sv_clf.project(x_tst.T)
        pred /= la.norm(pred, axis=0)
        
        sv_prf.append((np.sum(pred*targ_tst, 0)**2).mean())
        
        W = np.squeeze(sv_clf.coefs).T/la.norm(np.squeeze(sv_clf.coefs).T, axis=0)
        sv_ov.append(la.norm(((W@W.T)@basis[:,:2]),axis=0))
        
        
        gp_clf.fit(x.T, targ_trn.T)
        pred = gp_clf.predict(x_tst.T).T
        pred /= la.norm(pred, axis=0)
        gp_prf.append((np.sum(pred*targ_tst, 0)**2).mean())
            
    vm_perf.append(np.mean(vm_prf, 0))
    vm_ovlp.append(np.mean(vm_ov, 0))
    sv_perf.append(np.mean(sv_prf, 0))
    sv_ovlp.append(np.mean(sv_ov, 0))
    gp_perf.append(np.mean(gp_prf))
    
    
#%%

N = 50
noise_sigma = 0.2
# angle = np.pi/8
angle = 0
rf_width = N

num_test = 5

von_mis = True
# von_mis = False

# if von_mis:
vm_clf = ta.VonMisesRegression(lr=1e-2,max_iter=1000, max_cayley_iter=100, tol=1e-4)
# else:
sv_clf = ta.LinearDecoder(N, 2, svm.LinearSVR)
gp_clf = gp.GaussianProcessRegressor(gp.kernels.RBF(0.5))

nonlin_sig = util.RFColors(N, rf_width)

col_trn = np.random.choice(np.linspace(-np.pi, np.pi, 100), 2000)
col_tst = np.random.choice(np.linspace(-np.pi, np.pi, 100), 2000)

targ_trn = np.stack([np.cos(col_trn), np.sin(col_trn)])
targ_tst = np.stack([np.cos(col_tst), np.sin(col_tst)])

ys = nonlin_sig(col_trn).T
ys_tst = nonlin_sig(col_tst).T

vm_perf = []
sv_perf = []
vm_ovlp = []
sv_ovlp = []
gp_perf = []
for rf_mag in tqdm(np.linspace(0.1,1.0,10)):
    
    vm_prf_ = []
    vm_ov_ = []
    sv_prf_ = []
    sv_ov_ = []
    gp_prf_ = []
    for lin_mag in np.linspace(0.1,1.0,10):
    
        vm_prf = []
        vm_ov = []
        sv_prf = []
        sv_ov = []
        gp_prf = []
        for _ in range(num_test):
            
            basis = la.qr( np.random.randn(N,N))[0]
            basis[:,1] += np.tan(angle)*basis[:,0]
            basis[:,1] /= la.norm(basis[:,1])
            
            x = lin_mag*basis[:,:2]@targ_trn + ys*rf_mag + np.random.randn(N, 2000)*noise_sigma
            
            x_tst = lin_mag*basis[:,:2]@targ_tst + ys_tst*rf_mag + np.random.randn(N, 2000)*noise_sigma
            
            # if von_mis:
            vm_clf.fit(x.T, col_trn)
            
            vm_prf.append(vm_clf.score(x_tst.T, col_tst))
            
            vm_ov.append(la.norm(((vm_clf.coef_@vm_clf.coef_.T)@basis[:,:2]),axis=0))
            
        # else:
            sv_clf.fit(x.T, targ_trn.T)
            pred = sv_clf.project(x_tst.T)
            pred /= la.norm(pred, axis=0)
            
            sv_prf.append(np.sum(pred*targ_tst, 0).mean())
            
            W = np.squeeze(sv_clf.coefs).T/la.norm(np.squeeze(sv_clf.coefs).T, axis=0)
            sv_ov.append(la.norm(((W@W.T)@basis[:,:2]),axis=0))
            
            gp_clf.fit(x.T, targ_trn.T)
            pred = gp_clf.predict(x_tst.T).T
            pred /= la.norm(pred, axis=0)
            gp_prf.append(np.sum(pred*targ_tst, 0).mean())
           
        vm_prf_.append(np.mean(vm_prf, 0))
        vm_ov_.append(np.mean(vm_ov, 0))
        sv_prf_.append(np.mean(sv_prf, 0))
        sv_ov_.append(np.mean(sv_ov, 0))
        gp_prf_.append(np.mean(gp_prf))
           
    vm_perf.append(vm_prf_)
    vm_ovlp.append(vm_ov_)
    sv_perf.append(sv_prf_)
    sv_ovlp.append(sv_ov_)
    gp_perf.append(gp_prf_)


