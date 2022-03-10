import socket
import os
import sys

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.io as sio
from sklearn import svm, manifold, linear_model
from sklearn.model_selection import cross_val_score as cv_score
import sklearn.kernel_approximation as kaprx


class Task(object):
    def __init__(self, num_cols, T_inp1, T_inp2, T_resp, T_tot, go_cue=False):
        
        self.num_col = num_cols
        
        self.T_inp1 = T_inp1
        self.T_inp2 = T_inp2
        self.T_resp = T_resp
        self.T_tot = T_tot
        
        self.go_cue = go_cue
        
    def generate_data(self, n_seq, *seq_args, **seq_kwargs):
        
        upcol, downcol, cue = self.generate_colors(n_seq)
        
        inps, outs = self.generate_sequences(upcol, downcol, cue, *seq_args, **seq_kwargs)
        
        return inps, outs, upcol, downcol, cue
        
    
    def generate_colors(self, n_seq):
        
        upcol = np.random.choice(np.linspace(0,2*np.pi,N_cols), n_seq)
        downcol = np.random.choice(np.linspace(0,2*np.pi,N_cols), n_seq)
            
        cue = np.random.choice([-1.,1.], n_seq) 
        
        return upcol, downcol, cue
        
    def generate_sequences(self, upcol, downcol, cue, jitter=3, inp_noise=0.0, dyn_noise=0.0,
                           new_T=None, retro_only=False, pro_only=False):
        
        T_inp1 = self.T_inp1
        T_inp2 = self.T_inp2
        T_resp = self.T_resp
        if new_T is None:
            T = self.T_tot
        else:
            T = new_T
        n_seq = len(upcol)
        
        col_inp = np.stack([np.cos(upcol), np.sin(upcol), np.cos(downcol), np.sin(downcol)]).T + np.random.randn(n_seq,4)*inp_noise
        
        cuecol = np.where(cue>0, upcol, downcol)
        uncuecol = np.where(cue>0, downcol, upcol)
        
        cue += np.random.randn(n_seq)*inp_noise
        
        inps = np.zeros((n_seq, T, 5+1*self.go_cue))
        
        if jitter>0:
            t_stim1 = np.random.choice(range(T_inp1 - jitter, T_inp1 + jitter), ndat)
            t_stim2 = np.random.choice(range(T_inp2 - jitter, T_inp2 + jitter), ndat)
            t_targ = np.random.choice(range(T_resp - jitter, T_resp + jitter), ndat)
        else:
            t_stim1 = np.ones(n_seq, dtype=int)*(T_inp1)
            t_stim2 = np.ones(n_seq, dtype=int)*(T_inp2)
            t_targ = np.ones(n_seq, dtype=int)*(T_resp)
        
        if retro_only:
            inps[np.arange(n_seq),t_stim1,:4] = col_inp # retro
            inps[np.arange(n_seq),t_stim2, 4] = cue
        elif pro_only:
            inps[np.arange(n_seq),t_stim1,4] = cue # pro
            inps[np.arange(n_seq),t_stim2, :4] = col_inp
        else:
            inps[np.arange(n_seq//2),t_stim1[:n_seq//2],:4] = col_inp[:n_seq//2,:] # retro
            inps[np.arange(n_seq//2),t_stim2[:n_seq//2], 4] = cue[:n_seq//2]
            
            inps[np.arange(n_seq//2, n_seq),t_stim1[n_seq//2:],4] = cue[n_seq//2:] # pro
            inps[np.arange(n_seq//2, n_seq),t_stim2[n_seq//2:], :4] = col_inp[n_seq//2:,:]
        
        
        if self.go_cue:
            inps[np.arange(n_seq),t_targ,5] = 1
        
        # inps = np.zeros((ndat,T,1))
        # inps[np.arange(ndat),t_stim, 0] = cue
        
        outs = np.concatenate([np.stack([np.cos(cuecol), np.sin(cuecol), np.cos(uncuecol), np.sin(uncuecol)]), cue[None,:]], axis=0)
        # outs = np.stack([np.cos(cuecol), np.sin(cuecol), np.cos(uncuecol), np.sin(uncuecol)])
        # outs = np.stack([np.cos(cuecol), np.sin(cuecol)])

        outputs = np.zeros((T, n_seq, outs.shape[0]))
        outputs[t_targ,np.arange(n_seq),:] = outs.T
        outputs = np.cumsum(outputs, axis=0)

        return inps, outputs


def convexify(cols, bins):
    '''
    cols should be given between 0 and 2 pi, bins also
    '''
    
    dc = 2*np.pi/(len(bins))
    
    # get the nearest bin
    diffs = np.exp(1j*bins)[:,None]/np.exp(1j*cols)[None,:]
    distances = np.arctan2(diffs.imag,diffs.real)
    dist_near = np.abs(distances).min(0)
    nearest = np.abs(distances).argmin(0)
    # see if the color is to the "left" or "right" of that bin
    sec_near = np.sign(distances[nearest,np.arange(len(cols))]+1e-8).astype(int) # add epsilon to handle 0
    # fill in the convex array
    alpha = np.zeros((len(bins),len(cols)))
    alpha[nearest, np.arange(len(cols))] = (dc-dist_near)/dc
    alpha[np.mod(nearest-sec_near,len(bins)), np.arange(len(cols))] = 1 - (dc-dist_near)/dc
    
    return alpha

def softmax_cols(cols, bins):
    '''
    cols should be given between 0 and 2 pi, bins also
    '''
    
    num_bins = len(bins)
    dc = 2*np.pi/num_bins
    
    # get the nearest bin
    diffs = np.exp(1j*bins)[:,None]/np.exp(1j*cols)[None,:]
    distances = np.arctan2(diffs.imag,diffs.real)
    alpha = np.exp(np.abs(distances))/np.exp(np.abs(distances)).sum(0)
    
    return alpha


def box_conv(X, len_filt):
    '''
    Convolves X with a square filter, for all possible padding offsets
    '''
    T = X.shape[1]
    N = X.shape[0]
    
    f = np.eye(T+len_filt,T)
    f[np.arange(T)+len_filt,np.arange(T)] = -1
    filt = np.cumsum(f,0)
    
    x_pad = np.stack([np.concatenate([np.zeros((N,len_filt-i)), X, np.zeros((N,i))],axis=1) for i in range(len_filt+1)])
    filted = x_pad@filt
    
    return filted

def folder_hierarchy(dset_info):
    """
    dset info should have at least the following fields:
        session, tzf, tbeg, tend, twindow, tstep, num_bins
    """
    FOLDS = ('/{num_bins}_colors/'
        'sess_{session}/{tzf}/'
        '{tbeg}-{tend}-{twindow}_{tstep}/'
        'pca_{pca_thrs}_{do_pca}/'
        'impute_{impute_nan}/'
        '{color_weights}_knots/'
        '{regions}/'
        '{which_block}/')
    if dset_info['shuffle_probs']:
        FOLDS += 'shuffled/'
    
    return FOLDS.format(**dset_info)


# def file_extension():
# 	return 0