"""
This module contains functions for constructing trackers with various
parameter settings that seem to work well in typical situations.

Author: Travis Dick (travis.barry.dick@gmail.com)
"""

from BMICTracker import BMICTracker
from ESMTracker import ESMTracker
from NNTracker import NNTracker
from PFTracker import PFTracker
from DNNTracker import DNNTracker
from DESMTracker import DESMTracker
from DLKTracker import DLKTracker
from CTracker import CTracker
from ThreadedCascadeTracker import ThreadedCascadeTracker
from CascadeTracker import CascadeTracker

def make_nn_bmic(use_scv=False, res=(40,40), nn_iters=10, nn_samples=1000, bmic_iters=5):
    t1 = NNTracker(nn_iters, nn_samples, res[0], res[1], 0.06, 0.04, use_scv)
    t2 = NNTracker(nn_iters, nn_samples, res[0], res[1], 0.03, 0.02, use_scv)
    t3 = NNTracker(nn_iters, nn_samples, res[0], res[1], 0.015, 0.01, use_scv)
    t4 = BMICTracker(bmic_iters, 0.01, res[0], res[1], use_scv)
    return CascadeTracker([t1, t2, t3, t4], use_scv=use_scv)

def make_bmic(use_scv=False, res=(40,40), iters=30):
    return BMICTracker(iters, 0.001, res[0], res[1], use_scv)

def make_esm(use_scv=False, res=(60,60), iters=20):
    return ESMTracker(iters, 0.001, res[0], res[1], use_scv)

def make_pf(use_scv=False, res=(40,40), n_samples=500, MModel=7):
    t1 = PFTracker(n_samples, res[0], res[1], use_scv, MModel)
    return t1

def make_ctrk(use_scv=True, res=(30,30)):
    return CTracker(10, 0.001, res[0], res[1], use_scv)

def make_dnn_bmic(use_scv=False, res=(40,40), nn_iters=10, nn_samples=1100, bmic_iters=5, MModel=1, gnn=False, exp=False):
    t1 = DNNTracker(nn_iters, nn_samples, res[0], res[1], 0.06, 0.04, use_scv, MModel, gnn, exp)
    t2 = DNNTracker(nn_iters, nn_samples, res[0], res[1], 0.03, 0.02, use_scv, MModel, gnn, exp)
    t3 = DNNTracker(nn_iters, nn_samples, res[0], res[1], 0.015, 0.01, use_scv, MModel, gnn, exp)
    t4 = DLKTracker(bmic_iters, 0.001, res[0], res[1], use_scv, MModel)
    return CascadeTracker([t1,t2,t3, t4], set_warp_directly=True, use_scv=use_scv)

def make_desm(use_scv=False, res=(60,60), iters=20, MModel = 1):
    return DESMTracker(iters, 0.001, res[0], res[1], use_scv, MModel)

def make_dlkt(use_scv=False, res=(50,50), iters=20, MModel = 7):
    return DLKTracker(iters, 0.001, res[0], res[1], use_scv, MModel)

