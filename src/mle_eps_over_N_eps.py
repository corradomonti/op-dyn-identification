import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from tqdm.notebook import tqdm

import os
import re
import socket
import subprocess
import sys

# sns.set_style("whitegrid")
# np.set_printoptions(suppress=True, precision=2, threshold=20)
# pd.set_option('display.max_rows', 500)
# sys.path += ['../src/']

sigmoid = lambda x: 1. / (1 + np.exp(-beta * x))
log_sigmoid = lambda x: -np.log1p(np.exp(-beta * x))

def generate_extraction_graph(N, T, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    G = []
    for t in range(T):
        i = np.random.randint(N)
        while True:
            j = np.random.randint(N)
            if i != j: break
        G.append([i, j, t])
    return np.array(G)


def generate_dynamics(N, T, G, mu=0.5, eps=.25, beta=50, x0=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if x0 is None:
        x0 = np.random.uniform(size=N) * 2 - 1
    
    u_v_t_w = []
    
    t = 0
    X = [x0]
    for i, j, t in G:
        if t >= T:
            break
        xt = X[-1]
        xtp1 = xt.copy()
        dist = np.abs(xt[i] - xt[j])
        p = sigmoid(eps-dist)
        extraction = np.random.uniform()
        if extraction<=p:
            xtp1[i] += mu * (xt[j] - xt[i])
            xtp1[j] += mu * (xt[i] - xt[j])
            u_v_t_w.append( (i, j, t, 1) )
        else:
            u_v_t_w.append( (i, j, t, 0) )
        xtp1 = np.clip(xtp1, -1, 1)
    
        X.append(xtp1)
    X = np.vstack(X)
    
    return u_v_t_w, X


def neg_log_likelihood(eps_hat):
    mu_hat = mu
    xt = x0.copy()
    # X = [xt]
    log_likelihood = 0
    for i, j, t, has_interacted_at_t in u_v_t_w:
        dist = np.abs(xt[i] - xt[j])
        if has_interacted_at_t:
            log_sigma = log_sigmoid(eps_hat - dist)  # to avoid overflow
            log_likelihood += log_sigma
            xt[i] += mu_hat * (xt[j] - xt[i])
            xt[j] += mu_hat * (xt[i] - xt[j])
        else:
            log_likelihood -= np.log1p(np.exp(beta*(eps_hat - dist)))
        # X.append(xt.copy())
    return -log_likelihood


# eps = 0.5
mu = 0.3
beta = 15
T = 10000
# N = 50

epsrange = np.linspace(0, 1., 1001)
generation_seed = 1

n_ic = 10
n_trials = 10
eps_range = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) # 
n_range = np.array([3, 5, 10, 100, 500])
eps_star_list = np.zeros((n_ic, len(eps_range), len(n_range), n_trials))


for c, eps in enumerate(eps_range):
    for ceps, N in enumerate(n_range):
        G = generate_extraction_graph(N, T, seed=generation_seed)
        for i in range(n_ic):
            print(c, mu, ceps, eps, i)
            # if i%10==0:
            #     print(i)
            np.random.seed(i)
            x0 = np.random.uniform(size=N) * 2 - 1
            for trial in range(n_trials):
                u_v_t_w, X = generate_dynamics(N, T, G, mu=mu, eps=eps, beta=beta, x0=x0, seed=trial)
                u_v_t_w = np.array(u_v_t_w)
                
                nll_curve = np.vectorize(neg_log_likelihood)(epsrange)
                
                eps_star = np.nan if np.isclose(np.std(nll_curve), 0.) else epsrange[np.argmin(nll_curve)]
                eps_star_list[i, c, ceps, trial] = eps_star
        
        df_tmp = pd.DataFrame(eps_star_list[:, c, ceps, :])
        df_tmp.to_csv(f'../outputs/mle_eps_over_N_eps_{n_trials}_{eps}_{mu}_{beta}_{N}_{T}_{generation_seed}.csv')
