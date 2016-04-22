from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from sNM3F import sNM3F
plt.ion()

S = 10
N = 10
T = 20
R = np.random.random(size=(S, T, N))
P = 2
L = 2

bin_width = 10  # ms
lambda_bin_bg = 2 * bin_width / 1000.
lambda_bin_fg = 300 * bin_width / 1000.
n_true_modules = 4

# True module construction

true_modules = np.ones((n_true_modules, N, T)) * lambda_bin_bg

true_modules[0, :7, 9:13] = lambda_bin_fg
true_modules[1, 4:, 9:13] = lambda_bin_fg
true_modules[2, :7, 11:18] = lambda_bin_fg
true_modules[3, 4:, 11:18] = lambda_bin_fg

# Trial simulation
R = np.zeros((S, T, N))
for trial in range(S):
    active_modules = np.where(
        np.random.binomial(1, p=0.5, size=(n_true_modules,)))
    lambda_map_total = true_modules[active_modules].sum(axis=0)
    for t in range(T):
        for n in range(N):
            R[trial, t, n] = np.random.poisson(lam=lambda_map_total[n, t],
                                               size=bin_width).sum()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(lambda_map_total, interpolation='none')
    plt.title('firing rates')
    plt.subplot(1, 2, 2)
    plt.imshow(R[trial, :, :].T, interpolation='none')
    plt.title('Spike counts')


Btem, H, Bspa = sNM3F(R, P, L, max_iter=10000, tolerance=1e-5)


plt.figure()
for i in range(n_true_modules):
    plt.subplot(2, n_true_modules, i + 1)
    plt.imshow(true_modules[i, :], interpolation='none')
for i in range(P):
    for j in range(L):
        plt.subplot(2, n_true_modules, n_true_modules + i * P + j + 1)
        plt.imshow(np.outer(Btem[:, i], Bspa[j, :]).T, interpolation='none')
