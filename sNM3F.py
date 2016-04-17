from __future__ import division
import numpy as np

# Simulate some test data:
# TODO
S = 10
N = 40
T = 20
R = np.random.random(size=(S, N, T))

def sNM3F(R, P=2, L=2):
    """Space-by-time non-negative matrix factorization

    Parameters
    ==========

    R : samples-by-space-by-time data array
    P : number of temporal modules
    L : number of spatial modules

    """

    S, N, T = R.shape

    Btem = np.random.random(size=(T, P))
    H = np.random.random(size=(S, P, L))
    Bspa = np.random.random(size=(L, N))

    res_old = np.Inf
    res = reconstruction_error(R, Btem, H, Bspa)

    # while (res_old - res) > 0.001:

    # step 2
    G = np.transpose(np.dot(Btem, H), axes=(1, 0, 2))
    assert G.shape == (S, T, L)
    Gmat = np.reshape(G, (S*T, L))
    Rmat = np.reshape(np.transpose(R, axes=(0, 2, 1)), (S*T, N))
    assert Rmat.shape == (S*T, N)
    GtR = np.dot(Gmat.T, Rmat)
    assert GtR.shape == (L, N)
    GtGBspa = np.dot(np.dot(Gmat.T, Gmat), Bspa)
    assert GtGBspa.shape == (L, N)
    # step 2c
    Bspa = np.multiply(Bspa, np.divide(GtR, GtGBspa))

    # step) 3
    V = np.dot(H, Bspa)
    assert V.shape == (S, P, N)
    Vmat = np.reshape(np.transpose(V, axes=(0, 2, 1)), (S*N, P))
    Rmat = np.reshape(R, (S*N, T))
    RprimeV = np.dot(Rmat.T, Vmat)
    assert RprimeV.shape == (T, P)
    BtemVtV = np.dot(Btem, np.dot(Vmat.T, Vmat))
    assert BtemVtV.shape == (T, P)
    # step 3d
    Btem = np.multiply(Btem, np.divide(RprimeV, BtemVtV))
    
    # step 4

    # step 5
    # res_old = res
    # res = reconstruction_error(R, Btem, H, Bspa)
    # TODO: normalization

    return Btem, H, Bspa


def reconstruction_error(R, Btem, H, Bspa):
    pass

sNM3F(R)