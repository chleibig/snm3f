from __future__ import division, print_function
import numpy as np

EPS = np.finfo(np.float64).eps


def sNM3F(R, P, L, max_iter=10000, tolerance=1e-5):
    """sample-based non-negative matrix tri-factorization

    Parameters
    ==========

    R : samples-by-time-by-space data array, (S, T, N)
    P : number of temporal modules
    L : number of spatial modules
    max_iter: maximum number of iterations

    Returns
    =======

    Btem: (T, P)
    H: (S, P, L)
    Bspa: (L, N)

    """

    S, T, N = R.shape

    # 1)
    Btem = np.random.random(size=(T, P))
    H = np.random.random(size=(S, P, L))
    Bspa = np.random.random(size=(L, N))

    res_old = np.Inf
    res = reconstruction_error(R, Btem, H, Bspa)

    for i in range(max_iter):
        print('Iter: {}, reconstruction error diff: {}'.format(i,
                                                               res_old - res))

        # 2)
        G = np.transpose(Btem.dot(H), axes=(1, 0, 2))
        assert G.shape == (S, T, L)
        numerator = np.einsum('ijk,ijl->kl', G, R)
        denominator = np.einsum('kji,ijl->kl', G.T, G).dot(Bspa)
        Bspa = np.multiply(Bspa, np.divide(numerator, denominator + EPS))
        assert Bspa.shape == (L, N)

        # 3)
        V = np.einsum('ijk,kl->ijl', H, Bspa)
        assert V.shape == (S, P, N)
        numerator = np.einsum('ikj,ilj->kl', R, V)
        denominator = Btem.dot(np.einsum('ikj,jli->kl', V.T, V))
        Btem = np.multiply(Btem, np.divide(numerator, denominator + EPS))
        assert Btem.shape == (T, P)

        # 4)
        numerator = np.einsum('jk,ikm->ijm', Btem.T, R).dot(Bspa.T)
        denominator = np.transpose(
            Btem.T.dot(Btem).dot(H).dot(Bspa).dot(Bspa.T), axes=(1, 0, 2))
        H = np.multiply(H, np.divide(numerator, denominator + EPS))
        assert H.shape == (S, P, L)

        # 5)
        res_old = res
        res = reconstruction_error(R, Btem, H, Bspa)
        if (res_old - res) < tolerance:
            print('Converged to desired tolerance, stopping.')
            break
        if (i + 1) == max_iter:
            print('Maximum number of iterations reached, stopping.')
            break

    return normalize_and_rescale(Btem, H, Bspa)


def reconstruction_error(R, Btem, H, Bspa):
    reconstruction = np.transpose(Btem.dot(H).dot(Bspa), axes=(1, 0, 2))
    return np.linalg.norm(R - reconstruction, ord='fro', axis=(1, 2)).sum()


def normalize_and_rescale(Btem, H, Bspa):
    print('No normalization applied.')
    return Btem, H, Bspa
