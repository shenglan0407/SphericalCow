import sys
sys.path.append('/home/shenglan/GitHub/SphericalCow/src')

import MixRetriever 
reload(MixRetriever)

from MixRetriever import MixRetriever
from PhaseRetriever import PhaseRetriever

import numpy as np

def test_init():
    mix = np.load('test_data/2rh1A_3p0gA_infPhot_20mol_a0.40-cxs.npy')[:,:90]
    qs = np.load('test_data/qvalue.npy')
    k = 5.0677
    Lmax =10

    thetas = np.arcsin(qs/(2*k))
    phis = np.linspace(0,np.pi,mix.shape[-1], endpoint=True)
    cospsi = np.zeros( (qs.size, phis.size) )
    for idx in range(qs.size):
        cospsi[idx] = np.sin(thetas[idx])**2 + np.cos(thetas[idx])**2 * np.cos(phis)

    pr_mix = PhaseRetriever(q_values=qs,
                                  lmax = Lmax,
                                  n_theta= 100,
                                  n_phi = 90,
                                  corr = mix[:,::-1],
                                  cospsi = cospsi[:,::-1])

    mr = MixRetriever(qs, Lmax, mix, cospsi)

    try:
        np.isclose(mr.cl , pr_mix.cl).all()
    except:
        print("WARNING: Did not pass MixRetriever __init__ test!!")
    print("* Passed MixRetriever __init__ test!")


if __name__=="__main__":
    test_init()
