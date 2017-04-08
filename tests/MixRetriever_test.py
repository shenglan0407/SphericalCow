import sys
sys.path.append('/home/shenglan/GitHub/SphericalCow/src')

import MixRetriever 
reload(MixRetriever)

from MixRetriever import MixRetriever
from PhaseRetriever import PhaseRetriever

import numpy as np
import h5py

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

    # pr_mix = PhaseRetriever(q_values=qs,
    #                               lmax = Lmax,
    #                               n_theta= 100,
    #                               n_phi = 90,
    #                               corr = mix[:,::-1],
    #                               cospsi = cospsi[:,::-1])
    # x = mr.norm_leg_coefs(pr_mix.cl.copy())
    # np.save('test_data/norm_2rh1A_3p0gA_mix_cl.npy',x)

    mr = MixRetriever(qs, Lmax, mix, cospsi)
    x = np.load('test_data/norm_2rh1A_3p0gA_mix_cl.npy')
    
    try:
        assert(np.isclose(mr.cl ,x).all())
    except:
        print("WARNING: Did not pass MixRetriever __init__ test!!")
        sys.exit()

    print("* Passed MixRetriever __init__ test!")
    
    return mr

def test_add_known_component( mr ):

    corr1 = np.load('test_data/3p0gA-downsamp-infPhot-target-cxs.npy')[:,:90]
    name = '3p0g'
    qs = np.load('test_data/qvalue.npy')

    k = 5.0677

    thetas = np.arcsin(qs/(2*k))
    phis = np.linspace(0,np.pi,corr1.shape[-1], endpoint=True)
    cospsi = np.zeros( (qs.size, phis.size) )
    for idx in range(qs.size):
        cospsi[idx] = np.sin(thetas[idx])**2 + np.cos(thetas[idx])**2 * np.cos(phis)

    mr.add_known_structure( name, corr1, cospsi )

    # pr_mix = PhaseRetriever(q_values=qs,
    #                               lmax = mr.lmax,
    #                               n_theta= mr.n_theta,
    #                               n_phi = mr.n_phi,
    #                               corr = corr1,
    #                               cospsi = cospsi)

    # x = mr.norm_leg_coefs(pr_mix.cl.copy())
    # np.save('test_data/norm_3p0gA_cl.npy',x)

    x = np.load('test_data/norm_3p0gA_cl.npy')
    
    try:
        assert(np.isclose(mr.known_cl[name] ,x).all())
    except:
        print("WARNING: Did not pass MixRetriever add_known_structure test!!")
        sys.exit()

    print("* Passed MixRetriever add_known_structure test!")

def test_sum_component( ):
    s1 = np.load('test_data/2rh1A-downsamp-infPhot-target-cxs.npy')[1:,:90]
    s2 = np.load('test_data/3p0gA-downsamp-infPhot-target-cxs.npy')[1:,:90]
    mix = np.load('test_data/2rh1A_3p0gA_infPhot_20mol_a0.40-cxs.npy')[:,:90]

    qs = np.load('test_data/qvalue.npy')
    Lmax = 10
    k = 5.0677

    thetas = np.arcsin(qs/(2*k))
    phis = np.linspace(0,np.pi,mix.shape[-1], endpoint=True)
    cospsi = np.zeros( (qs.size, phis.size) )
    for idx in range(qs.size):
        cospsi[idx] = np.sin(thetas[idx])**2 + np.cos(thetas[idx])**2 * np.cos(phis)
    

    mr = MixRetriever(qs.copy(), Lmax, mix.copy(), cospsi.copy())
    mr.add_known_structure('3p0g', s2.copy(), cospsi.copy())
    mr.add_known_structure('2rh1', s1.copy(), cospsi.copy())

    y = mr.known_cl['2rh1']*0.4+mr.known_cl['3p0g']*0.6

    correlation_2_answer = _compute_corr(mr.cl.ravel(),y.ravel())

    try:
        assert( correlation_2_answer > 0.99)
    except:
        print("WARNING: Did not pass MixRetriever summing component test!!")
        print("WARNING: correlation of mixture leg coefs \
            with correct answer is only %.3f"%correlation_2_answer)
        sys.exit()

    print("* Passed MixRetriever summing component test!")

def test_fit1():
    f_test = h5py.File('test_data/2rh1A_4ldlA_theoretical_test.hdf5','r')

    s1 = f_test['2rh1'].value
    s2 = f_test['4ldl'].value
    mix = f_test['mix'].value

    qs = f_test['qvalues'].value
    cospsi = f_test['cospsi'].value

    Lmax = 10
    # # s2 = np.load('test_data/3p0gA-downsamp-infPhot-target-cxs.npy')[1:3,:90]
    # mix = np.load('test_data/2rh1A_3p0gA_infPhot_20mol_a0.40-cxs.npy')[:2,:90]

    # qs = np.load('test_data/qvalue.npy')[:2]
    # Lmax = 10
    # k = 5.0677

    # thetas = np.arcsin(qs/(2*k))
    # phis = np.linspace(0,np.pi,mix.shape[-1], endpoint=True)
    # cospsi = np.zeros( (qs.size, phis.size) )
    # for idx in range(qs.size):
    #     cospsi[idx] = np.sin(thetas[idx])**2 + np.cos(thetas[idx])**2 * np.cos(phis)
    

    mr = MixRetriever(qs.copy(), Lmax, mix.copy(), cospsi.copy())
    mr.add_known_structure('4ldl', s2.copy(), cospsi.copy())
    mr.add_known_structure('2rh1', s1.copy(), cospsi.copy())

    mr.unmix( num_components = 2,
        known_components = mr.known_cl.keys() )
    print mr.guess_concentration

def _compute_corr(x, y):
    x -= x.mean()
    y -= y.mean()
    
    return np.mean(x*y)/np.sqrt(x.var()*y.var())

if __name__=="__main__":
    # mr = test_init()
    # test_add_known_component( mr )
    # test_sum_component()
    test_fit1()

    