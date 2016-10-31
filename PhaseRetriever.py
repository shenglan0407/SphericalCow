# class that takes correlations and try to solve the hyper phase retrieval problem.
from numpy.polynomial.legendre import legval
import numpy as np

class PhaseRetriever(object):
    """
    Class for solving the hyperphase problem
    """
    
    def __init__(self, corr, cospsi, lmax, bark=False):
        
        self.n_q = corr.shape[0]
        if len(corr.shape) == 2:
            # that means there are only autocorrelator
            if len(cospsi.shape) == 1:
                # that means all the q values have the same cospsi, reshape
                cospsi = np.array( [cospsi] * self.n_q )
                
        elif len(corr.shape) == 3:
            # auto and cross-correlations
            if len(cospsi.shape) == 1:
                # that means all the q values have the same cospsi, reshape
                cospsi = np.array( [[cospsi] * self.n_q] * self.n_q )
        else:
            print("Error: correlations should be either a 2D or 3D array")
        
        self.corr = corr
        self.cospsi = cospsi
        
        try:
            assert( self.cospsi.shape[0] == self.n_q)
            assert( self.corr.shape[-1] == self.cospsi.shape[1])
            assert( len(self.corr.shape) == len(self.cospsi.shape) )
        except:
            print ("ERROR: Mismatch in cosines and correlations provided!")
        
        # project leg poly
        
        
        if bark:
            print ("\
                 .:##:::.\n \
              .:::::/;;\:.\n\
        ()::::::@::/;;#;|:.\n\
        ::::##::::|;;##;|::\n\
         \':::::::::\;;;/::\'\n\
              ':::::::::::\n\
               |O|O|O|O|O|O\n\
               :#:::::::##::.\n\
              .:###:::::#:::::.\n\
              :::##:::::::::::#:.\n\
               ::::;:::::::::###::.\n\
               \':::;::###::;::#:::::\n\
                ::::;::#::;::::::::::\n\
                :##:;::::::;::::###:::     .\n\
              .:::::; .:::##::::::::::::::::\n\
              ::::::; :::::::::::::::::##::\n\
              The Phase Retrieving Golden Retriever Puppy!")
    
##################
# Fit
##################
    def fit(self):
        S_q = None
        return S_q
    
##################
# Legendre projection
##################
    def leg_matrix( self ):
        self.cl = np.zeros( ( self.lmax, self.n_q, self.n_q ) )
        
        if len(self.corr) == 3:
            for i in range(self.n_q):
                for j in range(i, self.n_q):
                    c = self._leg_coefs( self.cospsi[i, j, :], 
                    self.corr[i,j,:], self.lmax )
                    self.cl[:,i,j] = c
                    self.cl[:,j,i] = c  # copy it to the lower triangle too

    # combine the following two method into one
    def get_leg_coef(cos_psi,corr,deg):
        ind = np.argsort(cos_psi)
        corr = corr[ind]
        cos_psi = sorted(cos_psi)
    
        d_cos_psi = [cos_psi[ii+1]-cos_psi[ii] for ii in range(len(cos_psi)-1)]
        try:
            assert deg>=0
            if deg%2==1:
                return 0
            else:
                if deg>0:
                    cc=[0]*(deg)
                    cc.append(1)
                elif deg == 0:
                    cc=[1]
        except AssertionError:
            print "Degree of polynomial must be a non-negative inter"
            return
    
        pp=legval(cos_psi,c=cc)
        coef= np.sum([pp[ii]*corr[ii]*d_cos_psi[ii] for ii in range(len(pp))[:-1]])
        coef = coef*(2*deg+1)/2
        return coef

    def _leg_coefs(self, x, y, lmax):
        c = []
        for l in range(lmax):
            c.append(get_leg_coef(x,y,l))
    
        return np.array(c)







##################
# phase-retrieval algorithms
##################




##################
# spherical space constraints
##################




##################
# q-space constraints
##################
    
    
    