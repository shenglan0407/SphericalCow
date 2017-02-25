# class that takes correlations and try to solve the hyper phase retrieval problem.

import numpy as np
import numpy.linalg as LA
import shtns

from numpy.polynomial.legendre import legval
from numpy.polynomial.legendre import leggauss

from scipy.interpolate import splev,splint,splrep

import matplotlib.pyplot as plt

class PhaseRetriever(object):
    """
    Class for solving the hyperphase problem
    """
    
    def __init__(self, 
        q_values,
        lmax, n_theta, n_phi,
        corr = None, cospsi = None,
        ref_SphModel = None, auto_only = False,
        bark=False):
        
        # q values correponding to correlations, in ascending order!!!
        self.q_values = q_values
        # can pass a reference Spherical Model, use it for guessing, and comparing
        self.ref_SphModel = ref_SphModel

        self.lmax = lmax

        self.n_q = len(q_values)

        self.n_theta = n_theta
        self.n_phi = n_phi

        self.sh = shtns.sht(lmax)
        self.sh.set_grid( n_theta, n_phi )
        
        if corr == None:
            self.corr = self.ref_SphModel.corr
            self.cospsi = self.ref_SphModel.cospsi

            self.auto_only = auto_only
            if auto_only:
                self.cl = np.zeros_like(self.ref_SphModel.cl)
                self.cl[:,range(self.n_q),range(self.n_q)] = self.ref_SphModel.cl[:,range(self.n_q),range(self.n_q)]
            else:
                self.cl = self.ref_SphModel.cl
        else:

            if len(corr.shape) == 2:
                # that means there are only autocorrelator
                self.auto_only = True
                if len(cospsi.shape) == 1:
                    # that means all the q values have the same cospsi, reshape
                    cospsi = np.array( [cospsi] * self.n_q )
                    
            elif len(corr.shape) == 3:
                self.auto_only = False
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
                assert( self.corr.shape[-1] == self.cospsi.shape[-1])
                assert( len(self.corr.shape) == len(self.cospsi.shape) )
            except:
                print ("ERROR: Mismatch in cosines and correlations provided!")
            
            # project leg poly
            self._compute_legendre_projection()
        
        
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
    def fit(self, num_iter,
        plt_init = False,
        smooth = False):
        
        self.smooth = smooth
        if smooth:
            self._define_smoothness_operators()

        self._initiate_guess_int()
        if plt_init:
            plt.figure()
            for qq in range( self.n_q ):
                plt.subplot( 1, self.n_q, qq+1)
                plt.imshow( self.I_guess[qq] )
                plt.colorbar()
            plt.show()

        deltas = []
        similarity_to_answer=[]
        self.all_slm_guess = np.zeros( ( self.n_q, ( self.lmax**2 + 3*self.lmax+2)/2 )
                        , dtype=np.complex128 )



        for ii in range(num_iter):
            I_guess_old = self.I_guess.copy()
            
            for q_idx in range(self.n_q):
            # get slms for all q values first
                slm_guess = self.sh.analys( self.I_guess[q_idx])
                self.all_slm_guess[q_idx] = slm_guess
            
            if self.auto_only:

                self._impose_auto_corr()

                for q_idx in range( self.n_q ):
                    
                    self.I_guess[q_idx] = self.sh.synth( self.all_slm_guess[q_idx] )

                    # impose positivity
                    update_idx = self.I_guess[q_idx] < 0
                    self.I_guess[q_idx][update_idx] = 0
                    
                    try:
                        this_mask = self.masks[q_idx]
                        self.I_guess[q_idx, this_mask] = self.ref_SphModel.S_q[q_idx, this_mask]
                    except AttributeError:
                        # there is no mask to apply
                        pass
#         
                    
            
            else:
                for q_idx in range(self.n_q):
                    
                    self._impose_cross_corr()
                    
                    self.I_guess[q_idx] = self.sh.synth( self.all_slm_guess[q_idx] )

                    # impose positivity
                    update_idx = self.I_guess[q_idx]<0
                    self.I_guess[q_idx][update_idx] = 0

                    # impose mask iffmasks exist
                    try:
                        this_mask = self.masks[q_idx]
                        self.I_guess[q_idx, this_mask] = self.ref_SphModel.S_q[q_idx, this_mask]
                    except AttributeError:
                        # there is no mask to apply
                        pass
  

           
            self.I_guess = self.impose_fridel( self.I_guess )

            # a better way to evaluate fit to reference model is to use correlation
            # will implement this later
            dd = np.sqrt( (( self.I_guess - I_guess_old)**2).mean(axis=(1,2)) )/ \
            self.ref_SphModel.S_q.mean(axis=(1,2))
            deltas.append(dd)

            dd = np.sqrt( ((self.I_guess - self.ref_SphModel.S_q)**2).mean(axis=(1,2)) )/ \
            self.ref_SphModel.S_q.mean(axis=(1,2))
            similarity_to_answer.append(dd)

        deltas=np.array(deltas)
        similarity_to_answer=np.array(similarity_to_answer)


        return self.I_guess, deltas, similarity_to_answer

    def _define_smoothness_operators(self):
        # define the differentiation operator
        diff_q = np.diag( 1.0/ ( self.q_values - np.roll( self.q_values, 1 ) ) )
        self.diff_op = diff_q.dot( np.roll( np.diag( np.ones( self.n_q ) ), -1, axis = 0)\
         - np.diag( np.ones( self.n_q ) ) )

        self.Dl_list = []

        for l_test in range( self.lmax+1 ): 
            Sl_correct = np.zeros( (self.n_q,2*l_test+1), dtype = np.complex)
            
            for qq in range( self.n_q ):
                Sl_correct[qq,l_test:] = \
                self.ref_SphModel.all_slm[qq][ self.sh.l == l_test]
                
                Sl_correct[ qq, :l_test] = \
                np.conjugate( self.ref_SphModel.all_slm[qq][ self.sh.l==l_test][1:][::-1])
            Dl = self.diff_op.dot(Sl_correct)
            self.Dl_list.append(Dl)      

##################
# Legendre projection
##################


    def _compute_legendre_projection( self ):
        # define gaussian quadrature, the points between -1, 1 and the weights
        xnew, ws = leggauss( self.cospsi.shape[-1] )

        # interpolate
        self.cl = np.zeros( ( self.lmax + 1, self.n_q, self.n_q ) )
        for i in range(self.n_q):
            if self.auto_only:
                signal = self.corr[i,:]
                signal_interp = self._interpolate_corr(self.cospsi[i,:],
                                                      xnew, signal)
                c = self._leg_proj_legguass( xnew, signal_interp, ws)
                self.cl[:,i,i] = c

            else:
                for j in range(i, self.n_q):
                    signal = self.corr[i,j,:]
                    signal_interp = self._interpolate_corr(self.cospsi[i,j,:],
                                                          xnew, signal)
                    c = self._leg_proj_legguass( xnew, signal_interp, ws)
                    self.cl[:,i,j] = c
                    self.cl[:,j,i] = c  # copy it to the lower triangle too
        #

    def _interpolate_corr( self, xold, xnew, signal ):

        # compute the bspline representation of the signal
        # k =5 gives the highest order
        tck = splrep(xold,signal,k=5,s=0)
        signal_interp = splev(xnew,tck, der=0)

        return signal_interp
    
    def _leg_proj_legguass(self, x,signal,weights):
        """projection into legendre polynomial using gauss-legendre polynomial
        """
        coefs = np.zeros(self.lmax + 1)
        weights_sum = np.sum(weights)
        
        for l in range(self.lmax + 1):
            if l%2==0:
                cc = np.zeros(l+1)
                cc[l] = 1
                coefs[l] = np.sum( weights*legval(x,cc)*signal) / weights_sum*(2*l+1)

        return coefs

 

##################
# initial guess
##################
    def _initiate_guess_int( self ):

        if self.ref_SphModel is not None:
            #this computes the average intensity of the reference intensity
            ref_int_average = self.ref_SphModel.S_q.mean(axis=(1,2))[:,None,None]

            I_guess = np.random.rand( self.n_q, self.n_theta, self.n_phi )\
            * ref_int_average

            try:
                for q_idx in range(self.n_q):
                    this_mask = self.masks[q_idx]
                    I_guess[q_idx, this_mask] = self.ref_SphModel.S_q[q_idx, this_mask]
            except AttributeError:
                # masks do not exist
                pass
        else:
            #if there is no reference intensity, then just guess randomly
            I_guess = np.random.rand( self.n_q, self.n_theta, self.n_phi )* 1e6

        # impose friedel symmetry on the initial guess
        self.I_guess = self.impose_fridel(I_guess)


##################
# phase-retrieval algorithms
##################





# ##################
# # spherical space constraints
# ##################



    def _impose_cross_corr( self ):
        """
        impose constraints set by the correlations, use this if there is cross-correlations data
        """
        for l_test in range( self.lmax+1 ):    

            ll,v = LA.eig( self.cl[l_test] )

            Ul = np.zeros( (2*l_test+1, 2*l_test+1), dtype=np.complex )
            Ll = np.zeros( (2*l_test+1, 2*l_test+1), dtype = float )
            Vl = np.zeros( (self.n_q, 2*l_test+1), dtype = float )

            for ii in range( self.n_q ):
                if ii < Ll.shape[0]:
                    Ll[ii,ii] = np.sqrt(np.real(ll[ii]))
                    Vl[:,ii] = v[:,ii]

            Sl = np.zeros( ( self.n_q, 2*l_test+1), dtype = np.complex)
            # fill the Sl matrix for this iteration
            for qq in range(self.n_q):
                Sl[qq,l_test:] = self.all_slm_guess[qq][ self.sh.l == l_test]
                Sl[qq, :l_test] = \
                np.conjugate( self.all_slm_guess[qq][self.sh.l==l_test][1:][::-1])


            if self.smooth:
                gamma = self.diff_op.dot(Vl).dot(Ll)
                Dl = self.Dl_list[l_test]
                # Dl = np.repeat( Dl.mean(axis = 1), l_test*2+1).reshape( self.n_q, l_test*2+1)

                M = np.dot(Ll, np.conjugate(Vl).T ).dot(Sl) + (gamma.T).dot(Dl)
            else:
                M = np.dot(Ll, np.conjugate(Vl).T ).dot(Sl)
            
            u,s,v = LA.svd(M, full_matrices=False)
            Ul = u.dot(v)
            Sl_guess = np.dot(Vl,Ll).dot(Ul)
            
            for qq in range( self.n_q ):
                for m in range( l_test+1 ):
                    self.all_slm_guess[qq][self.sh.idx(l_test,m)] = Sl_guess[qq][m+l_test]

    def _impose_auto_corr( self ):
        """
        impose constraints set by the correlations, use this if there is cross-correlations data
        """
        
        for l_test in range( self.lmax+1 ):

            Sl = np.zeros(( self.n_q, 2*l_test+1 ), dtype = np.complex)
            
            for qq in range( self.n_q):
                Sl[qq,l_test:] = self.all_slm_guess[qq][self.sh.l == l_test]
                Sl[qq, :l_test] = np.conjugate(self.all_slm_guess[qq][self.sh.l==l_test][1:][::-1])
            
            Sl2_diag = np.diag(np.sqrt(1.0/np.sum(np.abs(Sl)**2, axis =1) ) )
            
            Sl_guess = Sl2_diag.dot( np.sqrt( self.cl[l_test] ) ).dot(Sl)
            try:
                assert( np.isclose( np.diag( np.sum( np.abs( Sl_guess )**2, axis =1)), 
                    self.cl[l_test]).all())

            except AssertionError:
                print ("Warning! Large numerical errors in autocorrelation-only procrustes problem")
                
            for qq in range(self.n_q):
                for m in range(l_test+1):
                    self.all_slm_guess[qq][self.sh.idx(l_test,m)] = Sl_guess[qq][m+l_test]

    

# ##################
# # q-space constraints
# ##################

    def define_mask_from_intensity( self,  thresholds ):
        """
            thresholds is a list of length num_q
        """
        if self.ref_SphModel is not None:
            self.masks = []
            self.thresholds = thresholds

            for iq in range(self.n_q):
                self.masks.append( self.ref_SphModel.S_q[iq] < thresholds[iq] )

    def impose_fridel(self, x):
        
        if len(x.shape)== 2:
            mid_x = x.shape[0]/2
            mid_y = x.shape[1]/2
            half1=x[:mid_x,:mid_y].copy()
            half2=x[mid_x:,mid_y:].copy()
            x[:mid_x,:mid_y] = (half1 + half2[::-1,::-1])/2
            x[mid_x:,mid_y:] = x[:mid_x,:mid_y][::-1,::-1]
            
            half1=x[:mid_x,mid_y:].copy()
            half2=x[mid_x:,:mid_y].copy()
            x[:mid_x,mid_y:] = (half1 + half2[::-1,::-1])/2
            x[mid_x:,:mid_y] = x[:mid_x,mid_y:][::-1,::-1]
        
        elif len(x.shape) == 3:
            mid_x = x.shape[1]/2
            mid_y = x.shape[2]/2
            half1=x[:, :mid_x, :mid_y].copy()
            half2=x[:, mid_x:, mid_y:].copy()
            x[:, :mid_x,:mid_y] = (half1 + half2[:,::-1,::-1])/2
            x[:, mid_x:,mid_y:] = x[:, :mid_x,:mid_y][:,::-1,::-1]
            
            half1=x[:, :mid_x,mid_y:].copy()
            half2=x[:, mid_x:,:mid_y].copy()
            x[:, :mid_x,mid_y:] = (half1 + half2[:, ::-1,::-1])/2
            x[:, mid_x:,:mid_y] = x[:, :mid_x,mid_y:][:, ::-1,::-1]
        
        return x

