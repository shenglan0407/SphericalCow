# class that takes correlations and try to solve the hyper phase retrieval problem.
from numpy.polynomial.legendre import legval
import numpy as np
import numpy.linalg as LA
import shtns

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
        
        if len(self.corr.shape) == 3:
            self.cl = np.zeros( ( self.lmax + 1, self.n_q, self.n_q ) )
            for i in range(self.n_q):
                for j in range(i, self.n_q):
                    c = self._leg_coefs( self.cospsi[i, j, :], 
                    self.corr[i,j,:] )
                    self.cl[:,i,j] = c
                    self.cl[:,j,i] = c  # copy it to the lower triangle too
        
        elif len(self.corr.shape) == 2:
            self.cl = np.zeros( ( self.lmax + 1, self.n_q ) )
            for i in range(self.n_q):
                c = self._leg_coefs( self.cospsi[i, :], 
                    self.corr[i,:] )
                self.cl[:,i] = c

    def _leg_coefs(self, x, y):
        
        ind = np.argsort(x)
        y = y[ind]
        x = np.sort(x)
        dx = x[1:] - x[:-1]
        
        leg_coefs = np.zeros( self.lmax + 1 )
        for l in range( 0, self.lmax+1, 2 ):
            cc = np.zeros( l+1 )
            cc[-1] = 1
            
            pp = legval( x, c=cc )
            # may beed a better way to do numerical integration?
            coef = np.sum( pp[:-1] * y[:-1] * dx )
            coef *= (2*l+1)/2.
            
            leg_coefs[l] = coef
        
        return leg_coefs


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
























##################Old cod probably not needed##########################


#     def _error_reduction ( self ):
    
#         Il_input = self.Il_guess
#         losses = []
    
#         for iter in range(num_iter):
#         if iter%1000 ==0:
#             print "completed %d itertations"%(int(iter/1000))
#         assert ( len(S_q_input.shape) == 2)
#         all_slm_input = np.zeros((S_q_input.shape[0], int((sh_deca.lmax+2)*(sh_deca.lmax+1)/2) ) , dtype=np.complex)
    
#         for i in range( S_q.shape[0] ):
#             all_slm_input[i,:] = sh_deca.analys( S_q_input[i].reshape(sh_deca.nlat,sh_deca.nphi) )

#         Il_input = hf.Il_matrices(sh_deca, all_slm_input, lmax = sh_deca.lmax)

#         # update using cl
#         Il_output = update_Il_auto ( Il_input, cl_deca )
#     #     Il_output = update_Il( Il_input, cl_deca)
#         # inverse transform to S_new
#         S_q_output = np.array( np.real (hf.Il_to_Sq( Il_output, sh_deca ) ), dtype = np.float64)
#         # apply hydrid input output using only positivity
#     #     
#         S_q_output = S_q_output.reshape( S_q_input.shape[0], sh_deca.nlat*sh_deca.nphi )
#         S_q_output = normalize_by_max( S_q_output )
#         # print S_q_output.shape 
#         # print S_q_input.shape
#         assert ( S_q_output.shape == S_q_input.shape )
#         # positivity support
#         update_indices1 = np.where( S_q_output < 0 )
#         update_indices2 = np.where( S_q_output >= 0 )

#         # update input for next iteration
#         beta = 0.05
#         S_q_input[update_indices1] = S_q_input[update_indices1]- beta * S_q_output[update_indices1]
#         S_q_input[update_indices2] = S_q_output[update_indices2]
    
    
#     #     S_q_input = np.array( np.real( pos_support(S_q_output) ), dtype = np.float64)
#         S_q_input = normalize_by_max(S_q_input)
#         # compute loss
#         loss = np.sum( (S_q_output - S_q_input )**2.0 )
#         losses.append(loss)
    
#         pass

#     def _grad_descent( self ):
#         pass

#     def _hybrid_input_output( self ):
#         pass




# ##################
# # spherical space constraints
# ##################




# ##################
# # q-space constraints
# ##################
    
    
    