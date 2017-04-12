from PhaseRetriever import PhaseRetriever
import numpy as np

from numpy.polynomial.legendre import legval
from numpy.polynomial.legendre import leggauss

from scipy.optimize import curve_fit


class MixRetriever(PhaseRetriever):
    """
    MixRetriever primarily functions as a class to extract individual 
    correltations and molar concentrations from a mixture. It inherits
    from PhaseRetriever so that if we desire we can use the extracted 
    correlations to retriever hyperphase and reconstruct scattering 
    intensities. 
    """

    def __init__(self,
        q_values,lmax, corr, cospsi,
        n_theta = 100, n_phi = 100,
        bark = False, normalize_cl = False,
        **kwargs):

        # check that q_values, cospsi is ascending
        try:
            assert( (sorted(q_values)==q_values).all() )
        except AssertionError:
            q_values = np.array(sorted(q_values))

        try:
            # currently only deal with auto corr
            assert(len(corr.shape) == 2)
        except AssertionError:
            print ("ERROR: Current version of MixRetriever only deal with autocorr.\n \
                corr needs to be 2d and num q by num phi.")
            return

        corr, cospsi = self._check_cospsi( corr, cospsi )

        PhaseRetriever.__init__(self,
        q_values,lmax, n_theta, n_phi,
        corr = corr, cospsi = cospsi,
        **kwargs)

        # normalize leg coefs by the maximum in each q
        self.normalize_cl = normalize_cl
        if self.normalize_cl:
            self.cl = self.norm_leg_coefs( self.cl )

        # components of the mix, with known structures therefore known cls
        self.known_cl = {}
        # components of the mix, with unknown structures needed to be guessed
        self.guess_cl = {}
        # guess concentrations
        self.guess_concentration = {}

        if bark:
            print("\
                 ...... //^ ^\\\ \n \
                ......(/(_o_)\) \n \
                ......_/''*''\_ \n \
                .....(,,,)^(,,,) \n \
                The Mixed Retriever puppy is happy to see you!")
    
    def _check_cospsi( self, corr, cospsi):
        """
        checks if cospsi is acsending
        if not, sort it and corr accordingly
        """

        if len(cospsi.shape) == 1:
            try:
                assert( (sorted(cospsi)==cospsi).all() )
            except:
                sort_ind = np.argsort(cospsi)
                cospsi = sorted(cospsi)
                corr = corr[:,sort_ind]

        elif len(cospsi.shape) == 2 :
            for idx in range(cospsi.shape[0]):
                try:
                    assert( (sorted(cospsi[idx])==cospsi[idx]).all() )
                except:
                    sort_ind = np.argsort(cospsi[idx])
                    cospsi[idx] = sorted(cospsi[idx])
                    corr[idx] = corr[idx, sort_ind]

        return corr, cospsi

    def norm_leg_coefs(self, cl):
        """
        Normalize the leg coefs at each q by the max
        """
        new_cl = np.zeros_like( cl )
        for i in range(cl.shape[1]):
            for j in range(cl.shape[2]):
                if (cl[:,i,j]).max() >0:
                    new_cl[:,i,j] = cl[:,i,j]/(cl[:,i,j]).max()

        return new_cl


    #########################
    # mix component leg coefs
    #########################
    def add_known_structure(self, name, corr, cospsi):
        """
        add leg coefs of a known component 
        """
        if len(cospsi.shape) == 1:
            cospsi = np.array( [cospsi] * self.n_q )

        corr, cospsi = self._check_cospsi( corr, cospsi )
        cl = self._compute_component_legendre_projection(corr, cospsi)
        self.known_cl.update({name: cl})

    def add_unknown_strcuture(self, name, cospsi = None,
        corr = None, cl = None):
        """
        add an unknown structure, initial guess can be added as well
        
        name - str, name of the component

        keyword args
        cospsi - np.array, num q by num phi, default None
        must be defined at the same time as corr

        corr - np.array, num q by num phi , default None
        must be defined at the same time as cospsi
        
        cl - np.array, lmax+1 by num q by num q, default None
        leg coefs guess for an unknown component. 
        Can be defined instead of cospsi and corr
        """
        if cospsi is not None:
        # if cospsi and corr are both given, use them    
            if corr is None:
                pass
            else:
                if len(cospsi.shape) == 1:
                    cospsi = np.array( [cospsi] * self.n_q )
                try:
                    assert(corr.shape[0] == self.n_q)
                    assert(corr.shape[-1] == cospsi.shape[-1])
                except AssertionError:
                    print("EEROR: corr and cospsi shape mismatch")
                    return

                corr, cospsi = self._check_cospsi( corr, cospsi )
                cl = self._compute_component_legendre_projection(corr, cospsi)
                self.guess_cl.update({name: cl})
        
        elif cl is not None:
        # if cl is given
            try:
                assert( cl.shape[0] == self.lmax+1)
                assert( cl.shape[-1] == self.n_q)
            except AssertionError:
                print("ERORR: cl shape mismatch")

            self.guess_cl.update( {name:cl} )




    def _compute_component_legendre_projection( self, corr, cospsi ):
        # define gaussian quadrature, the points between -1, 1 and the weights
        xnew, ws = leggauss( self.cospsi.shape[-1] )

        # interpolate
        cl = np.zeros( ( self.lmax + 1, self.n_q, self.n_q ) )
        for i in range(self.n_q):
            if self.auto_only:
                signal = corr[i,:]
                signal_interp = self._interpolate_corr(cospsi[i,:],
                                                      xnew, signal)
                c = self._leg_proj_legguass( xnew, signal_interp, ws)
                cl[:,i,i] = c

            else:
                for j in range(i, self.n_q):
                    signal = corr[i,j,:]
                    signal_interp = self._interpolate_corr(cospsi[i,j,:],
                                                          xnew, signal)
                    c = self._leg_proj_legguass( xnew, signal_interp, ws)
                    cl[:,i,j] = c
                    cl[:,j,i] = c  # copy it to the lower triangle too
        if self.normalize_cl:
            cl = self.norm_leg_coefs( cl )
        return cl
    #########
    # unmix
    #########
    def unmix(self, num_components,
        known_components, 
        unknown_components = [],
        normalized_weights = True
        ):
        """
        unmix self.corr into components

        num_componets - int, number of components in the mixture
        
        known_components - list of str, names of components that have knonw structures

        keyword args
        unknown_components - list of str, name of components that have unknown structures
        
        """

        assert(len(known_components)+len(unknown_components) == num_components)

        if len(known_components) == num_components:
            # this is the case where all the structures are known and we only need to fit for concentrations
            assert( np.all( [name in self.known_cl.keys() for name in known_components] ) )
            if normalized_weights:
                self._fit1()
            else:
                self._fit2()


        else:
            # now there are unknown components
            pass


    #########################
    # fit for concentrations
    #########################
    def _norm_weighted_sum(self, X, *weights):
        n_things = len(X)
        sum = np.zeros_like( X[0] )
        for ii in range(n_things-1):
            sum +=  X[ii] * weights[ii] 
        sum += X[-1] * (1.0 - np.sum(weights) )

        return sum
    
    def _weighted_sum(self, X, *weights):
        n_things = len(X)
        sum = np.zeros_like( X[0] )
        for ii in range(n_things):
            sum +=  X[ii] * weights[ii] 
        
        return sum

    def _fit1(self):
        X = [ self.known_cl[k].flatten("c") for k in self.known_cl.keys()]
        Y = self.cl.flatten("c")

        p0 = [1.0/len( self.known_cl.keys() )] * (len( self.known_cl.keys() ) - 1 )
        con, _  = curve_fit(self._norm_weighted_sum, X, Y, p0=p0)

        for idx, k in enumerate( self.known_cl.keys() ):
            if idx < len(con):
                self.guess_concentration.update( {k: con[idx] })
            else:
                self.guess_concentration.update( {k: (1.0 - np.sum(con)) })

    def _fit2(self):

        X = [ self.known_cl[k].flatten("c") for k in self.known_cl.keys()]
        Y = self.cl.flatten("c")
        guess_num = self.cl[:,0,0].max()/\
        self.known_cl[ self.known_cl.keys()[0] ][:,0,0].max()
        p0 = [guess_num] * (len( self.known_cl.keys() ) )
        con, _  = curve_fit(self._weighted_sum, X, Y, p0=p0)

        for idx, k in enumerate( self.known_cl.keys() ):
            self.guess_concentration.update( {k: con[idx] })
            
    ###########################
    # fit for unknown leg coefs
    ###########################

