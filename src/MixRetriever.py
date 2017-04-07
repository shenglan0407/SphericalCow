from PhaseRetriever import PhaseRetriever
import numpy as np

from numpy.polynomial.legendre import legval
from numpy.polynomial.legendre import leggauss


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
        bark = False,
        **kwargs):

        # check that q_values, cospsi is ascending
        try:
            assert( (sorted(q_values)==q_values).all() )
        except AssertionError:
            q_values = np.array(sorted(q_values))

        corr, cospsi = self._check_cospsi( corr, cospsi )

        PhaseRetriever.__init__(self,
        q_values,lmax, n_theta, n_phi,
        corr = corr, cospsi = cospsi,
        **kwargs)

        # normalize leg coefs by the maximum in each q
        self.cl = self.norm_leg_coefs( self.cl )

        # components of the mix, with known structures therefore known cls
        self.known_cl = {}
        # components of the mix, with unknown structures needed to be guessed
        self.guess_cl = {}

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

    def norm_leg_coefs(sself, cl):
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
        corr, cospsi = self._check_cospsi( corr, cospsi )
        cl = self._compute_component_legendre_projection(corr, cospsi)
        self.known_cl.update({name: cl})


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

        cl = self.norm_leg_coefs( cl )
        return cl

    #########################
    # fit for concentrations
    #########################

    ###########################
    # fit for unknown leg coefs
    ###########################

