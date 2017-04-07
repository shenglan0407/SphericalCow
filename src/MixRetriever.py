from PhaseRetriever import PhaseRetriever
import numpy as np



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

        PhaseRetriever.__init__(self,
        q_values,lmax, n_theta, n_phi,
        corr = corr, cospsi = cospsi,
        **kwargs)


        if bark:
            print("\
                 ...... //^ ^\\\ \n \
                ......(/(_o_)\) \n \
                ......_/''*''\_ \n \
                .....(,,,)^(,,,) \n \
                The Mixed Retriever puppy is happy to see you!")


    #########################
    # mix component leg coefs
    #########################


    #########################
    # fit for concentrations
    #########################

    ###########################
    # fit for unknown leg coefs
    ###########################

