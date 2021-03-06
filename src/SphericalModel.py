# class for turning models into spherical harmonics functions
from numpy.polynomial.legendre import legval
import numpy as np
import numpy.linalg as LA
import os

from thor.scatter import simulate_shot

import shtns
import h5py

import copy


class SphericalModel(object):
    """
    Class that simulate the full S(q) for a given trajectory and projects it into 
    spherical harmonics
    
    Attributes
    ----------
    self.model: mdtraj object, assumed to have only one frame
    self.n_theta: int, number of thetas in q space, range = [0, pi]
    self.n_phi: int, number of phis, range = [0, 2pi] 
    self.q_values: np.array, q magnitudes simulated
    self.n_q: int, number of q magnitudes simulated
    self.lmax: int, maximum order (start from 0) of spherical harmonics projection
    self.sh: instance of shtns object, does the spherical harmonic transformation heavy lifting
    
    self.S_q: np.array, float, shape = (n_q, n_theta, n_phi) 
    S(q) scattering intensity in reciprocal space
    
    self.all_slm: np.array, complex, shape = (n_q, (lmax+2)*(lmax+1)/2 )
    All the spherical harmonics coefficients 
    
    self.cl: np.array, float, shape = (lmax+1, n_q, n_q)
    Legengre polynomial coefficients from spherical harmonic coefficients
    
    self.corr: np.array, float, shape = (n_q, n_q, n_psi)
    Correlations between all q_values
    
    self.cospsi: np.array, float, shape = (n_psi)
    Cosines of the angle psi used to compute correlations
    """

    
    def __init__( self, traj=None, 
        SpModel_path = None,
        moo = False ):
        """
        Generate an instance of the SphericalModel class.
        
        Parameters
        ----------
        traj : mdtraj object, with only one frame to represent one model
        
        """
        
        if traj is None:
            # load from path
            self.load( SpModel_path )
        else:
            self.model = traj[0]
        
        if moo:
            print("\
                                                                      >*\n \
                                                               #      >*\n \
                The Spherical Cow                              #  ###>***~~~~~|\n \
                                                               ####  *****^^^#\n \
                                                          _____|       *#####\n \
                                                         | ^^^#   \/ \/ #\n \
                                                        ##^^###         |\n \
                                                         ### ##*        *\n \
             |_                                ********~~|_____>         *\n \
             \\|_                 ________************        #>>***    ***\n \
             \\\\|_             __|     *************        ## >>>*  *****\n \
             |___  |______   __|         ***********       ##>### ^^^^^^^^^^\n \
                |____    |__|           **********       >>>>## ^<^^^^^@^^^^^\n \
                     #          ***      ********      **>>>># ^<^^@^^^@^^^^^\n \
                      #      ***********    ******     *>>>## ^<<^^^^^^^^<<<\n \
                      #      ***********    ******    **>>>## ^<<<<^^^<<<<<\n \
                     #        *********      ****   ***>>>#### ^<<<<<<<<<\n \
                     #         **********          ****>>>###### <<<<<\n \
                     ##        **********          ****>>>>##      ##\n \
                     ##         **  ***             ****>>>>        #     ##XXX\n \
                     ##**                            *******         ##>>>>#XX\n \
                      >>*                             ******         #######XXX\n \
                      >>*****                           ***         ##__\n \
                       >>*****   **** ***               **    *****     \__\n \
                       >># **    *********              *********>>>#      XXX\n \
                       ##        *********              *******>>>>>##     XXX\n \
                    |~~           ********                 *>>>>> >#######XXX\n \
                X~~~~ ###          *********          ######>          >>>XXXX\n \
              XXX  #>>>##          ********>>##  #######\n \
               XXX#>      #   ##>>>>>>>>>>>>>###UUUUU^^\n \
               XXX        #  ####>>>>>>>>>>UUUUUUUUU^^\n \
                          #  >>           UUUUUU^^^<()\n \
                         #  >              U()^<()  ()\n \
                       *#  *>               ()  ()\n \
                      **** #\n \
                        ***\n \
                        ** ")    
                        

    def save(self, save_filename):
        ff = h5py.File(save_filename,'a')
        ff.create_dataset('intensity', data = self.S_q)
        ff.create_dataset('intensity_original', data = self.Sq_original)
        ff.create_dataset('slm',data = self.all_slm)

        ff.create_dataset('qvalues', data = self.q_values)
        ff.create_dataset('leg_coefs', data = self.cl)
        ff.create_dataset('corr', data = self.corr)
        ff.create_dataset('cospsi', data = self.cospsi)
        ff.create_dataset('lmax', data = self.lmax)

        ff.close()

    
    def load(self, filename):
        ff = h5py.File(filename, 'r')
        self.S_q = ff['intensity'].value
        self.n_theta = self.S_q.shape[1]
        self.n_phi = self.S_q.shape[2]

        self.lmax = ff['lmax'].value
        self.sh = shtns.sht(self.lmax)
        # generate the shtns object
        self.sh.set_grid( self.n_theta, self.n_phi )

        self.Sq_original = ff['intensity_original'].value
        
        self.all_slm =ff['slm'].value
        self.corr = ff['corr'].value
        self.cl = ff['leg_coefs'].value
        self.cospsi = ff['cospsi'].value
        self.n_psi = self.cospsi.size

        self.q_values = ff['qvalues'].value
        self.n_q = self.q_values.size


        ff.close()
        
    def slice_by_qvalues(self, q_values, inplace = True):
        """
        qvalues: the q values you want to keep in the sliced model
        """
        if inplace:
            obj_to_slice = self
        else:
            obj_to_slice = copy.copy( self )
        
        q_mask = np.array( [ np.isclose(qq, q_values).any() for qq in obj_to_slice.q_values] )
        q_idx = np.where(q_mask == True)[0]

        obj_to_slice.q_values = self.q_values[q_mask]
        obj_to_slice.n_q = q_idx.size

        obj_to_slice.S_q = self.S_q[q_mask, :, :]
        obj_to_slice.Sq_original = self.Sq_original[q_mask,:,:]

        obj_to_slice.all_slm = self.all_slm[q_mask,:]
        old_corr = self.corr.copy()
        old_cl = self.cl.copy()

        obj_to_slice.corr = np.zeros( (obj_to_slice.n_q, obj_to_slice.n_q, 
            obj_to_slice.n_psi ) )
        obj_to_slice.cl = np.zeros( (obj_to_slice.lmax+1, obj_to_slice.n_q,
         obj_to_slice.n_q) )

        for qi in range(obj_to_slice.n_q):
            for qj in range(qi, obj_to_slice.n_q):
                old_qi = q_idx[qi]
                old_qj = q_idx[qj]
                obj_to_slice.corr[qi,qj,:] = old_corr[old_qi,old_qj,:]
                obj_to_slice.corr[qj,qi,:] = old_corr[old_qj,old_qi,:]

                obj_to_slice.cl[:,qi,qj] = old_cl[:, old_qi,old_qj]
                obj_to_slice.cl[:,qj,qi] = old_cl[:, old_qj,old_qi]

        if not inplace:
            return obj_to_slice

    def slice_by_lmax( self, new_lmax, inplace = False):
        
        if inplace:
            obj_to_slice = self
        else:
            obj_to_slice = copy.copy( self )

        obj_to_slice.lmax = new_lmax
        new_Sq, new_slm, new_sh_obj = obj_to_slice._compute_Sq_with_reduced_lmax( new_lmax )

        obj_to_slice.S_q = new_Sq
        obj_to_slice.all_slm = new_slm
        obj_to_slice.sh = new_sh_obj

        obj_to_slice._leg_coefs_from_sph_harm()
        obj_to_slice._sph_coefs_to_corr()

        if not inplace:
            return obj_to_slice
    
    def simulate( self, q_values, 
        n_theta, n_phi, 
        lmax, n_psi, 
        dont_rotate = True,
        make_positive = True):
        """
        Simulates S(q), projects into spherical harmonics, and computes correlation  
        
        Parameters
        ----------
        q_values : np.array, q magnitudes to be simulated
        n_theta : int, number of thetas in q space, range = [0, pi]
        This is not the same as Bragg theta; this theta = Bragg theta + pi/2
        n_phi: int, number of phis, range = [0, 2pi] 
        lmax: int, maximum order of spherical harmonics projection
        dont_rotate: default True, do not rotate the traj when simulating S(q)
        
        """
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.q_values = q_values
        self.n_q = q_values.shape[0]
        self.lmax = lmax
        
        self.n_psi = n_psi
        
        self.sh = shtns.sht(lmax)
    
        self.sh.set_grid( n_theta, n_phi )
        
        # simulate S_q
        self._compute_Sq( self.model, q_values, dont_rotate)

         # make sure that the inverse spherical transformation always gets a positive intensity map
        if make_positive:
            self._make_Sq_positive()
    
        # project into spherical harmonics
        self._sph_harm_project()
        
        # sum slm to get cl
        self._leg_coefs_from_sph_harm()
        
        # compute correlations
        self._sph_coefs_to_corr()
    
    
    @staticmethod
    def normalize_by_range( x ):    
        """
        normalizes the array x by it's dynamic range and shifts x so it ranges [0,1]
        x: 1d np.array
        """
        range = np.max(x) - np.min(x)
        x /= range
        x -= x.min()
        return x
    
    def _get_qxyz_on_unit_sphere( self ):
        thetas = np.arccos( self.sh.cos_theta )
        phis = (2.*np.pi/ self.n_phi )*np.arange(self.n_phi)
    
        qxyz = np.zeros((thetas.size,phis.size,3))
    
        for i, t in enumerate(thetas):
            for j, p in enumerate(phis):
                qxyz[i,j,2] = np.cos(t)
                qxyz[i,j,0] = np.sin(t) * np.cos(p)
                qxyz[i,j,1] = np.sin(t) * np.sin(p)
    
        return qxyz.reshape(thetas.size*phis.size,3)

    def _compute_Sq( self, traj, q_values, dont_rotate):
        qxyz = self._get_qxyz_on_unit_sphere()
        self.S_q = np.zeros( (self.q_values.shape[0], qxyz.shape[0]) )
        for i,q in enumerate(self.q_values):
            self.S_q[i,:] = simulate_shot(self.model, 1, q*qxyz, dont_rotate=dont_rotate, force_no_gpu=True)
        self.S_q = self.S_q.reshape( self.n_q, self.n_theta, self.n_phi )
    
    def _compute_Sq_with_reduced_lmax( self, new_lmax):
        """
        model: SphericalModel object
        pr: PhaseRetriver object
        """
        new_sh_obj = shtns.sht(new_lmax)

        old_sh_obj = self.sh
        
        new_S_q = np.zeros_like( self.S_q )
        
        old_all_slm = self.all_slm
        
        new_sh_obj.set_grid( self.n_theta, self.n_phi )
        n_q = self.n_q
        new_all_slm = np.zeros((self.n_q, int((new_lmax+2)*(new_lmax+1)/2) ) 
            , dtype=np.complex128)

        for qid in range(n_q):
            new_slm = new_sh_obj.spec_array()
            
            for lid in range(new_lmax+1):
                for mid in range(lid+1):
                    new_slm[new_sh_obj.idx(lid, mid)] = old_all_slm[qid][old_sh_obj.idx(lid,mid)]
            new_S_q[qid] = new_sh_obj.synth(new_slm)
            new_all_slm[qid, :] = new_slm

        return new_S_q, new_all_slm, new_sh_obj


    def _make_Sq_positive( self ):
        """
        Transform Sq into spherical harmonics and performes an inverse transformation. 
        Make the inverse transformation positive by subtracting the minimum. Transform again to get
        Sphiercal harmonic coefficients. This is kind of like cheating
        """
        Sq_pos = np.zeros_like(self.S_q)
        
        # self.sh_obj = shtns.sht(self.lmax)    
        # self.sh_obj.set_grid( self.n_theta, self.n_phi )
        for iq in range(self.S_q.shape[0]):
            ylm = self.sh.spec_array()
            ylm = self.sh.analys( self.S_q[iq] )
        
            S2 = self.sh.synth(ylm)
            Sq_pos[iq] = S2-S2.min()

        self.Sq_original = self.S_q.copy()
        self.S_q = Sq_pos.copy()

    def _sph_harm_project( self ):
        """
        computes spherical harmonic coefficients
        """
        # project into spherical harmonics
        self.all_slm = np.zeros((self.n_q, int((self.lmax+2)*(self.lmax+1)/2) ) , dtype=np.complex128)

        for i in range( self.n_q ):
            self.all_slm[i,:] = self.sh.analys( self.S_q[i] )

        # sum slm to get cl
        # is this necessary? lists are very inefficient data structures, Let try without it first
    #     self.Il = self.Il_matrices(sh, all_slm, lmax = lmax)

    def _leg_coefs_from_sph_harm (self):
        """
        computes legengre polynomial coefficients from spherical harmonic coefficients
        """

        self.cl = np.zeros(( self.lmax+1, self.n_q, self.n_q), dtype = float)
        for iq in range( self.n_q ):
            for jq in range( self.n_q ):
                for ll in range( self.lmax+1 ):
                    self.cl[ll, iq, jq] = np.sum( np.conjugate( self.all_slm[iq][ self.sh.l==ll]) \
                        * self.all_slm[jq][self.sh.l==ll]) * 2.0 \
                    - np.conjugate( self.all_slm[iq][self.sh.idx(ll,0)] ) * self.all_slm[jq][self.sh.idx(ll,0)]
        try:
            assert ( np.all ( np.isclose( np.imag(self.cl), 0) ) )
        except:
            print ( "WARNING: Non-zero imaginary values occurred in legendre polynomial coefficients (cl)" )

    def _sph_coefs_to_corr( self ):
        """
        computes correlation from legendre polynomial coefficients computed from spherical
        harmonic coefficients
    
        sets coefficients of order-l polynomials to zero
        """
        self.corr = np.zeros( (self.n_q, self.n_q, self.n_psi) )
        self.cospsi = np.linspace(-1,1, self.n_psi, endpoint=True)
        # Cl = Cl[:(Lmax+1),:,:]
    
        for i in range(self.n_q):
            for j in range(i, self.n_q):
                coefs = self.cl[:,i,j]
    #             even_coefs = Cl[:int(Lmax/2),i,j]
                # even_coefs= self.cl[::2,i,j]
                # for idx in range(even_coefs.size * 2):
                #     if idx%2 ==0:
                #         coefs.append(even_coefs[idx/2])
                #     else:
                #         coefs.append(0)
                # coefs = np.array(coefs)
            
                self.corr[i,j,:] = legval( self.cospsi, coefs )
                self.corr[j,i,:] = self.corr[i,j,:]  # copy it to the lower triangle too







    # def _Il_matrices(self, all_slm):
    
    #     self.Il = []
    #     for l in range(self.lmax+1):
    #         if l==0:
    #             slm = np.zeros((self.n_q, 1), dtype = np.complex64)
    #             slm[:,0] = all_slm[:,self.sh.idx(0,0)]
    #         else:
    #             slm = np.zeros((self.n_q, 2*l+1), dtype = np.complex64)
    #             for m in range(1,l+1):
    #                 slm[:,m+l] = all_slm[:, self.sh.idx(l,m)]
    #                 slm[:,-m+l] = (-1) ** m *np.conjugate(all_slm[:, self.sh.idx(l,m)]) 
    #             slm[:,l] = all_slm[:, self.sh.idx(l,0)]
                
    #         self.Il.append(slm)
