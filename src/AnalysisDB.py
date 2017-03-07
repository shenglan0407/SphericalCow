import h5py
import numpy as np
import mdtraj

from PhaseRetriever import PhaseRetriever
from SphericalModel import SphericalModel
from evalFit import *

class AnalysisDB(object):
	"""
	makde database to store results of analysis
	"""

	def __init__(self, traj_full_path, traj_guess_path,
	model_full_path, model_guess_path):

		self.traj_full_path = traj_full_path
		self.traj_guess_path = traj_guess_path

		self.model_full_path = model_full_path 
		self.model_guess_path = model_guess_path

		self.model_full = SphericalModel( SpModel_path = self.model_full_path )
		self.model_guess = SphericalModel( SpModel_path = self.model_guess_path )

	def makeDatabase(self, q_values,
		num_iter, lmax,
		save_path,
		beta = 0.0, 
		alpha = 1.0, 
		zeta = 0.0):

		self.alpha = alpha
		self.beta = beta
		self.zeta = zeta

		self.q_values = q_values
		self.lmax = lmax
		self.num_iter = num_iter

		traj_full = mdtraj.load_pdb( self.traj_full_path )
		traj_guess = mdtraj.load_pdb( self.traj_guess_path )

		if type(q_values) == str and q_values == 'all':
			try:
				assert( np.isclose( self.model_guess.q_values, 
					self.model_full.q_values ).all() )
			except AssertionError:
				print( "q_values of the model_full and model_guess do not match. \
					Cannot use all q_values." )
				return
			self.q_values = self.model_full.q_values
			# do nothing and do not slice the models
		else: 
			self.model_full.slice_by_qvalues( q_values=self.q_values, inplace=True )
			self.model_guess.slice_by_qvalues( q_values=self.q_values, inplace=True)
			try:
				assert( np.isclose( self.model_guess.q_values, 
					self.model_full.q_values ).all() )
			except AssertionError:
				print( "q_values of the model_full and model_guess do not match. \
					Failed!!!" )
				return

		# retriever the phases
		self._retrieve_phases()

		# compare guess to full model
		self._compare_full_guess()

		# compute RMSD
		self.rmsd = mdtraj.rmsd( traj_full, traj_guess, frame = 0)

		# save
		self._saveDB( save_path )


	def _retrieve_phases(self):
		# phase retriever obj
		self.pr = PhaseRetriever(self.q_values, self.lmax, 
                               self.model_full.n_theta, 
                               self.model_full.n_phi, 
                               corr = self.model_full.corr, 
                               cospsi = self.model_full.cospsi,
                               ref_SphModel=self.model_guess,
                               auto_only = False)

		self.Sq_guess, self.deltas, _, _ = self.pr.fit(self.num_iter,
	                                       plt_init= False,
	                                       initial_I_guess = self.model_guess.S_q.copy(),
	                                       smooth = False,
	                                       beta = self.beta,
									       alpha = self.alpha,
									       zeta = self.zeta)
	
	def _compare_full_guess(self):
		""" 
		compare guess intensity with the guess intensity and also compute baseline correlation with initial guess
		"""
		all_ls = range(self.lmax)[2:]

		self.corr_by_lmax = np.zeros( (len(all_ls), self.q_values.size) )
		self.corr_by_lmax_baseline = np.zeros((len(all_ls), self.q_values.size))

		for ii, llmax in enumerate(all_ls):
		    new_corr_full = corr_for_new_lmax( self.model_full, llmax, pr = self.pr)
		    new_corr_baseline = corr_for_new_lmax(self.model_full, llmax, model2 = self.model_guess)
		    
		    self.corr_by_lmax[ii] =  new_corr_full 
		    self.corr_by_lmax_baseline[ii] = new_corr_baseline

	def _saveDB(self, path):
		"""
		saves the results of the analysis 
		"""
		ff = h5py.File( path, 'w')

		ff.create_dataset( 'traj_full', data = self.traj_full_path)
		ff.create_dataset( 'traj_guess', data = self.traj_guess_path)
		ff.create_dataset( 'guess_intensity', data = self.Sq_guess)
		ff.create_dataset( 'rmsd', data = self.rmsd)
		ff.create_dataset( 'num_iter', data = self.num_iter)
		ff.create_dataset( 'lmax', data = self.lmax)
		ff.create_dataset( 'learning_rates', data = np.array( [self.alpha,self.beta,self.zeta] ))

		ff.create_dataset( 'qvalues', data = self.q_values)
		ff.create_dataset( ' deltas', data = self.deltas)
		ff.create_dataset( 'corr_by_lmax', data = self.corr_by_lmax)
		ff.create_dataset( 'corr_baseline', data = self.corr_by_lmax_baseline)

		ff.close()
		
	