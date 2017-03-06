import pandas
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
    
    def makeDatabase(self, q_values,
    	num_iter, lmax,
    	save_path,
    	beta, alpha, zeta):
    	
    	self.q_values = q_values
    	self.lmax = lmax
    	self.num_iter = num_iter

    	traj_full = mdtraj.load_pdb(traj_full_path)
    	traj_guess_path = mdtraj.load_pdb(traj_guess_path)

    	self.model_full = SphericalModel( SpModel_path = model_full_path )
    	self.model_guess = SphericalModel( SpModel_path = model_guess_path)

    	# slice the models according to q_values given
    	self.model_full.slice_by_qvalues( q_values=q_values, inplace=True)
		self.model_guess.slice_by_qvalues( q_values=q_values_low, inplace=True)

		# retriever the phases
		self._retrieve_phases()

		# compare guess to full model
		self._compare_full_guess()
		
		# compute RMSD
		self.rmsd = md.rmsd( traj_full, traj_guess, frame = 0)
		
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

		self.Sq_guess, self.deltas, _, _ = pr.fit(self.num_iter,
	                                       plt_init= False,
	                                       initial_I_guess = self.model_guess.S_q.copy(),
	                                       smooth = False)
	
	def _compare_full_guess(self):
		""" 
		compare guess intensity with the guess intensity and also compute baseline correlation with initial guess
		"""
		all_ls = range(self.lmax)[2:]

		self.corr_by_lmax = np.zeros( (len(all_ls), self.q_values.size) )
		self.corr_by_lmax_baseline = np.zeros((len(all_ls), self.q_values.size))

		for ii, llmax in enumerate(all_ls):
		    new_corr_full = corr_for_new_lmax( self.model_full, llmax, pr = self.pr)
		    new_corr_baseline = corr_for_new_lmax(self.model_full, llmax, model2 = model_guess)
		    
		    self.corr_by_lmax[ii] =  new_corr_full 
		    self.corr_by_lmax_baseline[ii] = new_corr_baseline

	def _saveDB(self, path):
		"""
		makes pandas data frame and saves at path
		"""
		pass
	