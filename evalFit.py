# analysis tools
import numpy as np
import shtns

def compute_corr(x, y):
	"""
	statistical correlation between x and y
	"""

	xx = x.copy()
	yy = y.copy()

	xx -= xx.mean()
	yy -= yy.mean()

	return np.mean(xx*yy)/np.sqrt(xx.var()*yy.var())


def get_Sq_with_reduced_lmax(new_lmax, model=None,pr = None):
	"""
	model: SphericalModel object
	pr: PhaseRetriver object
	"""
	new_sh_obj = shtns.sht(new_lmax)


	if model is None:
	    old_sh_obj = pr.sh
	    new_S_q = np.zeros_like(pr.I_guess)
	    old_all_slm = pr.all_slm_guess
	    new_sh_obj.set_grid( pr.n_theta, pr.n_phi )
	    n_q = pr.n_q
	else:
	    old_sh_obj = model.sh
	    new_S_q = np.zeros_like(model.S_q)
	    old_all_slm = model.all_slm
	    new_sh_obj.set_grid( model.n_theta, model.n_phi )
	    n_q = model.n_q
	    
	for qid in range(n_q):
	    new_slm = new_sh_obj.spec_array()
	    for lid in range(new_lmax+1):
	        for mid in range(lid+1):
	            new_slm[new_sh_obj.idx(lid, mid)] = old_all_slm[qid][old_sh_obj.idx(lid,mid)]
	    new_S_q[qid] = new_sh_obj.synth(new_slm)

	return new_S_q

def corr_for_new_lmax(model, new_lmax, model2=None,pr=None):
	"""
	model, model2: SphericalModel object
	pr: PhaseRetriver object
	"""

	if model2 is None:
	    new_I_guess = get_Sq_with_reduced_lmax( new_lmax, pr = pr)
	else:
	    new_I_guess = get_Sq_with_reduced_lmax( new_lmax, model=model2)
	    
	new_I_model = get_Sq_with_reduced_lmax( new_lmax, model = model)

	new_corr = np.array( [ compute_corr(new_I_model[iq],new_I_guess[iq]) for iq in range(new_I_guess.shape[0]) ] )

	return new_corr