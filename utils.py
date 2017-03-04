import numpy as np
import mdtraj

def make_perturbed_traj(traj_path, save_path, f_perturb):
	traj = mdtraj.load_pdb(traj_path)
	mean_dist = np.mean( np.sum( (traj.xyz[0] - traj.xyz[0].mean(0))**2, axis = 1), axis = 0)
	x = np.random.rand(traj.xyz.shape[1], traj.xyz.shape[2])*2 - 1
	shifts = x * f_perturb * mean_dist
	traj.xyz += shifts
	traj.save(save_path)
