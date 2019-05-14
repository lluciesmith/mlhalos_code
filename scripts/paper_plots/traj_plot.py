import matplotlib
matplotlib.rcParams.update({'axes.labelsize': 18})
from mlhalos import plot
import importlib
importlib.reload(plot)
from mlhalos import parameters
from mlhalos import window
import numpy as np
import matplotlib.pyplot as plt


############  DENSITY TRAJECTORIES  #############


def plot_density_trajectories(mass_scales, traj_in, traj_out):

    plot.plot_trajectories(mass_scales, traj_in, traj_out, num_particles=None, parameters=None,
                           max_number_trajectories=20, threshold=False, convergence=True, num_trajectory="multiple",
                           in_label= "IN class", out_label="OUT class")


if __name__ == "__main__":
    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
    w = window.WindowParameters(initial_parameters=ic)

    den_f = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/features/density_features.npy")
    #den_f = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/density_trajectories.npy")

    in_traj = np.random.choice(np.where(den_f[:, -1]== 1)[0], 4)
    print(in_traj)
    out_traj = np.random.choice(np.where(den_f[:, -1]== -1)[0], 4)
    print(out_traj)

    plot_density_trajectories(w.smoothing_masses, den_f[in_traj, :-1], den_f[out_traj, :-1])
    plt.savefig("/Users/lls/Documents/mlhalos_paper/trajectories.pdf")

    # TO DO LIST:
    #
    # CHANGE X- LABEL TO M_smoothing (implemented in `plot.py`)
    # CHECK IF THE TWO COLOURS ARE DISTINGUISHABLE IN BLACK AND WHITE
