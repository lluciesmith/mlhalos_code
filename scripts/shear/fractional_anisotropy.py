import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
from mlhalos import shear
import matplotlib.pyplot as plt

path = "/home/lls/stored_files"
number_of_cores = 60

initial_parameters = parameters.InitialConditionsParameters(path=path)

s = shear.Shear(initial_parameters=initial_parameters, shear_scale="all", number_of_processors=60)
eigenvalues = s.shear_eigenvalues
subtracted_eigenvalues = s.density_subtracted_eigenvalues
np.save("/home/lls/stored_files/shear/all_eigenvalues.npy", eigenvalues)
np.save("/home/lls/stored_files/shear/all_density_subtracted_eigenvalues.npy", subtracted_eigenvalues)

fa = np.zeros((s.shape**3, len(s.shear_scale)))

for i in range(len(s.shear_scale)):
    eig_i = eigenvalues[:, int(3*i): int(3*i) + 3]
    fa[:, i] = shear.fractional_anisotropy(eig_i)

np.save("/home/lls/stored_files/shear/fractional_anisotropies.npy", fa)

def plot():
    for i in range(50):
        plt.hist(fa_in[:,i], normed=True, bins=30, histtype="step", color="b")
        plt.hist(fa_out[:, i], normed=True, bins=30, histtype="step", color="g")
        plt.title("Fractional Anisotropy " + str(i))
        plt.tight_layout()
        plt.show()