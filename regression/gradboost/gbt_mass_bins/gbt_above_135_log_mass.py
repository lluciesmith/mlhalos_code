import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from mlhalos import plot
import matplotlib.pyplot as plt

def plot_importances(imp, label="1000 trees, depth 3", m=None, width=None, title=r"\log M > 13.5$"):
    if m is None:
        m = np.linspace(np.log10(3e10), np.log10(1e15), 50)[:-1]
        width = np.append(np.diff(m), np.diff(m)[-1])[:-1]

    plot.plot_importances_vs_mass_scale(imp, 10 ** m, width=width, label=label,
                                        title=title, subplots=1, figsize=(10, 5))
    plt.axvline(x=10 ** 13.5, color="grey", ls="--")
    plt.axvline(x=10**14.617934795177572, color="grey", ls="--")
    plt.legend(loc="best", fontsize=16)
    # plt.ylim(0, 0.2)

traj = np.load("/Users/lls/Documents/mlhalos_files/regression/features_w_periodicity_fix/ics_density_contrasts.npy")
training_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/"
                       "ic_traj/nest_2000_lr006/training_ids.npy")
halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")

ids_above_135 = np.where(np.log10(halo_mass[training_ids]) > 13.5)[0]


features_training = traj[training_ids[ids_above_135], :-1]
truth_training = np.log10(halo_mass[training_ids[ids_above_135]])

clf_10 = GradientBoostingRegressor(n_estimators=1000, max_features=0.8, subsample=0.8, learning_rate=0.01,
                                   max_depth=3, loss="lad")
clf_10.fit(features_training, truth_training)

plot_importances(clf_10.feature_importances_)
plt.subplots_adjust(bottom=0.15, top=0.9)
plt.savefig("/Users/lls/Desktop/log_m_above_135_bug_fixed_sklearn_depth3.png")
