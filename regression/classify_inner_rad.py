"""
Train and classify only r/r_vir<0.3 particles in halos - test whether there
is an improvement over all particles in the simulation.
Construct training set s.t. you have 2000 particles in each halo mass bin
and 5000 particles in the very last halo bin.

"""

import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml

radii_path = "/home/lls/stored_files/radii_stuff/"
saving_path = "/share/data1/lls/regression/in_halos_only/log_m_output/inner_rad/"

# Select only particles which are within 30% of virial radius

radii_properties_in = np.load(radii_path + "radii_properties_in_ids.npy")
radii_properties_out = np.load(radii_path + "radii_properties_out_ids.npy")

fraction = np.concatenate((radii_properties_in[:,2],radii_properties_out[:,2]))
ids_in_halo = np.concatenate((radii_properties_in[:,0],radii_properties_out[:,0]))

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
traj = np.load("/share/data1/lls/shear_quantities/quantities_id_ordered/density_trajectories.npy")

inner_ids = ids_in_halo[fraction < 0.3]
inner_ids = inner_ids.astype("int")
halo_mass_inner = halo_mass[inner_ids]

# Select a balanced training set

n, log_bins = np.histogram(np.log10(halo_mass[halo_mass>0]), bins=50)
bins = 10**log_bins

training_ind = []
for i in range(len(bins) - 1):
    p_in_bins = inner_ids[(halo_mass_inner >= bins[i]) & (halo_mass_inner < bins[i + 1])]
    if p_in_bins.size == 0:
        print("Pass")
        pass
    elif i == 49:
        ind = np.random.choice(p_in_bins, 5000, replace=False)
        training_ind.append(ind)
    else:
        ind = np.random.choice(p_in_bins, 2000, replace=False)
        training_ind.append(ind)

training_ind = np.concatenate(training_ind)

np.save(saving_path + "training_ids.npy", training_ind)

testing_indices = ~np.in1d(inner_ids, training_ind)
testing_ids = inner_ids[testing_indices]
np.save(saving_path + "testing_ids.npy", testing_ids)
y_test = halo_mass[testing_ids]
np.save(saving_path + "true_halo_mass.npy", y_test)


##### TRAINING + PREDICTING #######

log_mass = np.log10(halo_mass)

feat_training = np.column_stack((traj[training_ind], log_mass[training_ind]))
X_test = traj[testing_ids]

cv = True
clf = ml.MLAlgorithm(feat_training, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60, save=True,
                     path=saving_path + "classifier/classifier.pkl")
if cv is True:
    print(clf.best_estimator)
    print(clf.algorithm.best_params_)
    print(clf.algorithm.best_score_)
np.save(saving_path + "f_imp.npy", clf.feature_importances)

# classify

y_predicted = clf.algorithm.predict(X_test)
np.save(saving_path + "predicted_halo_mass.npy", 10**y_predicted)



def scatter_plot():
    for i in range(len(indices_each_bin)):
        ind = indices_each_bin[i]
        predicted_each_bin = log_predicted_mass[ind]

        ran = np.random.choice(ind, 10000, replace=False)
        plt.errorbar(np.log10(np.mean(true_mass_test[ind])), np.log10(np.mean(predicted_mass[ind])),
                     yerr=np.std(np.log10(predicted_mass[ind]))
                     )
        plt.scatter(np.log10(true_mass_test[ran]), np.log10(predicted_mass[ran]), alpha=0.2, s=0.1)
    xrange = np.linspace(10, 15, 10)
    plt.plot(xrange, xrange)
    #plt.xscale("log")
    #plt.yscale("log")
    plt.xlabel("True mass")
    plt.ylabel("Predicted mass")
