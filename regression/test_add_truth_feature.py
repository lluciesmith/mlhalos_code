import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml

# Get training set

saving_path = "/share/data2/lls/regression/test_truth_feature/"
path_features = "/share/data2/lls/features_w_periodicity_fix/"

training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")

ics_den_features = np.lib.format.open_memmap(path_features + "ics_density_contrasts.npy", mode="r", shape=(256**3, 50))
ics_den_training = ics_den_features[training_ids]
ics_den_testing = ics_den_features[testing_ids]
del ics_den_features


halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
log_mass = np.log10(halo_mass[training_ids])
halo_mass_testing = np.log10(halo_mass[testing_ids])
del halo_mass


# train on density + true halo mass

training_features = np.column_stack((ics_den_training, log_mass, log_mass))
print(training_features.shape)

cv = True
third_features = int((training_features.shape[1] -1)/3)
param_grid = {"n_estimators": [1000, 1300],
              "max_features": [third_features, "sqrt", 25, 40],
              "min_samples_leaf": [5, 15],
              #"criterion": ["mse", "mae"],
              }

clf = ml.MLAlgorithm(training_features, method="regression", cross_validation=cv, split_data_method=None, n_jobs=60,
                     save=True, path=saving_path + "classifier/classifier.pkl", param_grid=param_grid)
if cv is True:
    print(clf.best_estimator)
    print(clf.algorithm.best_params_)
    print(clf.algorithm.best_score_)

np.save(saving_path + "f_imp.npy", clf.feature_importances)

# test

X_test = np.column_stack((ics_den_testing, halo_mass_testing))
y_predicted = clf.algorithm.predict(X_test)
np.save(saving_path + "predicted_log_halo_mass.npy", y_predicted)



###### PLOT RESULTS ######

from regression.plots import plotting_functions as pf

y_w_truth_feat = np.load("/Users/lls/Documents/mlhalos_files/truth_feature/predicted_log_halo_mass.npy")
y_den = np.load("/Users/lls/Documents/mlhalos_files/den_only_periodicity_fix/predicted_log_halo_mass.npy")
x = np.load("/Users/lls/Documents/mlhalos_files/lowz_density/true_mass_test_set.npy")

bins_plotting = np.linspace(x.min(), x.max(), 15, endpoint=True)

# VIOLINS

pf.compare_violin_plots(y_w_truth_feat, x, y_den, x,
                        bins_plotting, label1="ics density + truth", label2="ics density", color1="g",
                        color2="r")


# 2D HISTOGRAM

pf.compare_2d_histograms(x, y_w_truth_feat, x, y_den,
                         title1="ics density + truth", title2="ics density", save_path=None)