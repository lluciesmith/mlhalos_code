"""
Test what happens when you increase the number of training examples.
In particular, in increasing the number of particles at the edges of the distribution.

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from mlhalos import machinelearning as ml
from sklearn.externals import joblib
from sklearn.metrics import r2_score


saving_path = "/share/data2/lls/regression/adaboost/ic_traj_training_2/"

# data

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
traj = np.load("/share/data2/lls/features_w_periodicity_fix/ics_density_contrasts.npy")

training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
traj_training = traj[training_ids, :]
log_halo_training = np.log10(halo_mass[training_ids])

try:
    random_test_set = np.load("/share/data2/lls/regression/adaboost/ic_traj_training/test_ids.npy")
    log_halo_testing = np.load("/share/data2/lls/regression/adaboost/ic_traj_training/halo_mass_test_ids.npy")

except:
    testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")
    random_test_set = np.random.choice(testing_ids, 200000, replace=False)
    np.save(saving_path + "test_ids.npy", random_test_set)

    log_halo_testing = np.log10(halo_mass[random_test_set])
    np.save(saving_path + "halo_mass_test_ids.npy", log_halo_testing)

traj_testing = traj[random_test_set, :]

del traj
del halo_mass

# adaboost predictions

# param_grid = {"n_estimators": [500, 1000],
#               "learning_rate": [0.001, 0.005, 0.01],
#               "loss": ["linear", "exponential"]
#               }
param_grid = {"n_estimators": [2000, 3000, 4000],
              "learning_rate": [0.001, 0.01],
              "base_estimator__max_depth": [2, 5]
              }
base_estimator = DecisionTreeRegressor(max_depth=5)
ada_b = AdaBoostRegressor(base_estimator=base_estimator, random_state=20)
ada_CV = GridSearchCV(estimator=ada_b, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1,
                      scoring="r2")

ada_CV.fit(traj_training, log_halo_training)

ml.write_to_file_cv_results(saving_path + "/cv_results.txt", ada_CV)
joblib.dump(ada_CV, saving_path + "clf.pkl")

# predictions

pred_test = ada_CV.predict(traj_testing)
np.save(saving_path + "predicted_test_set.npy", pred_test)

ada_r2_train = np.zeros(len(ada_CV.best_estimator_.estimators_),)
for i, y_pred in enumerate(ada_CV.best_estimator_.staged_predict(traj_training)):
    ada_r2_train[i] = r2_score(log_halo_training, y_pred)

np.save(saving_path + "r2_train_staged_scores.npy", ada_r2_train)

ada_r2_test = np.zeros(len(ada_CV.best_estimator_.estimators_),)
for i, y_pred in enumerate(ada_CV.best_estimator_.staged_predict(traj_testing)):
    ada_r2_test[i] = r2_score(log_halo_testing, y_pred)

np.save(saving_path + "r2_test_staged_scores.npy", ada_r2_test)
#
# scp cv_results.txt halo_mass_test_ids.npy predicted_test_set.npy r2_train_staged_scores.npy r2_test_staged_scores.npy
# lls@chewbacca.star.ucl.ac.uk:/Users/lls/Documents/mlhalos_files/regression/adaboost
# /ic_traj_training/

#################### TO RUN ON CHEWBACCA ########################

import numpy as np
import matplotlib.pyplot as plt
from regression.plots import plotting_functions as pf


path = "/Users/lls/Documents/mlhalos_files/regression/adaboost/ic_traj_training/"
log_mass_testing = np.load(path + "halo_mass_test_ids.npy")
pred_mass_test = np.load(path + "predicted_test_set.npy")

bins_plotting = np.linspace(log_mass_testing.min(), log_mass_testing.max(), 15, endpoint=True)
pf.get_violin_plot(bins_plotting, pred_mass_test, log_mass_testing)

r2_train = np.load(path + "r2_train_staged_scores.npy")
r2_test = np.load(path + "r2_test_staged_scores.npy")

plt.figure()
plt.plot(np.arange(len(r2_train)), r2_train, label="r$2$ train")
plt.plot(np.arange(len(r2_train)), r2_test, label="r$2$ test")
plt.xlabel("Num estimators")
plt.ylabel("r$2$")
plt.legend(loc="best")

