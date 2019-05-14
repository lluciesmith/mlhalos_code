import sys
sys.path.append("/home/lls/mlhalos_code/scripts")
from sklearn.externals import joblib
import numpy as np


clf = joblib.load("/home/lls/stored_files/shear_and_density/full_eigenvalues/not_rescaled/classifier/classifier.pkl")

trees = clf.best_estimator_.estimators_

imp = np.array([tree.feature_importances_ for tree in trees])

np.save("/home/lls/stored_files/shear_and_density/full_eigenvalues/not_rescaled/trees_importances.npy", imp)


# # Test on remaining particles in Nina's simulation
#
# test_N = np.load("/home/lls/stored_files/features_test.npy")
#
# pred_N = clf.classifier.predict_proba(test_N[:, :-1])
# true_N = test_N[:, -1]
#
# np.save("/home/lls/stored_files/classifier/results/pred_Nina.npy", pred_N)
# np.save("/home/lls/stored_files/classifier/results/true_Nina.npy", true_N)
#
# del test_N
# del pred_N
# del true_N
#
#
# # Test on particles in Andrew's simulation
#
# # Trajectory features
#
# test_A = np.load("/home/lls/stored_files/rsim/features.npy")
#
# pred_A = clf.classifier.predict_proba(test_A[:, :-1])
# true_A = test_A[:, -1]
#
# np.save("/home/lls/stored_files/classifier/results/pred_Andrew.npy", pred_A)
# np.save("/home/lls/stored_files/classifier/results/true_Andrew.npy", true_A)
#
# del test_A
# del pred_A
# del true_A
#
# # Trajectory + EPS label feature
#
# test_A_w_EPS = np.load("/home/lls/stored_files/rsim/with_EPS_label/features_w_EPS_label.npy")
#
# pred_A_w_EPS = clf.classifier.predict_proba(test_A_w_EPS[:, :-1])
# true_A_w_EPS = test_A_w_EPS[:, -1]
#
# np.save("/home/lls/stored_files/classifier/results/pred_w_EPS_Andrew.npy", pred_A_w_EPS)
# np.save("/home/lls/stored_files/classifier/results/true_w_EPS_Andrew.npy", true_A_w_EPS)
#
# del test_A_w_EPS
# del pred_A_w_EPS
# del true_A_w_EPS
#
