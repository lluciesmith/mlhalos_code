import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.externals import joblib

rf_CV = joblib.load("/share/data1/lls/regression/in_halos_only/log_m_output/larger_training_set/clf.pkl")
halo_mass_testing = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/larger_training_set/"
                            "halos_testing_set_large.npy")

# predictions

dup = np.copy(halo_mass_testing)
dup1 = np.tile(dup, (50, 1)).transpose()

noise_04 = np.random.normal(0, 2.7, size=[len(halo_mass_testing), 50])
test_feat_04_corr = dup1 + noise_04

noise_07 = np.random.normal(0, 1.2, [len(halo_mass_testing), 50])
test_feat_07_corr = dup1 + noise_07

pred = rf_CV.predict(np.column_stack((test_feat_07_corr, test_feat_04_corr)))
np.save("/share/data1/lls/regression/in_halos_only/log_m_output/larger_training_set/predicted_halos_test_correct.npy",
        pred)