import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml

path = "/share/data1/lls/shear_quantities/"


def get_ten_sets_feature_importances(density, ellipticity, prolateness):
    f_imp = np.zeros((10, 150))

    a = np.arange(256 ** 3)
    for i in range(10):
        training_index = np.random.choice(a, 50000)
        features = np.column_stack((density[training_index, :-1], ellipticity[training_index, :-1],
                                    prolateness[training_index, :]))

        rf = ml.MLAlgorithm(features, algorithm='Random Forest', split_data_method=None, cross_validation=False,
                            num_cv_folds=5, n_jobs=60, max_features="auto")
        f_imp[i] = rf.feature_importances
    return f_imp




######### importances of 10 runs for density + density-subtracted ellipticity + den_sub_prolateness ##########
num_ids = 256**3

den_features = np.lib.format.open_memmap(path + "density_features.npy", mode="r", shape=(num_ids, 51))
den_sub_ell = np.lib.format.open_memmap(path + "den_sub_ellipticity_features.npy", mode="r", shape=(num_ids, 51))
den_sub_prol = np.lib.format.open_memmap(path + "den_sub_prolateness_features.npy", mode="r", shape=(num_ids, 51))

importances = get_ten_sets_feature_importances(den_features, den_sub_ell, den_sub_prol)
np.save(path + "importances_10_runs_den+den_sub_ell+den_sub_prol_max_feat_auto.npy", importances)
