import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import machinelearning as ml

path = "/share/data1/lls/shear_quantities/"


def get_ten_sets_feature_importances(density):
    f_imp = np.zeros((10, 50))

    a = np.arange(256 ** 3)
    for i in range(10):
        training_index = np.random.choice(a, 50000)
        features = density[training_index, :]

        rf = ml.MLAlgorithm(features, algorithm='Random Forest', split_data_method=None, cross_validation=False,
                            n_jobs=60, max_features="auto")
        #print(rf.best_estimator)
        f_imp[i] = rf.feature_importances
    return f_imp




######### importances of 10 runs for density + density-subtracted ellipticity + den_sub_prolateness ##########
num_ids = 256**3

den_features = np.lib.format.open_memmap(path + "density_features.npy", mode="r", shape=(num_ids, 51))
importances = get_ten_sets_feature_importances(den_features)
np.save(path + "importances_10_runs_density_only_less_estimators.npy", importances)