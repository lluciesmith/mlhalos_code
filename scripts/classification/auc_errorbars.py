import sys
sys.path.append("/home/lls/mlhalos_code/scripts")
import numpy as np
from mlhalos import machinelearning as ml

density_shear_features = np.load("/home/lls/stored_files/shear_and_density/density_shear_features.npy")

aucs = np.zeros((10, 2))

for i in range(10):
    index_training_i = np.random.choice(range(len(density_shear_features)), 100000)
    features = density_shear_features[index_training_i]

    trained_algo = ml.MLAlgorithm(features, split_data_method="train_test_split", train_size=50000, n_jobs=60)

    print(trained_algo.classifier.best_params_)

    auc_validation = trained_algo.classifier.best_score_
    auc_test = ml.get_auc_score(trained_algo.predicted_proba_test, trained_algo.true_label_test)

    aucs[i, 0] = auc_validation
    aucs[i, 1] = auc_test

np.save("/home/lls/stored_files/shear_and_density/aucs_val_test.npy", aucs)
