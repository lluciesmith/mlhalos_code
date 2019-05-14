"""
Classify particles of this L=50 Mpc a N=512 box in /share/data1/lls/pioneer50/
with trained random forest from original WMAP5 L=50 Mpc h^-1 a box.

"""

import numpy as np
from sklearn.externals import joblib

if __name__ == "__main__":
    # Trained classifier for density features and shear features
    path_classifier = "/share/data1/lls/shear_quantities/"

    # DENSITY-ONLY CLASSIFICATION

    clf_density = joblib.load(path_classifier + "classifier_den/clf_upgraded.pkl")

    density_features = np.lib.format.open_memmap("/share/data1/lls/pioneer50/features_3e10/density_contrasts.npy",
                                                 mode="r", shape=(512**3, 50))

    b = np.array_split(np.arange(512**3), 10000, axis=0)

    for i in range(10000):
        d = density_features[b[i], :]
        pred_i = clf_density.predict_proba(d)
        np.save("/share/data1/lls/pioneer50/features_3e10/predictions/densities/pred_" + str(i) + ".npy", pred_i)
        del d
        del pred_i

    pred_all = np.zeros((512**3, 2))
    for i in range(10000):
        a = np.load("/share/data1/lls/pioneer50/features_3e10/predictions/densities/pred_" + str(i) + ".npy")
        pred_all[b[i], :] = a
        del a

    np.save("/share/data1/lls/pioneer50/features_3e10/predictions/density_predicted_probabilities.npy", pred_all)