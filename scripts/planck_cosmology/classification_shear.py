"""
Classify particles of this L=50 Mpc a N=512 box in /share/data1/lls/pioneer50/
with trained random forest from original WMAP5 L=50 Mpc h^-1 a box.

"""

import numpy as np
from sklearn.externals import joblib

if __name__ == "__main__":
    # Trained classifier for density features and shear features
    path_classifier = "/share/data1/lls/shear_quantities/"
    path_pionner = "/share/data1/lls/pioneer50/features_3e10/"

    # DENSITY-ONLY CLASSIFICATION

    clf_density_plus_shear = joblib.load(path_classifier + "classifier_den+den_sub_ell+den_sub_prol/clf_upgraded.pkl")

    density_features = np.lib.format.open_memmap(path_pionner + "density_contrasts.npy",
                                                 mode="r", shape=(512**3, 50))
    ell = np.lib.format.open_memmap(path_pionner + "density_subtracted_ellipticity.npy",
                                    mode="r", shape=(512**3, 50))
    prol = np.lib.format.open_memmap(path_pionner + "density_subtracted_prolateness.npy",
                                     mode="r", shape=(512**3, 50))
    b = np.array_split(np.arange(512**3), 10000, axis=0)

    for i in range(10000):
        d = density_features[b[i], :]
        e = ell[b[i], :]
        p = prol[b[i], :]

        features_i = np.column_stack((d, e, p))

        pred_i = clf_density_plus_shear.predict_proba(features_i)
        np.save(path_pionner + "predictions/den_shear/pred_" + str(i) + ".npy", pred_i)
        del d
        del pred_i

    pred_all = np.zeros((512**3, 2))
    for i in range(10000):
        a = np.load(path_pionner + "predictions/den_shear/pred_" + str(i) + ".npy")
        pred_all[b[i], :] = a
        del a

    np.save(path_pionner + "predictions/shear_density_predicted_probabilities.npy", pred_all)