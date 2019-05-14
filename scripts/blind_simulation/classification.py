"""
Classify particles with trained random forest from other L=50 Mpc h^-1 a box.

"""

import numpy as np
from sklearn.externals import joblib

if __name__ == "__main__":
    # Trained classifier for density features and shear features
    path_classifier = "/share/data1/lls/shear_quantities/"

    # DENSITY-ONLY CLASSIFICATION

    density_features = np.load("/share/data1/lls/reseed50/features/density_contrasts.npy")
    clf_density = joblib.load(path_classifier + "classifier_den/clf_upgraded.pkl")

    y_predicted_density = clf_density.predict_proba(density_features)
    # np.save("/share/data1/lls/reseed50/predictions/density_predicted_probabilities.npy", y_predicted_density)

    del y_predicted_density
    del clf_density

    # DENSITY + SHEAR CLASSIFICATION

    ell = np.load("/share/data1/lls/reseed50/features/density_subtracted_ellipticity.npy")
    prol = np.load("/share/data1/lls/reseed50/features/density_subtracted_prolateness.npy")
    features = np.column_stack((density_features, ell, prol))

    del ell
    del prol
    del density_features

    clf_density_plus_shear = joblib.load(path_classifier + "classifier_den+den_sub_ell+den_sub_prol/clf_upgraded.pkl")

    y_predicted_shear_density = clf_density_plus_shear.predict_proba(features)
    # np.save("/share/data1/lls/reseed50/predictions/shear_density_predicted_probabilities.npy",
    #         y_predicted_shear_density)

