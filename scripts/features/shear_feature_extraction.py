import sys
sys.path.append("/Users/lls/Documents/mlhalos_code/")
import numpy as np
from mlhalos import parameters


def transform_in_features(property, initial_conditions, save=True, path=None):
    property_in = np.column_stack((property[initial_conditions.ids_IN], np.ones(len(initial_conditions.ids_IN), )))
    property_out = np.column_stack((property[initial_conditions.ids_OUT], np.ones(len(initial_conditions.ids_OUT), ) * -1))

    del property

    property_features = np.concatenate((property_in, property_out))

    del property_in
    del property_out
    return property_features


if __name__ == "__main__":

    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")


    # From eigenvalues to eigenvalues features (with label)

    eigenvalues = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/all_eigenvalues.npy")
    eigenvalues_features = transform_in_features(eigenvalues, ic, save=True)
    np.save("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/features/eigenvalues_features.npy",
            eigenvalues_features)
    del eigenvalues_features

    # From density-subtracted eigenvalues to density-subtracted eigenvalues features (with label)

    den_sub_eigenvalues = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities"
                          "/all_density_subtracted_eigenvalues.npy")
    den_sub_eigenvalues_features = transform_in_features(den_sub_eigenvalues, ic, save=True)
    np.save("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/features/den_sub_eigenvalues_features.npy",
            den_sub_eigenvalues_features)
    del den_sub_eigenvalues_features

    # From ellipticity to ellipticity features (with label)

    ellipticity = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/ellipticity.npy")
    ellipticity_features = transform_in_features(ellipticity, ic, save=True)
    np.save("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/features/ellipticity_features.npy",
            ellipticity_features)
    del ellipticity_features

    # From prolateness to prolateness features (with label)

    prolateness = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/prolateness.npy")
    prolateness_features = transform_in_features(prolateness, ic, save=True)
    np.save("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/features/prolateness_features.npy",
            prolateness_features)
    del prolateness_features

    # From density-subtracted ellipticity to density-subtracted ellipticity features (with label)

    den_sub_ellipticity = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities"
                                  "/density_subtracted_ellipticity.npy")
    den_sub_ellipticity_features = transform_in_features(den_sub_ellipticity, ic, save=True)
    np.save("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/features/den_sub_ellipticity_features.npy",
            den_sub_ellipticity_features)
    del den_sub_ellipticity_features


    # From density-subtracted prolateness to density-subtracted prolateness features (with label)

    den_sub_prolateness = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities"
                                  "/density_subtracted_prolateness.npy")
    den_sub_prolateness_features = transform_in_features(den_sub_prolateness, ic, save=True)
    np.save("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/features/den_sub_prolateness_features.npy",
            den_sub_prolateness_features)
    del den_sub_prolateness_features





