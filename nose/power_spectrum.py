import numpy as np

import scripts.ellipsoidal.power_spectrum
from scripts.ellipsoidal import ellipsoidal_barrier as eb
from mlhalos import parameters
from scripts.hmf import super_sampling as ssc


path = "/Users/lls/Documents/CODE"


def test_power_spectrum():
    initial_parameters = parameters.InitialConditionsParameters(path=path)
    k_vector = np.linspace(0.1, 1000, 100000)

    pwspectrum = scripts.ellipsoidal.power_spectrum.get_power_spectrum("WMAP5", initial_parameters, z=99)
    pk = pwspectrum(k_vector) / pwspectrum._lingrowth

    pk_0 = scripts.ellipsoidal.power_spectrum.get_power_spectrum("WMAP5", initial_parameters, z=0)
    np.testing.assert_allclose(pk_0(k_vector), pk)


def test_k_computation():
    L = 50
    shape = 256

    k_i = ssc.k_1D(L, shape)
    K = np.zeros((shape,shape,shape))
    for i in range(len(k_i)):
        for j in range(len(k_i)):
            for z in range(len(k_i)):
                K[i, j, z] = np.sqrt(k_i[i] ** 2 + k_i[j] ** 2 + k_i[z] ** 2)
    K = np.zeros((shape, shape, shape))
    K[0,0,0] = 1

    k_saved = ssc.get_all_ks(shape, L)
    np.testing.assert_allclose(k_saved, K)
