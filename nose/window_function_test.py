import numpy as np
from mlhalos import window
from mlhalos import parameters
import pynbody


path = "/Users/lls/Documents/CODE"


def test_mass_assignment_sharp_k():
    r = pynbody.array.SimArray([20])
    r.units = "Mpc a h^-1"
    ic = parameters.InitialConditionsParameters(path=path)

    w_sk = window.WindowParameters(initial_parameters=ic, volume="sharp-k")
    mass_sk = w_sk.get_mass_from_radius(ic, r, ic.mean_density)

    w_th = window.WindowParameters(initial_parameters=ic, volume="sphere")
    mass_th = w_th.get_mass_from_radius(ic, r, ic.mean_density)

    assert mass_sk == (9 * np.pi /2) * mass_th


def test_radius_assignment_sharp_k():
    m = pynbody.array.SimArray([1e13])
    m.units = "Msol h^-1"
    ic = parameters.InitialConditionsParameters(path=path)

    w_sk = window.WindowParameters(initial_parameters=ic, volume="sharp-k")
    r_sk = w_sk.get_smoothing_radius_corresponding_to_filtering_mass(ic, m)

    w_th = window.WindowParameters(initial_parameters=ic, volume="sphere")
    r_th = w_th.get_smoothing_radius_corresponding_to_filtering_mass(ic, m)
    r_sk_test = (2 / (9 * np.pi))**(1/3) * r_th

    assert np.allclose(r_sk, r_sk_test), "Th sharp k radius is" + str(r_sk) + "and the testing one is " + str(r_sk_test)

#
# def test_mass_assignment_top_hat():
#     r = pynbody.array.SimArray([20])
#     r.units = "Mpc a h^-1"
#     ic = parameters.InitialConditionsParameters(path=path)
#
#     w_sk = window.WindowParameters(initial_parameters=ic, volume="sharp-k")
#     r_sk = w_sk.get_smoothing_radius_corresponding_to_filtering_mass(ic, m)
#
#     w_th = window.WindowParameters(initial_parameters=ic, volume="sphere")
#     r_th = w_th.get_smoothing_radius_corresponding_to_filtering_mass(ic, m)
#     r_sk_test = (2 / (9 * np.pi))**(1/3) * r_th
#
#     assert np.allclose(r_sk, r_sk_test), "Th sharp k radius is" + str(r_sk) + "and the testing one is " + str(r_sk_test)


def test_correct_units():
    initial_parameters = parameters.InitialConditionsParameters(path=path)
    snapshot = initial_parameters.initial_conditions
    smoothing_radii = np.linspace(0.0057291, 0.20, 1500)

    r_a_h = smoothing_radii/snapshot.properties['a'] * snapshot.properties['h']
    r_a_h = r_a_h.view(pynbody.array.SimArray)
    r_a_h.units = "Mpc a h^-1"

    th = pynbody.analysis.hmf.TophatFilter(snapshot)
    m = th.R_to_M(r_a_h)
    m.units = "Msol h^-1"


    rho_M = pynbody.analysis.cosmology.rho_M(snapshot, unit="Msol Mpc**-3")
    w = window.WindowParameters(initial_parameters=initial_parameters, volume="sphere")
    filtering_masses = w.get_mass_from_radius(initial_parameters, smoothing_radii, rho_M)
    filtering_masses_with_h = filtering_masses * snapshot.properties['h']
    filtering_masses_with_h.units = "Msol h^-1"

    np.testing.assert_allclose(filtering_masses_with_h, m)


def test_flattening():
    a = np.linspace(0,10,10)
    b = np.linspace(10, 3, 10)
    c = np.linspace(220, 2210, 10)

    B = np.zeros((len(a), len(a), len(a)))
    for i in range(len(a)):
        for j in range(len(a)):
            for k in range(len(a)):
                B[i,j,k] = a[i] + b[j] + c[k]

    B1 = np.array([a[i] + b[j] + c[k] for i in range(len(a)) for j in range(len(a)) for k in range(len(a))])
    np.allclose(B.flatten(), B1)





