import numpy as np
import pynbody
import sys
import matplotlib.pyplot as plt
# camb_path = "/home/lls/stored_files/"
camb_path = "/Users/lls/Software/CAMB-Jan2017/"
mlhalos_path = "/Users/lls/Documents/mlhalos_code"
sys.path.append(mlhalos_path)
sys.path.append(camb_path)
import pycamb.camb as camb


def power_spectrum_from_CAMB_WMAP5(snapshot, save=True, path=None, omcdm=0.234, omb=0.045, camb_path=None):
    #import pycamb.camb as camb

    h = snapshot.properties['h']
    H0 = h * 100
    omch2 = omcdm * (h**2)
    ombh2 = omb * (h**2)
    assert omcdm + omb == snapshot.properties['omegaM0']
    omlambdah2 = snapshot.properties['omegaL0'] * (h**2)

    # Define new parameters instance with WMAP5 cosmological parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2,
                       #mnu=0,
                       #sigma8=0.817
                       )
    pars.set_matter_power(kmax=0.11552E+05, k_per_logint=None, silent=False)
    pars.InitPower.set_params(ns=0.96)
    pars.validate()

    # Now get linear matter power spectrum at redshift 0

    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=0.71023E-04, maxkh=0.11552E+05, npoints=150)
    powerspec = np.column_stack((kh, pk[0]))

    if save is True:
        if path is None:
            path = camb_path
        np.savetxt(path + "luisa/camb_Pk_WMAP5", powerspec)
    else:
        return kh, pk


def power_spectrum_from_CAMB_Planck2015(snapshot, save=True, path=None, omcdm=0.2595, omb=0.0491, camb_path=None):
    #import pycamb.camb as camb

    h = snapshot.properties['h']
    H0 = h * 100
    omch2 = omcdm * (h**2)
    ombh2 = omb * (h**2)
    assert omcdm + omb == snapshot.properties['omegaM0']
    # omlambdah2 = snapshot.properties['omegaL0'] * (h**2)

    # Define new parameters instance with WMAP5 cosmological parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2,
                       #mnu=0,
                       #sigma8=0.817
                       )
    pars.set_matter_power(kmax=0.11552E+05, k_per_logint=None, silent=False)
    pars.InitPower.set_params(ns=0.96)
    pars.validate()

    # Now get linear matter power spectrum at redshift 0

    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(npoints=150)
    powerspec = np.column_stack((kh, pk[0]))

    if save is True:
        if path is None:
            path = camb_path
        np.savetxt(path + "luisa/camb_Pk_PLANCK2015", powerspec)
    else:
        return kh, pk


def get_power_spectrum(cosmology, ic, z=99, camb_path=None):
    if camb_path is None:
        camb_path = ic.camb_path
    if z == 99:
        snapshot = ic.initial_conditions
    elif z == 0:
        snapshot = ic.final_snapshot
    else:
        snapshot = None
        NameError("Select a valid redshift")

    if cosmology == "WMAP7":
        powerspec = pynbody.analysis.hmf.PowerSpectrumCAMB(snapshot)
        powerspec.set_sigma8(0.81)

    elif cosmology == "WMAP5":
        try:
            powerspec = pynbody.analysis.hmf.PowerSpectrumCAMB(snapshot,
                                                               filename=camb_path + "luisa/camb_Pk_WMAP5")
            print("WARNING: Used CAMB saved power spectrum in" + str(camb_path) + "luisa/camb_Pk_WMAP5")
        except IOError:
            print("WARNING: Save power spectrum not found - Computing and saving a new power spectrum in "
                  + str(camb_path) + "luisa/")
            power_spectrum_from_CAMB_WMAP5(ic.initial_conditions, save=True)
            powerspec = pynbody.analysis.hmf.PowerSpectrumCAMB(snapshot,
                                                               filename=camb_path + "luisa/camb_Pk_WMAP5")
        powerspec.set_sigma8(0.817)
        print("sigma8 is " + str(powerspec.get_sigma8()))

    elif cosmology == "PLANCK":
        try:
            powerspec = pynbody.analysis.hmf.PowerSpectrumCAMB(snapshot,
                                                               filename=camb_path + "luisa/camb_Pk_PLANCK2015")
            print("WARNING: Used CAMB saved power spectrum in" + str(camb_path) + "luisa/camb_Pk_PLANCK2015")
        except IOError:
            print("WARNING: Save power spectrum not found - Computing and saving a new power spectrum in "
                  + str(camb_path) + "luisa/")
            power_spectrum_from_CAMB_Planck2015(ic.initial_conditions, save=True)
            powerspec = pynbody.analysis.hmf.PowerSpectrumCAMB(snapshot,
                                                               filename=camb_path + "luisa/camb_Pk_PLANCK2015")
        powerspec.set_sigma8(0.831)
        print("sigma8 is " + str(powerspec.get_sigma8()))
    else:
        NameError("Other cosmologies not yet implemented")

    return powerspec


def k_Pk_simulation(snapshot, k_bins=None, shape=256):
    rho = snapshot.dm["rho"]
    den_mean = pynbody.analysis.cosmology.rho_M(snapshot, unit=rho.units)
    delta = ((rho - den_mean)/den_mean).reshape(shape, shape, shape)

    deltak = np.fft.fftn(delta)
    delta2 = (np.real(deltak * np.conj(deltak))).reshape(shape ** 3, )

    a = np.load("/Users/lls/Documents/mlhalos_files/Fourier_transform_matrix.npy")
    k = 2 * np.pi * a / 50
    kr = k.reshape(256**3,)
    if k_bins is None:
        log_k = np.linspace(np.log10(kr[kr>0].min()*2), np.log10(kr.max()), 15)
        k_bins = 10**log_k
    # k is in units h/Mpc

    # top_hat = (3. * (np.sin(k * radius) - ((k * radius) * np.cos(k * radius)))) / ((k * radius) ** 3)

    pk = []
    for i in range(len(k_bins) - 1):
        ind = (kr >= k_bins[i]) & (kr < k_bins[i+1])
        pk.append(np.mean(delta2[ind]))

    midk = (k_bins[1:] + k_bins[:-1])/2

    Pk = np.array(pk) / (50**3) #Pk is in units h^-3 Mpc^3
    return np.column_stack((midk, Pk))



