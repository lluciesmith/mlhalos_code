"""
These weird halos have two weird properties:
1- The `mostboundID` from .properties is NOT in the halo but in a different one (usually the next halo ID).
Furthermore, the `mostboundID` is not the same as the first id in the list of halo['iord'].
2- They have `rcrit_200` from .properties to be 0!

The first part of the script tests if centering around the `mostboundID` or the first ID in `iord` list and then
wrapping, will give same radius of a particle. i.e. if it will make a difference or not.

The second part calculates the virial radius of these weird halos using the pynbody function
`pynbody.analysis.halo.virial_radius` to see if they get a sensible answer. One can check that the virial radius of
halos of similar mass should also be similar.

"""

import numpy as np
import pynbody
import pynbody.filt as filt
import pynbody.transformation as transformation
import pynbody.analysis.cosmology as cosmology
import pynbody.util as util
import logging
logger = logging.getLogger('pynbody.analysis.halo')


# weird_halos = np.load('weird_halos.npy')
#
# final_snapshot="/Users/lls/Documents/CODE/Nina-Simulations/double/snapshot_104"
# f = pynbody.load(final_snapshot)
# h = f.halos(make_grp=True)
#
# test=[]
# for halo_ID in weird_halos:
#
#     # METHOD 1
#
#     mb_id = h[halo_ID].properties['mostboundID']
#     pynbody.analysis.halo.center(f[halo_ID], vel=False)
#     f.wrap()
#     pynbody.analysis.halo.center(h[1962], vel=False)
#
#     particle = h[halo_ID]['iord'][8]
#     r_mb_id = f[particle]['pos']
#
#
#     # METHOD 2
#
#     first_id = h[halo_ID]['iord'][0]
#     pynbody.analysis.halo.center(f[first_id], vel=False)
#     f.wrap()
#     pynbody.analysis.halo.center(h[halo_ID], vel=False)
#
#     r_first_id = f[particle]['pos']
#
#     # check if they are the same
#     check = np.allclose(r_mb_id, r_first_id)
#
#     test.append(check)
#
# test = np.array(test)


###################### CALCULATE VIRIAL RADIUS OF EACH "WEIRD" HALO USING PYNBODY FUNCTION ######################

# def virial_radius_modified(sim, cen=None, overden=178, r_max=None):
#     """Calculate the virial radius of the halo centered on the given
#     coordinates.
#
#     This is here defined by the sphere centered on cen which contains a
#     mean density of overden * rho_M_0 * (1+z)^3.
#
#     """
#
#     if r_max is None:
#         r_max = (sim["x"].max() - sim["x"].min())
#     else:
#         if cen is not None:
#             sim = sim[filt.Sphere(r_max, cen)]
#         else:
#             sim = sim[filt.Sphere(r_max)]
#
#     r_min = 0.0
#
#     if cen is not None:
#         tx = transformation.inverse_translate(sim, cen)
#     else:
#         tx = transformation.null(sim)
#
#     target_rho = overden * \
#         sim.properties[
#             "omegaM0"] * cosmology.rho_crit(sim, z=0) * (1.0 + sim.properties["z"]) ** 3
#     logger.info("target_rho=%s", target_rho)
#
#     with tx:
#         sim = sim[filt.Sphere(r_max)]
#         with sim.immediate_mode:
#             mass_ar = np.asarray(sim['mass'])
#             r_ar = np.asarray(sim['r'])
#
#         #pure numpy implementation
#         rho = lambda r: np.dot(
#             mass_ar, r_ar < r) / (4. * np.pi * (r ** 3) / 3)
#
#         #numexpr alternative - not much faster because sum is not threaded
#         #def rho(r) :
#         #    r_ar; mass_ar; # just to get these into the local namespace
#         #    return ne.evaluate("sum((r_ar<r)*mass_ar)")/(4.*math.pi*(r**3)/3)
#
#         #rho = lambda r: util.sum_if_lt(mass_ar,r_ar,r)/(4. * math.pi * (r ** 3) / 3)
#         result = util.bisect(r_min, r_max, lambda r: target_rho -
#                              rho(r), epsilon=0, eta=1.e-3 * target_rho, verbose=False)
#
#     return result

weird_halos = np.load('/Users/lls/Documents/CODE/stored_files/all_out/radii_files/weird_halos.npy')

final_snapshot="/Users/lls/Documents/CODE/Nina-Simulations/double/snapshot_104"
f = pynbody.load(final_snapshot)
h = f.halos(make_grp=True)

virial_radii = []

for halo_ID in weird_halos:
    try:
        pynbody.analysis.halo.center(f[h[halo_ID].properties['mostboundID']], vel=False)
        f.wrap()
        pynbody.analysis.halo.center(h[halo_ID], vel=False)
        f.wrap()

        vir_r = pynbody.analysis.halo.virial_radius(f)
        print("halo " + str(halo_ID) + " has virial radius " + str(vir_r) + " " +str(vir_r.units))
        virial_radii.append(vir_r)
    except:
        print("halo " + str(halo_ID) + " gave ValueError: Bisect algorithm does not converge")
        pass

virial_radii = np.array(virial_radii)
