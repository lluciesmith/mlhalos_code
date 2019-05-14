import numpy as np
from mlhalos import parameters
from mlhalos import inertia
import pynbody
from scipy.spatial import cKDTree


################### UNIT TEST 1 ############################

def test_radius_from_coordinates():
    """ This passed on 1st May 2018 """
    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)

    DP = inertia.DensityPeaks(initial_parameters=ic, num_filtering_scales=5)
    d = DP.density_contrasts

    In = inertia.Inertia(initial_parameters=ic, density_contrasts=d, num_filtering_scales=5)

    # Find peaks -- particles ids and positions
    pos_max = DP.get_positions_local_maxima_density_contrast(d[:,2])
    particle_ids_max = DP.get_particle_ids_local_maxima_density_contrast(d[:,2])

    # Embed box in a bigger box to account for periodicity
    # Find peaks -- particles ids and positions

    nearest_peaks = DP.get_nearest_peak(particle_ids_max, pos_max, output="id+position")

    # get dx,dy,dz from nearest peak
    position_particles = ic.initial_conditions["pos"]
    dx, dy, dz = In.get_difference_coordinates(position_particles, nearest_peaks[:, 1:])
    r = np.sqrt((dx**2) + (dy**2) + (dz**2))

    # check that resulting distance is the same as the one returned by scikit-learn
    extended_peaks = DP.extended_peaks_with_periodic_boundary_conditions(particle_ids_max, pos_max, ic.initial_conditions)
    dist, ids = DP.find_kNearestNeighbour_ml(position_particles, extended_peaks[:, 1:], return_distance=True)
    np.testing.assert_allclose(dist, r)


################### UNIT TEST 2 ############################

def test_nearest_peak_id_different_centering():
    """ This passed on 1st May 2018 """
    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)

    DP = inertia.DensityPeaks(initial_parameters=ic, num_filtering_scales=5)
    d = DP.density_contrasts

    particle_ids_max = DP.get_particle_ids_local_maxima_density_contrast(d[:,2])
    pos_max = DP.get_positions_local_maxima_density_contrast(d[:,2])
    nearest_peaks = DP.get_nearest_peak(particle_ids_max, pos_max, output="id+position")

    for i in range(3):
        p_centering = np.random.choice(np.arange(256**3), 1)
        print("Centering on particle ID " + str(p_centering))
        tr = pynbody.analysis.halo.center(ic.initial_conditions[p_centering[0]], vel=False, mode="hyb")

        ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)
        DP = inertia.DensityPeaks(initial_parameters=ic, num_filtering_scales=5)
        d = DP.density_contrasts

        pos_max_i = DP.get_positions_local_maxima_density_contrast(d[:,2])
        p_ids_i = DP.get_particle_ids_local_maxima_density_contrast(d[:,2])

        nearest_peaks_i = DP.get_nearest_peak(p_ids_i, pos_max_i, output="id+position")

        np.testing.assert_allclose(nearest_peaks_i[:, 0], nearest_peaks[:, 0])
        np.testing.assert_allclose(p_ids_i, particle_ids_max)
        tr.revert()

################### UNIT TEST 3 ############################


def test_knn_against_ckdtree():
    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)

    DP = inertia.DensityPeaks(initial_parameters=ic, num_filtering_scales=2)
    d = DP.density_contrasts

    position_peaks = DP.get_positions_local_maxima_density_contrast(d[:, 2])
    pos_ids = ic.initial_conditions["pos"]

    t = cKDTree(position_peaks)
    queries = t.query(pos_ids, k=1)
    nearest_peaks = DP.find_kNearestNeighbour_ml(pos_ids, position_peaks, return_distance=True)

    np.testing.assert_allclose(queries, nearest_peaks)
