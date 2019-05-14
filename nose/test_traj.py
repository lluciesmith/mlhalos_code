import numpy as np
from mlhalos import trajectories
from mlhalos import parameters


def test_trajectories():
    """
    Create a test for the trajectories.
    We take initial parameters, choose particle_ids = [12599035, 2642532] , run trajectory analysis
    and make sure the trajectories predicted are the same as those in files trajectory_particle_2642532_out.npy and
    trajectory_particle_12599035.npy.
    """
    trajectory_true_out = np.load("trajectory_particle_2642532_out.npy")
    trajectory_true_in = np.load("trajectory_particle_12599035.npy")

    def trajectories_test():
        p = parameters.InitialConditionsParameters(
                    initial_snapshot="/Users/lls/Documents/CODE/Nina-Simulations/double/ICs_z99_256_L50_gadget3.dat",
                    final_snapshot="/Users/lls/Documents/CODE/Nina-Simulations/double/snapshot_104",
                    min_halo_number=0, max_halo_number=400,
                    ids_type='all', num_particles=None)

        trajectory_test = trajectories.Trajectories(init_parameters=p, particles=[12599035, 2642532], num_of_filters=20,
                                                    num_particles=1)
        trajectory_in = trajectory_test.delta_in
        trajectory_out = trajectory_test.delta_out
        return trajectory_in, trajectory_out

    trajectory_test_in, trajectory_test_out = trajectories_test()

    assert np.allclose(trajectory_true_in, trajectory_test_in)
    assert np.allclose(trajectory_true_out, trajectory_test_out)
