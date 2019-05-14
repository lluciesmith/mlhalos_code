"""
:mod:`parameters`

Sets the initial conditions parameters needed to run round trip algorithm,
starting from simulation snapshots and optional definition of in and out particles.

# /Users/lls/Documents/CODE
# /home/lls/stored_files
"""

import pynbody
import numpy as np
import scipy


class InitialConditionsParameters(object):
    """
    Sets initial parameters for the classification problem of particles living 'in' or 'out'
    of a chosen range of halos at a final snapshot, given their initial conditions state
    """
    def __init__(self, initial_snapshot=None, final_snapshot=None, snapshot=None, load_final=True,
                 min_halo_number=0, max_halo_number=400, min_mass_scale=3e10, max_mass_scale=1e15, ids_type='all',
                 num_particles=None, n_particles_per_cat=None, path="/home/lls/stored_files", sigma8=0.817):
        """
        Instantiates :class:`InitialConditionsParameters`.

        Args:
            initial_snapshot (str): Initial conditions snapshot. Default is snapshot at z=99.
            final_snapshot (str): Final snapshot with particles in halos. Default is snapshot at z=0.
            snapshot (str, None): Optional choice of initial conditions snapshot.
            min_halo_number (int): Lowest halo-id of range halos containing 'in' particles.
                Default is 0. Note that smaller halo-id corresponds to larger halo.
            max_halo_number (int): Highest halo-id of range halos containing 'in' particles.
                Default is 400.
            min_mass_scale (float): Minimum mass scale for computing trajectories.
            max_mass_scale (float): Maximum mass scale for computing trajectories.
            ids_type (str, int): ['all', 'random', 'innermost'].
                'all' takes all in particles of halos min_halo_number - max_halo_number and all out
                particles outside the given range and in no halo.
                'random' takes a num_particles random subset of 'all' particles.
                'innermost' takes the twenty innermost particles of halos (in and out).
            num_particles (int, None): Number of random particles if ids_type='random' to draw from
                all particles in the simulation.
            n_particles_per_cat (int, None): Optional, equal number of in, out-in-some-halo,
                out-in-no-halo particles to draw id ids_type=='random'.
            path (str): Snapshots' path - could be the default or "/Users/lls/Documents/CODE/"

        Returns:
            initial_conditions (SimSnap): Loaded initial conditions snapshot with attributes in physical_units.
            final_snapshot(SimSnap): Loaded final snapshot with attributes in physical_units.
            M_min (SimArray): Mass of smallest halo in range halos containing 'in' particles.
            M_max (SimArray): Mass of largest halo in range halos containing 'in' particles.
            num_particles (str): Number of random particles instantiated, if any.
            boxsize_no_units (float): Size of simulation box with no units attached.
            mean_density (SimArray): Mean matter density of the Universe in the initial conditions.
            ids_IN: Particle-ids living in halos of given range.
            ids_OUT: Particle-ids in no halo or living in halos outside given "in" range.

        """

        # Initial conditions

        if initial_snapshot is None:
            initial_snapshot = str(path) + "/Nina-Simulations/double/ICs_z99_256_L50_gadget3.dat"

        self.initial_conditions = pynbody.load(initial_snapshot)
        self.initial_conditions.physical_units()

        if sigma8 is not None:
            self.initial_conditions.properties['sigma8'] = sigma8
            print("WARNING: Setting sigma8 at z=0 to " + str(sigma8))

        self.min_halo_number = min_halo_number
        self.max_halo_number = max_halo_number

        self.M_min = pynbody.array.SimArray(min_mass_scale)
        self.M_max = pynbody.array.SimArray(max_mass_scale)
        self.M_min.units = "Msol"
        self.M_max.units = "Msol"

        self.num_particles = num_particles
        self.n_particles_per_cat = n_particles_per_cat

        self.shape = int(scipy.special.cbrt(len(self.initial_conditions)))

        # Evaluate mean density and boxsize at snapshot `self.snapshot`.
        # if snapshot is not None:
        #     self.snapshot = pynbody.load(snapshot)
        # elif snapshot is None:
        #     self.snapshot = None
        # self.boxsize_no_units = self.get_boxsize_with_no_units_at_snapshot_in_units(snapshot=self.snapshot)
        # self.boxsize_comoving = self.get_boxsize_in_comoving_units(snapshot=self.snapshot)
        # self.mean_density = self.get_mean_matter_density_in_the_box(snapshot=self.snapshot)

        # Evaluate mean density and boxsize in the initial conditions.
        self.boxsize_no_units = self.get_boxsize_with_no_units_at_snapshot_in_units(snapshot=self.initial_conditions)
        self.boxsize_comoving = self.get_boxsize_in_comoving_units(snapshot=self.initial_conditions)
        self.mean_density = self.get_mean_matter_density_in_the_box(snapshot=self.initial_conditions)

        if path == "/home/lls/stored_files":
            self.camb_path = "/home/lls/stored_files/camb/"
        else:
            self.camb_path = "/Users/lls/Software/CAMB-Jan2017/"

        self.path = path

        # Final snapshot

        if load_final is True:
            if final_snapshot is None:
                final_snapshot = str(path) + "/Nina-Simulations/double/snapshot_104"

            self.final_snapshot = pynbody.load(final_snapshot)
            self.final_snapshot.physical_units()
            self.halo = self.final_snapshot.halos(make_grp=True)

            if sigma8 is not None:
                self.final_snapshot.properties['sigma8'] = sigma8
                print("WARNING: Setting sigma8 at z=0 to " + str(sigma8))

            # IDS IN AND OUT

            if ids_type == 'all':
                self.generate_all_ids()

            elif ids_type == 'random':
                if num_particles is not None:
                    self.generate_random_ids()

                if n_particles_per_cat is not None:
                    self.generate_all_ids()
                    self.ids_IN = np.random.choice(self.ids_IN, self.n_particles_per_cat, replace=False)
                    self.ids_OUT = np.random.choice(self.ids_OUT, self.n_particles_per_cat, replace=False)

            elif ids_type == 'innermost':
                self.generate_innermost_ids_of_halos()

            elif isinstance(ids_type, (int, float, list, np.ndarray)):
                ids_type = np.array(ids_type)
                self.ids_IN = ids_type[np.in1d(ids_type, self.get_all_particles_in_range_halos())]
                self.ids_OUT = ids_type[np.in1d(ids_type, self.get_all_particles_out_range_halos())]
            else:
                raise ValueError("Unknown ids_type")

    def generate_all_ids(self):
        self.ids_IN = self.get_all_particles_in_range_halos()
        self.ids_OUT = self.get_all_particles_out_range_halos()

    def generate_random_ids(self):
        random_ids = self.get_random_subset_particles()
        self.ids_IN = random_ids[0]
        self.ids_OUT = random_ids[1]

    def generate_innermost_ids_of_halos(self):
        ids_OUT = []
        ids_IN = []

        for i in range(self.max_halo_number + 1, len(self.halo)):
            ids_out = self.halo[i]['iord'][0:20]
            ids_OUT.append(ids_out)

        self.ids_OUT = np.array(ids_OUT).ravel()

        for i in range(self.min_halo_number, self.max_halo_number + 1):
            ids_in = self.halo[i]['iord'][0:20]
            ids_IN.append(ids_in)

        self.ids_IN = np.array(ids_IN).ravel()

    def get_boxsize_with_no_units_at_snapshot_in_units(self, snapshot=None, units="Mpc"):
        if snapshot is None:
            snapshot = self.initial_conditions

        snapshot.physical_units()
        boxsize = snapshot.properties['boxsize'].in_units(units)
        return boxsize

    def get_boxsize_in_comoving_units(self, snapshot=None):
        if snapshot is None:
            snapshot = self.initial_conditions

        h = snapshot.properties['h']
        a = snapshot.properties['a']
        boxsize = snapshot.properties['boxsize'].in_units("Mpc") * h/ a
        return boxsize

    def get_mean_matter_density_in_the_box(self, snapshot=None, units="Mpc"):
        """
        Mean matter density of the universe at snapshot.
        Snapshot is by default the initial conditions snapshot.
        """
        if snapshot is None:
            snapshot = self.initial_conditions

        boxsize = self.get_boxsize_with_no_units_at_snapshot_in_units(snapshot=snapshot, units=units)
        volume = boxsize ** 3

        if snapshot["mass"].units != "Msol":
            m = snapshot["mass"].in_units("Msol")
        else:
            m = snapshot["mass"]

        rho_bar = pynbody.array.SimArray(len(snapshot) * m[0] / volume)
        rho_bar.units = 'Msol ' + units + '**-3'

        return rho_bar

    ##########################################################################################
    # FUNCTIONS GENERATING PARTICLE ID LISTS
    ##########################################################################################

    def get_2D_array_ids_and_its_halo(self):
        """ Get 2d array of form [particle_id, halo_of_particle]. """

        particle_id_list = np.array(self.final_snapshot['iord'])
        halos_corresponding_particle_id_list = self.final_snapshot['grp']

        list_id_and_corresponding_halo = np.column_stack((particle_id_list, halos_corresponding_particle_id_list))
        return list_id_and_corresponding_halo

    @staticmethod
    def select_only_particles_belonging_to_some_halo(list_id_and_corresponding_halos):
        """
        Pick out all particles that belong to some halo from list_id_and_correspondding_halos.
        Particles that are in no halo are labeled -1.
        """
        ids_in_some_halo_and_correspoding_halo = list_id_and_corresponding_halos[
            np.where(list_id_and_corresponding_halos[:, 1] != -1)]

        return ids_in_some_halo_and_correspoding_halo

    def particle_in_out_halo(self, particle_id_list):
        """label in particles +1 and out particles -1."""

        id_to_h = self.final_snapshot['grp'][particle_id_list]

        output = np.ones(len(id_to_h)) * -1
        output[np.where((id_to_h >= self.min_halo_number) & (id_to_h <= self.max_halo_number))] = 1
        output = output.astype("int")
        return output

    def get_particle_ids_with_label_in_or_out_range_halos(self):
        """
        Get a 2D array of the form [particle_ID_number, label], such that in particles have label +1
        and out particles have label -1.
        """
        list_id_and_corresponding_halo = self.get_2D_array_ids_and_its_halo()
        ids_in_some_halo_and_corresponding_halo = self.select_only_particles_belonging_to_some_halo(
            list_id_and_corresponding_halo)

        label_in_or_out = self.particle_in_out_halo(ids_in_some_halo_and_corresponding_halo[:, 0])
        list_id_and_corresponding_label = np.column_stack(
            (ids_in_some_halo_and_corresponding_halo[:, 0], label_in_or_out))

        return list_id_and_corresponding_label

    # ALL PARTICLES

    def get_all_particles_in_range_halos(self):
        """ Get list all particle-ids that live in chosen range halos """
        list_id_and_corresponding_label = self.get_particle_ids_with_label_in_or_out_range_halos()

        pick_ids_and_label_in_range_halos = list_id_and_corresponding_label[
            np.where(list_id_and_corresponding_label[:, 1] == 1)]
        ids_in_range_halo = pick_ids_and_label_in_range_halos[:, 0]

        return ids_in_range_halo

    def get_all_particles_out_range_halos(self):
        """ Concatenate out-in-some-halo and out-in-no-halo particle IDs"""
        out_other_halo = self.get_all_particles_out_range_in_other_halo()
        out_no_halo = self.get_all_particles_out_any_halo()

        all_out = np.concatenate((out_other_halo, out_no_halo))
        return np.sort(all_out)

    def get_all_particles_out_range_in_other_halo(self):
        """ Get list all particle-ids that live outside chosen range halos """
        list_id_and_corresponding_label = self.get_particle_ids_with_label_in_or_out_range_halos()
        list_ids_and_label_out_range_halos = list_id_and_corresponding_label[
            np.where(list_id_and_corresponding_label[:, 1] == -1)]

        ids_out_range_halo = list_ids_and_label_out_range_halos[:, 0]
        return ids_out_range_halo

    def get_all_particles_out_any_halo(self):
        """ Get list all particle-ids that live in no halo """
        list_id_and_corresponding_halo = self.get_2D_array_ids_and_its_halo()

        list_ids_and_label_out_halos = list_id_and_corresponding_halo[
            np.where(list_id_and_corresponding_halo[:, 1] == -1)]
        ids_in_no_halo = list_ids_and_label_out_halos[:, 0]

        return ids_in_no_halo

    # RANDOM PARTICLES

    @staticmethod
    def concatenate_all_in_out_particles(ids_all_in, ids_all_out):
        return np.concatenate((ids_all_in, ids_all_out))

    @staticmethod
    def get_random_particles_in_range_halos(random_subset_ids, ids_all_in):
        """ Select num_particles random subset of all in particles """
        ids_in = random_subset_ids[np.in1d(random_subset_ids, ids_all_in)]
        return ids_in

    @staticmethod
    def get_random_particles_out_range_in_other_halo(random_subset_ids, ids_all_out):
        """ Select num_particles random subset of all out particles """
        ids_out = random_subset_ids[np.in1d(random_subset_ids, ids_all_out)]
        return ids_out

    def get_random_subset_particles(self):
        ids_all_in = self.get_all_particles_in_range_halos()
        ids_all_out = self.get_all_particles_out_range_halos()

        all_in_out = self.concatenate_all_in_out_particles(ids_all_in, ids_all_out)
        random_subset_ids = np.random.choice(all_in_out, self.num_particles, replace=False)

        ids_in = self.get_random_particles_in_range_halos(random_subset_ids, ids_all_in)
        ids_out = self.get_random_particles_out_range_in_other_halo(random_subset_ids, ids_all_out)

        return ids_in, ids_out


class InitialConditionsParametersLgadget(object):
    """
    Sets initial parameters for the classification problem of particles living 'in' or 'out'
    of a chosen range of halos at a final snapshot, given their initial conditions state
    """
    def __init__(self, loaded_sim, min_mass_scale=3e10, max_mass_scale=1e15, path=None, sigma8=0.82, is_sim_final=True):
        """
        Instantiates :class:`InitialConditionsParameters`.

        Args:
            initial_snapshot (str): Loaded initial conditions --
            path (str): Snapshots' path - could be the default or "/Users/lls/Documents/CODE/"

        Returns:
            initial_conditions (SimSnap): Loaded initial conditions snapshot with attributes
            in physical_units.
            final_snapshot(SimSnap): Loaded final snapshot with attributes in physical_units.
            M_min (SimArray): Mass of smallest halo in range halos containing 'in' particles.
            M_max (SimArray): Mass of largest halo in range halos containing 'in' particles.
            num_particles (str): Number of random particles instantiated, if any.
            boxsize_no_units (float): Size of simulation box with no units attached.
            mean_density (SimArray): Mean matter density of the Universe in the initial conditions.
            ids_IN: Particle-ids living in halos of given range.
            ids_OUT: Particle-ids in no halo or living in halos outside given "in" range.

        """

        # Initial conditions

        self.initial_conditions = loaded_sim

        if sigma8 is not None:
            self.initial_conditions.properties['sigma8'] = sigma8
            print("WARNING: Setting sigma8 at z=99 to " + str(sigma8))

        self.M_min = pynbody.array.SimArray(min_mass_scale)
        self.M_max = pynbody.array.SimArray(max_mass_scale)
        self.M_min.units = "Msol"
        self.M_max.units = "Msol"

        self.shape = int(scipy.special.cbrt(len(self.initial_conditions)))
        self.path = path

        # Evaluate mean density and boxsize in the initial conditions.
        self.boxsize_no_units = self.get_boxsize_with_no_units_at_snapshot_in_units(snapshot=self.initial_conditions)
        self.boxsize_comoving = self.get_boxsize_in_comoving_units(snapshot=self.initial_conditions)
        self.mean_density = self.get_mean_matter_density_in_the_box(snapshot=self.initial_conditions)


    def get_boxsize_with_no_units_at_snapshot_in_units(self, snapshot=None, units="Mpc"):
        if snapshot is None:
            snapshot = self.initial_conditions

        # snapshot.physical_units()
        boxsize = snapshot.properties['boxsize'].in_units(units)
        return boxsize

    def get_boxsize_in_comoving_units(self, snapshot=None):
        if snapshot is None:
            snapshot = self.initial_conditions

        h = snapshot.properties['h']
        a = snapshot.properties['a']
        boxsize = snapshot.properties['boxsize'].in_units("Mpc") * h / a
        return boxsize

    def get_mean_matter_density_in_the_box(self, snapshot=None):
        """
        Mean matter density of the universe at snapshot.
        Snapshot is by default the initial conditions snapshot.
        """
        if snapshot is None:
            snapshot = self.initial_conditions

        boxsize = self.get_boxsize_with_no_units_at_snapshot_in_units(snapshot=snapshot)
        volume = boxsize ** 3

        if snapshot["mass"].units != "Msol":
            m = snapshot["mass"].in_units("Msol")
        else:
            m = snapshot["mass"]

        rho_bar = pynbody.array.SimArray(len(snapshot) * m[0] / volume)
        rho_bar.units = 'Msol Mpc**-3'

        return rho_bar
