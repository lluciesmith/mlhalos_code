# WAY 2: Use ckdtree from scipy

def query_ckdtree(self, position_peaks, snapshot=None):
    # takes too long!
    if snapshot is None:
        snapshot = self.initial_parameters.initial_conditions

    t = cKDTree(position_peaks)
    pos_ids = snapshot["pos"]
    queries = t.query(pos_ids, k=1)
    return queries


def build_kdtree(self, snapshot=None):
    if snapshot is None:
        snapshot = self.initial_parameters.initial_conditions

    pos_ids = snapshot["pos"]
    t = KDTree(pos_ids)
    return t


def get_nearest_neighbours_using_tree(self, peak_positions, radius=20, snapshot=None):
    tree = self.build_kdtree(snapshot=snapshot)
    nn = tree.query_ball_point(peak_positions, radius)
    return nn


def assign_peak_id(self, nn_peaks, particle_ids=None):
    if particle_ids is None:
        particle_ids = np.arange(256 ** 3)

    peak_ids = np.zeros((256 ** 3)) - 1
    for i in range(len(nn_peaks)):
        ids_in_peak_i = np.array(nn_peaks[i])[np.in1d(nn_peaks[i], particle_ids)]
        peak_ids[ids_in_peak_i] = i

    return peak_ids


def assign_peak_label_to_ids_tree(self, peak_positions, radius, snapshot=None):
    if snapshot is None:
        snapshot = self.initial_parameters.initial_conditions

    nn_each_peak = self.get_nearest_neighbours_using_tree(peak_positions, radius=radius, snapshot=snapshot)
    peak_labels = self.assign_peak_id(nn_each_peak, particle_ids=None)
    return peak_labels


# WAY 3: Take particles ids within spheres around peaks with some partial centering of box on sphere

def get_ids_in_sphere(self, snapshot, centre, radius, mode="hyb", wrap=True):
    sph = snapshot[pynbody.filt.Sphere(radius, centre)]
    t = pynbody.analysis.halo.center(sph, mode=mode, vel=False, move_all=False, wrap=wrap)
    nn = sph["iord"]
    t.revert()
    return nn


def get_ids_around_peaks(self, peak_positions, radius, snapshot=None):
    if snapshot is None:
        snapshot = self.initial_parameters.initial_conditions

    result = [self.get_ids_in_sphere(snapshot, x, radius) for x in peak_positions]
    return result


def assign_peak_label_to_ids_sphere(self, peak_ids, peak_positions, radius, snapshot=None):
    if snapshot is None:
        snapshot = self.initial_parameters.initial_conditions

    peak_labels = np.zeros((256 ** 3)) - 1
    for i in range(len(peak_positions)):
        assert snapshot[peak_ids[i]]["pos"] == peak_positions[i]
        ids = self.get_ids_in_sphere(snapshot, peak_positions[i], radius)
        peak_labels[ids] = i
    return peak_labels


# WAY 1

def get_ids_local_maxima_density_contrast(self, delta):
    ids_maxima = self.delta_class.get_particle_ids_local_maxima(delta)
    return ids_maxima


def get_distance_wrt_single_peak(self, peak_id, snapshot=None):
    if snapshot is None:
        snapshot = self.initial_parameters.initial_conditions

    pynbody.analysis.halo.center(snapshot[peak_id], vel=False)
    snapshot.wrap()
    position_particles = snapshot["pos"]
    return position_particles


def get_distance_wrt_multiple_peaks(self, peaks_ids, snapshot=None):
    position_from_peaks = np.zeros((len(peaks_ids), int(self.initial_parameters.shape ** 3), 3))

    for i in range(len(peaks_ids)):
        print("Done peak " + str(i))
        position_from_peaks[i] = self.get_distance_wrt_single_peak(peaks_ids[i], snapshot=snapshot)

    return position_from_peaks


def position_from_peaks_delta(self, initial_parameter, delta, snapshot=None):
    peak_ids = self.get_ids_local_maxima_density_contrast(delta)
    xyz_from_peaks = self.get_distance_wrt_multiple_peaks(peak_ids, snapshot=snapshot)
    return xyz_from_peaks
