import numpy as np
from mlhalos import parameters
import pynbody
import matplotlib.pyplot as plt
import numpy.ma as ma
import matplotlib

# inertia

path_inertia = "/Users/lls/Documents/mlhalos_files/regression/inertia/"
#
# inertia_log_true_mass = np.load(path_inertia + "true_halo_mass.npy")
# bins_plotting = np.linspace(inertia_log_true_mass.min(), inertia_log_true_mass.max(), 15, endpoint=True)
#ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)

# get predicted mass in 3D array

den_plus_inertia = np.load(path_inertia + "inertia_plus_den/predicted_halo_mass.npy")
test_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")
predicted_all = np.zeros(256**3)
predicted_all[test_ids] = den_plus_inertia
predicted_all = predicted_all.reshape(256, 256, 256)

# mask training ids and particles in no halo

ids_training = np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
h_m = np.load("/Users/lls/Documents/CODE/stored_files/halo_mass_particles.npy").reshape(256, 256, 256)
all_ids = np.arange(256**3)
mask_tr = np.in1d(all_ids, ids_training).reshape(256, 256, 256)
mask = (mask_tr) ^ (h_m.reshape(256, 256, 256) == 0)


def plot_z_slice(z_ind, predicted, true, mask):
    true_masked = ma.masked_array(true, mask=mask)
    predicted_masked= ma.masked_array(predicted, mask=mask)
    norm = matplotlib.colors.LogNorm(vmin=true_masked[:, :, z_ind].min(), vmax=true_masked[:, :, z_ind].max())

    fig, axs = plt.subplots(1, 2, figsize=(13,7))
    a0 = axs[0].imshow(true_masked[:, :, z_ind], norm=norm)
    #fig.colorbar(a0, ax=axs[0])
    axs[0].set_title('True mass')
    a1 = axs[1].imshow(predicted_masked[:, :, z_ind], norm=norm)
    fig.colorbar(a1, ax=axs[1])
    axs[1].set_title('Predicted mass')

    fig.tight_layout()
    axs[0].set_xlim([0, 256])
    axs[0].set_ylim([0, 256])
    axs[1].set_xlim([0, 256])
    axs[1].set_ylim([0, 256])

    return fig


######### scatter plots with colors being predicted/true predictions ######

ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)

den_plus_inertia = np.load(path_inertia + "inertia_plus_den/predicted_halo_mass.npy")
test_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")
predicted_all = np.zeros(256**3)
predicted_all[test_ids] = den_plus_inertia

h_m = np.load("/Users/lls/Documents/CODE/stored_files/halo_mass_particles.npy")

all_ids = np.arange(256**3)
ids_training = np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
ids_not_used = all_ids[(np.in1d(all_ids, ids_training)) | (h_m == 0) | (h_m<10**11)]


def plot_scatter_z_slice(predicted, true, masked_ids, z_ind, initial_parameters):
    all_ids = np.arange(256 ** 3).reshape(256, 256, 256)
    ids_slice = all_ids[:, :, z_ind].flatten()
    ids_testing_slice = ids_slice[~np.in1d(ids_slice, masked_ids)]

    fig, axs = plt.subplots(1, 2, figsize=(13, 7))

    maxi = np.log10(h_m.max())
    mini = np.log10(h_m[h_m > 0].min())
    norm = plt.Normalize(mini, maxi)

    boxsize = initial_parameters.final_snapshot.properties["boxsize"]
    x = (initial_parameters.final_snapshot[ids_testing_slice]["x"]/boxsize).in_units("1")
    y = (initial_parameters.final_snapshot[ids_testing_slice]["y"] / boxsize).in_units("1")

    a0 = axs[0].scatter(x, y, c=np.log10(predicted[ids_testing_slice]), norm=norm, s=0.001)
    axs[0].set_title('Predicted mass')
    a1 = axs[1].scatter(x, y, c=np.log10(true[ids_testing_slice]), norm=norm, s=0.001)
    axs[1].set_title('True mass')
    fig.colorbar(a1, ax=axs[1])
    fig.tight_layout()

    fig.subplots_adjust(wspace=0.05)
    axs[0].set_xlabel(r"$x/\mathrm{L_{box}}$")
    axs[1].set_xlabel(r"$x/\mathrm{L_{box}}$")
    axs[0].set_ylabel(r"$y/\mathrm{L_{box}}$")
    axs[1].set_yticklabels([])
    axs[0].set_xlim([-0.02, 1.02])
    axs[0].set_ylim([-0.02, 1.02])
    axs[1].set_xlim([-0.02, 1.02])
    axs[1].set_ylim([-0.02, 1.02])
    print(np.log10(predicted[ids_testing_slice]).min())
    print(np.log10(true[ids_testing_slice]).min())

    return fig

def plot_scatter_halo_slice(predicted, true, masked_ids, halo_num, initial_parameters, alpha=1):
    ids_slice = np.sort(initial_parameters.halo[halo_num]['iord'])
    ids_testing_slice = ids_slice[~np.in1d(ids_slice, masked_ids)]

    fig, axs = plt.subplots(1, 2, figsize=(13, 7))
    maxi = np.log10(h_m.max())
    mini = np.log10(h_m[h_m>0].min())
    norm = plt.Normalize(mini, maxi)
    boxsize = initial_parameters.final_snapshot.properties["boxsize"]
    x = (initial_parameters.final_snapshot[ids_testing_slice]["x"]/boxsize).in_units("1")
    y = (initial_parameters.final_snapshot[ids_testing_slice]["y"] / boxsize).in_units("1")
    a0 = axs[0].scatter(x, y, c=np.log10(predicted[ids_testing_slice]), norm=norm, alpha=alpha, s=0.001)
    axs[0].set_title('Predicted mass')
    a1 = axs[1].scatter(x, y, c=np.log10(true[ids_testing_slice]), norm=norm, alpha=alpha, s=0.001)
    axs[1].set_title('True mass')
    fig.colorbar(a1, ax=axs[1])
    fig.colorbar(a0, ax=axs[0])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    axs[0].set_xlabel(r"$x/\mathrm{L_{box}}$")
    axs[1].set_xlabel(r"$x/\mathrm{L_{box}}$")
    axs[0].set_ylabel(r"$y/\mathrm{L_{box}}$")
    axs[1].set_yticklabels([])
    # axs[0].set_xlim([-0.02, 1.02])
    # axs[0].set_ylim([-0.02, 1.02])
    # axs[1].set_xlim([-0.02, 1.02])
    # axs[1].set_ylim([-0.02, 1.02])

    return fig

def plot_scatter_ratio_slice(predicted, true, masked_ids, initial_parameters, subsnap=True, halo_num=10,
                             z_ind=40):
    if subsnap is True:
        all_ids = np.arange(256 ** 3).reshape(256, 256, 256)
        ids_slice = all_ids[:, :, z_ind].flatten()
        ids_testing_slice = ids_slice[~np.in1d(ids_slice, masked_ids)]
    else:

        ids_slice = np.sort(initial_parameters.halo[halo_num]['iord'])
        ids_testing_slice = ids_slice[~np.in1d(ids_slice, masked_ids)]

    fig = plt.figure()
    # maxi = np.log10(h_m.max())
    # mini = np.log10(h_m[h_m>0].min())
    # norm = plt.Normalize(mini, maxi)
    boxsize = initial_parameters.final_snapshot.properties["boxsize"]
    x = (initial_parameters.final_snapshot[ids_testing_slice]["x"]/boxsize).in_units("1")
    y = (initial_parameters.final_snapshot[ids_testing_slice]["y"] / boxsize).in_units("1")
    col = (np.log10(predicted[ids_testing_slice]) - np.log10(true[ids_testing_slice]))/ np.log10(true[ids_testing_slice])
    im = plt.scatter(x, y, c=col)
    plt.colorbar(im)
    plt.xlabel(r"$x/\mathrm{L_{box}}$")
    plt.ylabel(r"$y/\mathrm{L_{box}}$")
    # axs[0].set_xlim([-0.02, 1.02])
    # axs[0].set_ylim([-0.02, 1.02])
    # axs[1].set_xlim([-0.02, 1.02])
    # axs[1].set_ylim([-0.02, 1.02])

    return fig

plot_scatter_z_slice(predicted_all, h_m, ids_not_used, 20, ic)
plot_scatter_halo_slice(predicted_all, h_m, ids_not_used, 50, ic)

