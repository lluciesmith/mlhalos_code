import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code/")
import numpy as np
from regression.plots import plotting_functions as pf


def get_ids_from_falsein1_to_truein2(predicted2, predicted1, true, ids_tested):

    label2 = np.zeros((len(predicted2)))
    label1 = np.zeros((len(predicted1)))
    bins_plotting = np.linspace(true.min(), true.max(), 30, endpoint=True)
    for i in range(len(bins_plotting) - 1):
        index2 = (true >= bins_plotting[i]) & (true < bins_plotting[i+1]) \
              & (predicted2 >= bins_plotting[i]) & (predicted2 < bins_plotting[i+1])
        index1 = (true >= bins_plotting[i]) & (true < bins_plotting[i+1]) \
              & (predicted1 >= bins_plotting[i]) & (predicted1 < bins_plotting[i+1])

        if not label2[index2].size == 0:
            assert label2[index2].all() == 0
            label2[index2] = 1

        if not label1[index1].size == 0:
            assert label1[index1].all() == 0
            label1[index1] = 1

    ids_correct_in2 = ids_tested[(label1 == 0) & (label2 == 1)]
    return ids_correct_in2


def get_ids_from_truein1_to_falsein2(predicted2, predicted1, true, ids_tested):

    label2 = np.zeros((len(predicted2)))
    label1 = np.zeros((len(predicted1)))
    bins_plotting = np.linspace(true.min(), true.max(), 30, endpoint=True)
    for i in range(len(bins_plotting) - 1):
        index2 = (true >= bins_plotting[i]) & (true < bins_plotting[i+1]) \
              & (predicted2 >= bins_plotting[i]) & (predicted2 < bins_plotting[i+1])
        index1 = (true >= bins_plotting[i]) & (true < bins_plotting[i+1]) \
              & (predicted1 >= bins_plotting[i]) & (predicted1 < bins_plotting[i+1])

        if not label2[index2].size == 0:
            assert label2[index2].all() == 0
            label2[index2] = 1

        if not label1[index1].size == 0:
            assert label1[index1].all() == 0
            label1[index1] = 1

    ids_correct_in2 = ids_tested[(label1 == 1) & (label2 == 0)]
    return ids_correct_in2


# data

test_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/even_radii_and_random/"
                   "testing_ids.npy")


path_inertia = "/Users/lls/Documents/mlhalos_files/regression/inertia/"
x = np.load(path_inertia + "true_halo_mass.npy")
y_inden = np.load(path_inertia + "inertia_plus_den/predicted_halo_mass.npy")
y_inden = np.log10(y_inden)

path_density = "/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/even_radii_and_random/"
y_den = np.load(path_density + "predicted_halo_mass.npy")
y_den = np.log10(y_den)

path_shear = "/Users/lls/Documents/mlhalos_files/regression/shear/"
shear_predicted_mass = np.load(path_shear + "predicted_halo_mass.npy")
all_log_predicted_mass_shear = np.log10(shear_predicted_mass)

# radius data

radii_properties_in = np.load("/Users/lls/Documents/mlhalos_files/stored_files/radii_stuff/radii_properties_in_ids.npy")
radii_properties_out = np.load(
    "/Users/lls/Documents/mlhalos_files/stored_files/radii_stuff/radii_properties_out_ids.npy")
fraction = np.concatenate((radii_properties_in[:, 2], radii_properties_out[:, 2]))
ids_in_halo = np.concatenate((radii_properties_in[:, 0], radii_properties_out[:, 0]))

ids_sort = ids_in_halo[np.argsort(ids_in_halo)]
f_sort = fraction[np.argsort(ids_in_halo)]

f_tested = f_sort[np.in1d(ids_sort, test_ids)]
n_tested, b_rad = np.histogram(f_tested[~np.isinf(f_tested)], bins=np.linspace(0, 5, 20))
mass_tested, b_mass = np.histogram(x, bins=20)


######## IDS THAT FROM MISCLASSIFIED BECAME CORRECTLY CLASSIFIED ########

# corrected with inertia

corrected_ids_inertia = get_ids_from_falsein1_to_truein2(y_inden, y_den, x, test_ids)
corrected_m_inertia = x[np.in1d(test_ids, corrected_ids_inertia)]
f_corrected_inertia = f_sort[np.in1d(ids_sort, corrected_ids_inertia)]

n_corr_inertia, b_rad = np.histogram(f_corrected_inertia[~np.isinf(f_corrected_inertia)], bins=b_rad)
mass_corrected_inertia, b_mass = np.histogram(corrected_m_inertia, bins=b_mass)

# corrected with shear

corrected_ids_shear = get_ids_from_falsein1_to_truein2(all_log_predicted_mass_shear, y_den, x, test_ids)
corrected_m_shear = x[np.in1d(test_ids, corrected_ids_shear)]
f_corrected_shear = f_sort[np.in1d(ids_sort, corrected_ids_shear)]

n_corr_shear, b_rad = np.histogram(f_corrected_shear[~np.isinf(f_corrected_shear)], bins=b_rad)
mass_corrected_shear, b_mass = np.histogram(corrected_m_shear, bins=b_mass)


######## IDS THAT FROM CORRECTLY CLASSIFIED BECAME MISCLASSIFIED ########

# corrected with inertia

t_to_f_inertia = get_ids_from_truein1_to_falsein2(y_inden, y_den, x, test_ids)
t_to_f_inertia_rad = f_sort[np.in1d(ids_sort, t_to_f_inertia)]
t_to_f_inertia_mass = x[np.in1d(test_ids, t_to_f_inertia)]
num_t_to_f_inertia, b_rad = np.histogram(t_to_f_inertia_rad[~np.isinf(t_to_f_inertia_rad)], bins=b_rad)
mass_t_to_f_inertia, b_mass = np.histogram(t_to_f_inertia_mass, bins=b_mass)

# corrected with shear

t_to_f_shear = get_ids_from_truein1_to_falsein2(all_log_predicted_mass_shear, y_den, x, test_ids)
t_to_f_shear_rad = f_sort[np.in1d(ids_sort, t_to_f_shear)]
t_to_f_shear_mass = x[np.in1d(test_ids, t_to_f_shear)]
num_t_to_f_shear, b_rad = np.histogram(t_to_f_shear_rad[~np.isinf(t_to_f_shear_rad)], bins=b_rad)
mass_t_to_f_shear, b_mass = np.histogram(t_to_f_shear_mass, bins=b_mass)

######## PLOT ########

# # plot vs radius
#
# plt.plot((b_rad[1:] + b_rad[:-1])/2, n_corr/n_tested * 100, label="corrected w inertia", color="g")
# plt.scatter((b_rad[1:] + b_rad[:-1])/2, n_corr/n_tested * 100, color="g")
# plt.plot((b_rad[1:] + b_rad[:-1])/2, n_corr_shear/n_tested * 100, label="corrected w shear", color="r")
# plt.scatter((b_rad[1:] + b_rad[:-1])/2, n_corr_shear/n_tested * 100, color="r")
# plt.ylabel(r"$N_{\mathrm{corr}}/N_{\mathrm{test}} \times 100$")
# plt.xlabel(r"$r/r_{\mathrm{vir}}$")
# plt.legend(loc="best")
# plt.savefig(path_inertia + "inertia_plus_den/corrected_ids_vs_radius.pdf")
#
# # plot vs mass
#
# plt.plot((b_mass[1:] + b_mass[:-1])/2, mass_corrected_inertia/mass_tested * 100, label="corrected w inertia", color="g")
# plt.scatter((b_mass[1:] + b_mass[:-1])/2, mass_corrected_inertia/mass_tested * 100, color="g")
# plt.plot((b_mass[1:] + b_mass[:-1])/2, mass_corrected_shear/mass_tested * 100, label="corrected w shear", color="r")
# plt.scatter((b_mass[1:] + b_mass[:-1])/2, mass_corrected_shear/mass_tested * 100, color="r")
# #plt.axhline(y=0, color="k")
# plt.ylabel(r"$N_{\mathrm{corr}}/N_{\mathrm{test}} \times 100$")
# plt.xlabel(r"$\log (M /\mathrm{M}_{\odot})$")
# plt.savefig(path_inertia + "inertia_plus_den/corrected_ids_vs_mass.pdf")


# plot vs radius

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 5.2))
fig.subplots_adjust(hspace=0.3, left=0.07, right=0.93, top=0.92,wspace=0)

ax1.plot((b_rad[1:] + b_rad[:-1])/2, n_corr_inertia/n_tested * 100, label=r"false --$>$ true", color="g")
ax1.scatter((b_rad[1:] + b_rad[:-1])/2, n_corr_inertia/n_tested * 100, color="g")
ax1.plot((b_rad[1:] + b_rad[:-1])/2, num_t_to_f_inertia/n_tested * 100, label=r"true --$>$ false", color="g", ls="--")
ax1.scatter((b_rad[1:] + b_rad[:-1])/2, num_t_to_f_inertia/n_tested * 100, color="g", marker="x")
ax1.set_ylabel(r"$N_{\mathrm{corr}}/N_{\mathrm{test}} \times 100$")
ax1.set_xlabel(r"$r/r_{\mathrm{vir}}$")
ax1.legend(loc="best")
ax1.set_title("Density + Inertia")

ax2.plot((b_rad[1:] + b_rad[:-1])/2, n_corr_shear/n_tested * 100, label=r"false --$>$ true", color="r")
ax2.scatter((b_rad[1:] + b_rad[:-1])/2, n_corr_shear/n_tested * 100, color="r")
ax2.plot((b_rad[1:] + b_rad[:-1])/2, num_t_to_f_shear/n_tested * 100, label=r"true --$>$ false", color="r", ls="--")
ax2.scatter((b_rad[1:] + b_rad[:-1])/2, num_t_to_f_shear/n_tested * 100, color="r", marker="x")
ax2.set_xlabel(r"$r/r_{\mathrm{vir}}$")
ax2.set_title("Density + Shear")
ax2.legend(loc="best")
plt.savefig(path_inertia + "inertia_plus_den/clf_change_vs_radius.png")
plt.clf()

# plot vs mass

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 5.2))
fig.subplots_adjust(hspace=0.3, left=0.07, right=0.93, top=0.92,wspace=0)

ax1.plot((b_mass[1:] + b_mass[:-1])/2, mass_corrected_inertia/mass_tested * 100, label=r"false --$>$ true", color="g")
ax1.scatter((b_mass[1:] + b_mass[:-1])/2, mass_corrected_inertia/mass_tested * 100, color="g")
ax1.plot((b_mass[1:] + b_mass[:-1])/2, mass_t_to_f_inertia/mass_tested * 100, label=r"true --$>$ false", color="g", ls="--")
ax1.scatter((b_mass[1:] + b_mass[:-1])/2, mass_t_to_f_inertia/mass_tested * 100, color="g", marker="x")
ax1.set_ylabel(r"$N_{\mathrm{corr}}/N_{\mathrm{test}} \times 100$")
ax1.set_xlabel(r"$\log (M /\mathrm{M}_{\odot})$")
ax1.legend(loc="best")
ax1.set_title("Density + Inertia")
# plt.clf()

ax2.plot((b_mass[1:] + b_mass[:-1])/2, mass_corrected_shear/mass_tested * 100, label=r"false --$>$ true", color="r")
ax2.scatter((b_mass[1:] + b_mass[:-1])/2, mass_corrected_shear/mass_tested * 100, color="r")
ax2.plot((b_mass[1:] + b_mass[:-1])/2, mass_t_to_f_shear/mass_tested * 100, label=r"true --$>$ false", color="r",
         ls="--")
ax2.scatter((b_mass[1:] + b_mass[:-1])/2, mass_t_to_f_shear/mass_tested * 100, color="r", marker="x")
ax2.set_xlabel(r"$\log (M /\mathrm{M}_{\odot})$")
ax2.set_title("Density + Shear")
ax2.legend(loc="best")
plt.savefig(path_inertia + "inertia_plus_den/clf_change_vs_mass.png")