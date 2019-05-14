import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/lls/Documents/mlhalos_code/")
import numpy as np

from mlhalos import plot
from mlhalos import machinelearning as ml
from mlhalos import parameters
from mlhalos import distinct_colours

path = "/Users/lls/Documents/CODE/stored_files/shear/classification/"
#path = "/home/lls/stored_files/shear_quantities/"

def get_testing_index():
    training_index = np.load("/Users/lls/Documents/CODE/stored_files/all_out/50k_features_index.npy")
    testing_index = ~np.in1d(np.arange(256 ** 3), training_index)
    return testing_index


def get_training_features(features):
    training_index = np.load("/Users/lls/Documents/CODE/stored_files/all_out/50k_features_index.npy")
    training_features = features[training_index]
    return training_features


def get_testing_features(features):
    testing_index = get_testing_index()
    testing_features = features[testing_index]
    return testing_features


def get_false_positives(ids, y_predicted, y_true, threshold=None):
    if threshold is None:
        threshold = 0.5

    labels = y_predicted[:, 1] >= threshold
    y_bool = (y_true == 1)

    FPs = ids[labels & ~y_bool]
    return FPs


def get_false_negatives(ids, y_predicted, y_true, threshold=None):
    if threshold is None:
        threshold = 0.5

    labels = y_predicted[:, 1] >= threshold
    y_bool = (y_true == 1)

    FNs = ids[~labels & y_bool]
    return FNs


def get_false_positives_index(y_predicted, y_true, threshold=None):
    if threshold is None:
        threshold = 0.5

    labels = y_predicted[:, 1] >= threshold
    y_bool = (y_true == 1)

    ind = (labels & ~y_bool)
    return ind


def get_false_negatives_index(y_predicted, y_true, threshold=None):
    if threshold is None:
        threshold = 0.5

    labels = y_predicted[:, 1] >= threshold
    y_bool = (y_true == 1)

    ind = (~labels & y_bool)
    return ind


def get_percentage_false_positives_in_out_halos_per_threshold(y_predicted, y_true, threshold, halos_particles):
    shape = len(threshold)

    FPs_in_halos = np.zeros(shape, )
    FPs_out_halos = np.zeros(shape, )

    for i in range(shape):
        FPs = get_false_positives_index(y_predicted, y_true, threshold=threshold[i])

        in_h = np.where(halos_particles[FPs] > 0)[0]
        out_h = np.where(halos_particles[FPs] == 0)[0]
        tot = halos_particles[FPs]

        if len(in_h) != 0:
            FPs_in_halos[i] = len(in_h) / len(tot)
        else:
            FPs_in_halos[i] = len(in_h)

        if len(out_h) != 0:
            FPs_out_halos[i] = len(out_h) / len(tot)
        else:
            FPs_out_halos[i] = len(out_h)

        if (len(in_h) != 0) & (len(out_h) != 0):
            assert FPs_in_halos[i] + FPs_out_halos[i] == 1

    return FPs_in_halos, FPs_out_halos


def false_positives_ids_index_per_threshold(y_predicted, y_true, threshold_list):
    FPs = np.array([get_false_positives_index(y_predicted, y_true, threshold=threshold_list[i])
                    for i in range(len(threshold_list))])
    return FPs


def plot_difference_fraction_FPs(FPs_run_one, FPs_run_two,
                                 title="Fraction FPs in density - FPs density + den-sub ellipticity",
                                 label="belong to halos"):
    colors= distinct_colours.get_distinct(2)
    plt.plot(threshold, FPs_run_one - FPs_run_two, label=label, color=colors[0])
    plt.axhline(y=0, color="k")
    plt.xlabel("Threshold")
    plt.ylabel(r"\mathrm{\Delta} \mathrm{FPs}")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()


def get_ids_and_halos_test_set():
    # halos are ordered such that each halo mass corresponds to particles in order of particle ID (0,1,2,...)
    # Need to reorder the array such that it has first all IN particles, the all OUT particles

    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/")
    ids = np.concatenate((ic.ids_IN, ic.ids_OUT))
    halos = np.load("/Users/lls/Documents/CODE/stored_files/halo_mass_particles.npy")

    testing_index = get_testing_index()
    ids_tested = ids[testing_index]
    halos_testing_particles = halos[ids_tested]

    return ids_tested, halos_testing_particles


# Load predictions

y_pred_den_sub_ell = np.load(path + "predicted_den+den_sub_ell.npy")
y_true_den_sub_ell  = np.load(path + "true_den+den_sub_ell.npy")

y_pred_den = np.load(path + "predicted_den.npy")
y_true_den = np.load(path + "true_den.npy")
assert (y_true_den_sub_ell  == y_true_den).all()


############# FALSE POSITIVES #############

# Find FPs of density run and FPs of denisty+density-subtracted ellipticity run

ids_tested, halos_testing_particles = get_ids_and_halos_test_set()

FPs_den_den_sub_ell_ind = get_false_positives_index(y_pred_den_sub_ell, y_true_den_sub_ell)
halos_FPs_den_plus_den_sub_ell = halos_testing_particles[FPs_den_den_sub_ell_ind]

FPs_den_ind = get_false_positives_index(y_pred_den, y_true_den)
halos_FPs_den = halos_testing_particles[FPs_den_ind]


############# FALSE NEGATIVES #############

# Find FNs of density run and FNs of denisty+density-subtracted ellipticity run

FNs_den_den_sub_ell_ind = get_false_negatives_index(y_pred_den_sub_ell, y_true_den_sub_ell)
halos_FNs_den_plus_den_sub_ell = halos_testing_particles[FNs_den_den_sub_ell_ind]

FNs_den_ind = get_false_negatives_index(y_pred_den, y_true_den)
halos_FNs_den = halos_testing_particles[FNs_den_ind]

h_400_mass = 1836194204280.7886


############## PLOTS #################


# Plot which halos do the FPs live in in the density run and the density + density-subtracted ellipticity run

n, bins, patch = plt.hist(np.log10(halos_FPs_den_plus_den_sub_ell[halos_FPs_den_plus_den_sub_ell > 0]), bins=20,
                                   #log=True,
                  histtype="step", label="density + den-sub ellipticity")
n1, bins1, patch1 = plt.hist(np.log10(halos_FPs_den[halos_FPs_den > 0]), bins=bins,
                             #log=True,
                             histtype="step", label="density")
plt.legend(loc="best")


# Plot difference in fraction of FPs in halos as a function of threshold for density-only and density+den-sub
# ellipticity case

threshold = np.linspace(0., 1., 50)[::-1]
FPs_den_in_halos, FPs_den_out_halos = get_percentage_false_positives_in_out_halos_per_threshold(y_pred_den,
                                                                                                y_true_den,
                                                                                                threshold,
                                                                                                halos_testing_particles)
FPs_den_sub_ell_in_halos, FPs_den_sub_ell_out_halos = get_percentage_false_positives_in_out_halos_per_threshold(
    y_pred_den_sub_ell, y_true_den, threshold, halos_testing_particles)

plot_difference_fraction_FPs(FPs_den_out_halos, FPs_den_sub_ell_out_halos, label="not belong to halos")


# Which FPs went from wrong to right classification?

FPs_den_ind = false_positives_ids_index_per_threshold(y_pred_den, y_true_den, threshold)
FPs_den_den_sub_ell_ind = false_positives_ids_index_per_threshold(y_pred_den_sub_ell, y_true_den, threshold)

assert FPs_den_ind.shape[0] == len(threshold)
assert FPs_den_den_sub_ell_ind.shape[0] == len(threshold)

common_FPs = np.array([ids_tested[FPs_den_ind[i] & FPs_den_den_sub_ell_ind[i]] for i in range(len(threshold))])
wrong_to_correct_density_FPs = np.array([ids_tested[FPs_den_ind[i] & ~FPs_den_den_sub_ell_ind[i]]
                                         for i in range(len(threshold))])

halos_common_FPs = np.array([halos_testing_particles[FPs_den_ind[i] & FPs_den_den_sub_ell_ind[i]]
                             for i in range(len(threshold))])
halos_wrong_to_correct_density_FPs = np.array([halos_testing_particles[FPs_den_ind[i] & ~FPs_den_den_sub_ell_ind[i]]
                                         for i in range(len(threshold))])

frac_w_to_cor = np.zeros((50, 2))

for i in range(len(threshold)):
    in_h = len(np.where(halos_wrong_to_correct_density_FPs[i] > 0)[0])
    out_h = len(np.where(halos_wrong_to_correct_density_FPs[i] == 0)[0])
    if in_h != 0:
        in_halos = in_h/ len(halos_wrong_to_correct_density_FPs[i])
    else:
        in_halos = in_h
    if out_h != 0:
        out_halos = out_h / len(halos_wrong_to_correct_density_FPs[i])
    else:
        out_halos = out_h

    frac_w_to_cor[i,0] = in_halos
    frac_w_to_cor[i, 1] = out_halos

plt.figure(figsize=(8,6))
plt.plot(threshold, frac_w_to_cor[:,0], label="in halos")
plt.axhline(y=0.5, color="k", ls="--")
#plt.plot(threshold, frac_w_to_cor[:,1], label="out halos")
plt.xlabel("Threshold")
plt.ylabel("Fraction FPs in halos")
plt.legend(loc="best")
plt.title("FPs in density run correctly classified by den+ell run")
plt.tight_layout()


for i in range(len(wrong_to_correct_density_FPs)):
    w_to_c_threshold = wrong_to_correct_density_FPs[i]

    w_to_c_threshold_in_halos = w_to_c_threshold[halos_testing_particles[w_to_c_threshold]>0]
    w_to_c_threshold_out_halos = w_to_c_threshold[halos_testing_particles[w_to_c_threshold]==0]

    print("Threshold " + str(threshold[i]) + str(" :"))
    print("The fraction of false positives IN HALOS which flipped from wrongly to correctly classified is  " +
          str(len(w_to_c_threshold_in_halos)/len(w_to_c_threshold)))
    print("The fraction of false positives OUT HALOS which flipped from wrongly to correctly classified is  " +
          str(len(w_to_c_threshold_out_halos)/len(w_to_c_threshold)))








# plt.plot(threshold, FPs_den_in_halos, label="density")
# plt.plot(threshold, FPs_den_plus_den_sub_ell_in_halos, label="den + den-sub ell")
# plt.xlabel("Threshold")
# plt.ylabel("Number of FPs")
# plt.yscale("log")
# plt.legend(loc="best")
# plt.title("FPs not belonging to a halo")
# plt.tight_layout()


# fpr_den, tpr_den, auc_den, th = ml.roc(y_pred_den, y_true_den)
# fpr_den_sub_ell, tpr_den_sub_ell, auc_den_sub_ell, th = ml.roc(y_pred_den_sub_ell, y_true_den_sub_ell)
# plot.roc_plot(np.column_stack((fpr_den,fpr_den_sub_ell)), np.column_stack((tpr_den,tpr_den_sub_ell )),
#               [auc_den,auc_den_sub_ell], labels=["density", "den + den-sub ell"])

# plt.savefig(path + "FPs_in_no_halo.pdf")



