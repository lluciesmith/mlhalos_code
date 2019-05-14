import numpy as np
import sys
sys.path.append("/home/lls/mlhalos_code")
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from mlhalos import machinelearning as ml
import matplotlib as mpl
from mlhalos import plot
from mlhalos import parameters
from mlhalos import window

path = "/Users/lls/Documents/mlhalos_files/multiclass/"
p_ics_only = np.load(path + "ics_only/fpr_tpr_auc.npy")
FPR_ics_only, TPR_ics_only, AUC_ics_only = p_ics_only[:,:50], p_ics_only[:, 50:100], p_ics_only[:,-1]
x_ics = FPR_ics_only.transpose()
y_ics = TPR_ics_only.transpose()

p_z0 = np.load(path + "ics_z0_density/fpr_tpr_auc.npy")
FPR_z0, TPR_z0, AUC_z0 = p_z0[:,:50], p_z0[:, 50:100], p_z0[:,-1]
x_z0 = FPR_z0.transpose()
y_z0 = TPR_z0.transpose()

p_z0_only = np.load(path + "z0_only/fpr_tpr_auc_z0_only.npy")
FPR_z0_only, TPR_z0_only, AUC_z0_only = p_z0_only[:,:50], p_z0_only[:, 50:100], p_z0_only[:,-1]
x_z0_only = FPR_z0_only.transpose()
y_z0_only = TPR_z0_only.transpose()

t = np.load(path + "classes_test_set.npy")

halo_mass = np.load("/Users/lls/Documents/mlhalos_files/halo_mass_particles.npy")
log_halo_mass_in_ids = np.log10(halo_mass[halo_mass != 0])
bins = np.concatenate((np.linspace(log_halo_mass_in_ids.min(), 14, 15), [log_halo_mass_in_ids.max() + 0.1]))

l = r"$ < \log (M_\mathrm{true}/\mathrm{M}_{\odot}) < $"
for i in range(15):
    fpr = np.column_stack((x_ics[:, i], x_z0[:, i], x_z0_only[:,i]))
    tpr = np.column_stack((y_ics[:, i], y_z0[:, i], y_z0_only[:, i]))
    auc = [AUC_ics_only[i], AUC_z0[i], AUC_z0_only[i]]

    f = plot.roc_plot(fpr, tpr, auc, labels=["$z=99$ densities", "$z=99$ + $z=0$ densities", "$z=0$ only"],
                      title=' %.2f' % (bins[i]) + l + ' %.2f' % (bins[i+1]))
    f.subplots_adjust(top=0.9)
    plt.savefig("/Users/lls/Documents/mlhalos_files/multiclass/z0_only/roc_class_" + str(i) + ".png")
    del f

# importances

ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/mlhalos_files", load_final=True)
w = window.WindowParameters(initial_parameters=ic)
# f_imp_z0 = np.load("/Users/lls/Documents/mlhalos_files/multiclass/z0_only/f_imp.npy")
# f_imp_z0_ics = np.load("/Users/lls/Documents/mlhalos_files/multiclass/ics_z0_density/f_imp.npy")
f_imp = np.load("/Users/lls/Documents/mlhalos_files/multiclass/ics_only/f_imp.npy")

for i in range(50):
    # f = plot.plot_importances_vs_mass_scale(f_imp_z0[i], w.smoothing_masses, save=False, yerr=None,
    #                                         label="$z=0$ density", path=".",
    #                                     title=' %.2f' % (bins[i]) + l + ' %.2f' % (bins[i+1]), width=0.5, log=False,
    #                                     subplots=1, figsize=(6.9, 5.2), frameon=False, legend_fontsize=None, ecolor="k")
    # plt.savefig("/Users/lls/Documents/mlhalos_files/multiclass/z0_only/imp_" + str(i) + ".png")
    # del f
    # plt.clf()
    # f = plot.plot_importances_vs_mass_scale(f_imp_z0_ics[i], w.smoothing_masses, save=False, yerr=None,
    #                                     label=["ics density", "$z=0$ density"], path=".",
    #                                     title=' %.2f' % (bins[i]) + l + ' %.2f' % (bins[i+1]), width=0.5, log=False,
    #                                     subplots=2, figsize=(6.9, 5.2), frameon=False, legend_fontsize=None, ecolor="k")
    # plt.savefig("/Users/lls/Documents/mlhalos_files/multiclass/ics_z0_density/imp_" + str(i) + ".png")
    # plt.clf()
    # del f
    f = plot.plot_importances_vs_mass_scale(f_imp[i], w.smoothing_masses, save=False, yerr=None,
                                         label="$z=99$ density", path=".",
                                        title=' %.2f' % (bins[i]) + l + ' %.2f' % (bins[i+1]), width=0.5, log=False,
                                        subplots=1, figsize=(6.9, 5.2), frameon=False, legend_fontsize=None, ecolor="k")
    plt.savefig("/Users/lls/Documents/mlhalos_files/multiclass/ics_only/imp_" + str(i) + ".png")
    plt.clf()
    del f

