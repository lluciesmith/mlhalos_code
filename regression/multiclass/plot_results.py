import numpy as np
import sys
sys.path.append("/home/lls/mlhalos_code")
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from mlhalos import machinelearning as ml
import matplotlib as mpl

################ ONE vs REST ################

t = np.load("/share/data2/lls/multiclass/inertia_plus_den/classes_test_set.npy")

path = "/share/data2/lls/multiclass/lowz/"

p = np.load(path + "ics_density_only/predicted_classes.npy")
# p = np.load("/Users/lls/Documents/mlhalos_files/regression/multiclass/one_vs_rest/predicted_classes.npy")
pred_class = np.argmax(p, axis=1)

y_true = label_binarize(t, classes=list(np.arange(t.max() + 1)))

FPR = np.zeros((15, 50))
TPR = np.zeros((15, 50))
AUC = np.zeros((15,))
for i in range(15):
    FPR[i], TPR[i], AUC[i], threshold = ml.roc(p[:,i], y_true[:,i], true_class=1)

# np.save(path + "ics_density_only/fpr_tpr_auc.npy", np.column_stack((FPR, TPR, AUC)))
# del p
#
#
# p = np.load(path + "ics_z0_density/predicted_classes.npy")
# # p = np.load("/Users/lls/Documents/mlhalos_files/regression/multiclass/one_vs_rest/predicted_classes.npy")
# pred_class = np.argmax(p, axis=1)
#
# y_true = label_binarize(t, classes=list(np.arange(t.max() + 1)))
#
# FPR = np.zeros((15, 50))
# TPR = np.zeros((15, 50))
# AUC = np.zeros((15,))
# for i in range(15):
#     FPR[i], TPR[i], AUC[i], threshold = ml.roc(p[:,i], y_true[:,i], true_class=1)
#
# np.save(path + "ics_z0_density/fpr_tpr_auc.npy", np.column_stack((FPR, TPR, AUC)))
# del p
#
#
# p = np.load(path + "ics_z8_density/predicted_classes.npy")
# # p = np.load("/Users/lls/Documents/mlhalos_files/regression/multiclass/one_vs_rest/predicted_classes.npy")
# pred_class = np.argmax(p, axis=1)
#
# y_true = label_binarize(t, classes=list(np.arange(t.max() + 1)))
#
# FPR = np.zeros((15, 50))
# TPR = np.zeros((15, 50))
# AUC = np.zeros((15,))
# for i in range(15):
#     FPR[i], TPR[i], AUC[i], threshold = ml.roc(p[:,i], y_true[:,i], true_class=1)
#
# np.save(path + "ics_z8p_density/fpr_tpr_auc.npy", np.column_stack((FPR, TPR, AUC)))
# del p




halo_mass = np.load("/Users/lls/Documents/mlhalos_files/halo_mass_particles.npy")
log_halo_mass_in_ids = np.log10(halo_mass[halo_mass != 0])
bins = np.concatenate((np.linspace(log_halo_mass_in_ids.min(), 14, 15), [log_halo_mass_in_ids.max() + 0.1]))

plt.figure(figsize=(13,6))
x = FPR.transpose()
y = TPR.transpose()
norm = mpl.colors.Normalize(vmin=AUC.min(), vmax=AUC.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
cmap.set_array([])
col = [cmap.to_rgba(AUC[i]) for i in range(15)]
for i in range(15):
    l = r"$ < \log (M_\mathrm{true}/\mathrm{M}_{\odot}) < $"
    plt.plot(x[:,i], y[:,i], lw=1.5, label=' %.2f' % (bins[i]) + l + ' %.2f' % (bins[i+1]), color=col[i])
cbar = plt.colorbar(cmap, ticks=np.linspace(AUC.min(), AUC.max(), 4, endpoint=True))
cbar.set_label('AUC')

plt.legend(loc="best", fontsize=12)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim(-0.05, 0.5)
plt.ylim(0.5, 1.05)
plt.axhline(y=1, color="k", ls="--")
plt.axvline(x=0, color="k", ls="--")
plt.savefig("/Users/lls/Documents/mlhalos_files/regression/multiclass/one_vs_rest/rocs.png")

