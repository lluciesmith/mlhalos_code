import numpy as np
import sys
sys.path.append('/home/lls/mlhalos_code')
from mlhalos import parameters
from mlhalos import machinelearning as ml
import matplotlib.pyplot as plt

ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE/", load_final=True)

p = np.load("/Users/lls/Documents/CODE/stored_files/all_out/radii_files/balanced_training_set/predicted_den.npy")
t = np.load("/Users/lls/Documents/CODE/stored_files/all_out/radii_files/balanced_training_set/true_den.npy")
testing_indices = np.load("/Users/lls/Documents/CODE/stored_files/all_out/radii_files/balanced_training_set"
                          "/testing_features_indices.npy")
ids_all = np.concatenate((ic.ids_IN, ic.ids_OUT))
ids_tested = ids_all[testing_indices]

r_in = np.load("/Users/lls/Documents/CODE/stored_files/all_out/radii_files/radii_properties_in_ids.npy")
r_out = np.load("/Users/lls/Documents/CODE/stored_files/all_out/radii_files/radii_properties_out_ids.npy")
ids_in_halo = np.concatenate((r_in[:, 0], r_out[:, 0]))
fraction = np.concatenate((r_in[:, 2], r_out[:, 2]))
radius = np.concatenate((r_in[:,1], r_out[:,1]))

f = ic.final_snapshot
ids_no_halo = f['iord'][f['grp'] == -1]

# 30% radius

ids_30_in_halo = ids_in_halo[(fraction <= 0.3) & (radius > 25.6)]
ids_30 = np.concatenate((ids_30_in_halo, ids_no_halo))

ids_30_classified = ids_tested[np.in1d(ids_tested, ids_30)]
true_label_ids_30 = t[np.in1d(ids_tested, ids_30_classified)]
predicted_probabilities_ids_30 = p[np.in1d(ids_tested, ids_30_classified)]
fpr_30, tpr_30, auc_30, threshold = ml.roc(predicted_probabilities_ids_30, true_label_ids_30)

# 30-60% radius

ids_30_60_in_halo = ids_in_halo[(fraction > 0.3) & (fraction <= 0.6) & (radius > 25.6)]
ids_30_60 = np.concatenate((ids_30_60_in_halo, ids_no_halo))

ids_30_60_classified = ids_tested[np.in1d(ids_tested, ids_30_60)]
true_label_ids_30_60 = t[np.in1d(ids_tested, ids_30_60_classified)]
predicted_probabilities_ids_30_60 = p[np.in1d(ids_tested, ids_30_60_classified)]
fpr_30_60, tpr_30_60, auc_30_60, threshold = ml.roc(predicted_probabilities_ids_30_60, true_label_ids_30_60)

# 60-100% radius

ids_60_100_in_halo = ids_in_halo[(fraction > 0.6) & (fraction < 1) & (radius > 25.6)]
ids_60_100 = np.concatenate((ids_60_100_in_halo, ids_no_halo))

ids_60_100_classified = ids_tested[np.in1d(ids_tested, ids_60_100)]
true_label_ids_60_100 = t[np.in1d(ids_tested, ids_60_100_classified)]
predicted_probabilities_ids_60_100 = p[np.in1d(ids_tested, ids_60_100_classified)]
fpr_60_100, tpr_60_100, auc_60_100, threshold = ml.roc(predicted_probabilities_ids_60_100, true_label_ids_60_100)

# orig

fpr_orig, tpr_orig, auc_orig, threshold = ml.roc(p, t)

# plot

figure, ax = plt.subplots(ncols=1, nrows=1)
col_30 = "#333300"
col_60 = "#669900"
col_100 = "#e69900"

ax.plot(fpr_orig, tpr_orig, color="grey", ls='--', lw=1.5)
ax.plot(fpr_30, tpr_30, color=col_30, lw=1.5, label="Inner (AUC = " + '%.3f' % auc_30 + ")")
ax.plot(fpr_30_60, tpr_30_60, color=col_60, lw=1.5, label="Mid (AUC = " + '%.3f' % auc_30_60 + ")")
ax.plot(fpr_60_100, tpr_60_100, color=col_100, lw=1.5, label="Outer (AUC = " + '%.3f' % auc_60_100 + ")")

ax.legend(loc=(0.35, 0.1), fontsize=15, frameon=False)
ax.set_xlabel('False Positive Rate', fontsize=17)
ax.set_ylabel('True Positive Rate', fontsize=17)
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)

plt.savefig("/Users/lls/Documents/CODE/stored_files/all_out/radii_files/balanced_training_set/roc_radius_bins.pdf")


