import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/lls/Documents/mlhalos_code')
from utils import classification_results as rmbins
from mlhalos import machinelearning as ml
from mlhalos import distinct_colours


results = np.load("/Users/lls/Documents/CODE/stored_files/all_out/classification_results.npy")

# Take classification results located at /Users/lls/Documents/CODE/stored_files/all_out and plot ROC curves for the
# three mass bins separately

radii_properties_in = np.load("/Users/lls/Documents/CODE/stored_files/all_out/radii_files/radii_properties_in_ids.npy")
radii_properties_out = np.load("/Users/lls/Documents/CODE/stored_files/all_out/radii_files/radii_properties_out_ids"
                               ".npy")

fraction = np.concatenate((radii_properties_in[:,2],radii_properties_out[:,2]))
radius = np.concatenate((radii_properties_in[:,1],radii_properties_out[:,1]))
ids_in_halo = np.concatenate((radii_properties_in[:,0],radii_properties_out[:,0]))

f, h = rmbins.load_final_snapshot_and_halos()
ids_no_halo = f['iord'][f['grp'] == -1]


# 30% radius

ids_30_in_halo = ids_in_halo[(fraction < 0.3) & (radius > 25.6)]
ids_30 = np.concatenate((ids_30_in_halo, ids_no_halo))

ids_30_classified = results[:, 0][np.in1d(results[:,0], ids_30)]
true_label_ids_30 = results[:, 1][np.in1d(results[:,0], ids_30_classified)]
predicted_probabilities_ids_30 = results[:, 2:4][np.in1d(results[:,0],ids_30_classified)]
fpr_30, tpr_30, auc_30, threshold = ml.roc(predicted_probabilities_ids_30, true_label_ids_30)


# 30-60% radius

ids_30_60_in_halo = ids_in_halo[(fraction > 0.3) & (fraction < 0.6) & (radius > 25.6)]
ids_30_60 = np.concatenate((ids_30_60_in_halo, ids_no_halo))

ids_30_60_classified = results[:, 0][np.in1d(results[:,0], ids_30_60)]
true_label_ids_30_60 = results[:, 1][np.in1d(results[:,0], ids_30_60_classified)]
predicted_probabilities_ids_30_60 = results[:, 2:4][np.in1d(results[:,0],ids_30_60_classified)]
fpr_30_60, tpr_30_60, auc_30_60, threshold = ml.roc(predicted_probabilities_ids_30_60, true_label_ids_30_60)

# 60-100% radius

ids_60_100_in_halo = ids_in_halo[(fraction > 0.6) & (fraction < 1) & (radius > 25.6)]
ids_60_100 = np.concatenate((ids_60_100_in_halo, ids_no_halo))

ids_60_100_classified = results[:, 0][np.in1d(results[:,0], ids_60_100)]
true_label_ids_60_100 = results[:, 1][np.in1d(results[:,0], ids_60_100_classified)]
predicted_probabilities_ids_60_100 = results[:, 2:4][np.in1d(results[:,0],ids_60_100_classified)]
fpr_60_100, tpr_60_100, auc_60_100, threshold = ml.roc(predicted_probabilities_ids_60_100, true_label_ids_60_100)

# original

fpr, tpr, auc, threshold = ml.roc(results[:,2:4], results[:,1])



def plot():
    figure, ax = plt.subplots(figsize=(8,6))
    col = distinct_colours.get_distinct(4)
    ax.plot(fpr, tpr, color='k', label="original (auc " + str(float('%.3g' % auc)) + ")")
    ax.plot(fpr_30, tpr_30, color=col[0],
            label=r"30\% (auc " + str(float('%.3g' % auc_30)) + ")")
    ax.plot(fpr_30_60, tpr_30_60, color=col[1], label=r"30\%-60\% (auc " + str(float('%.3g' % auc_30_60))+ ")")
    ax.plot(fpr_60_100, tpr_60_100, color=col[3], label=r"60\%-100\% (auc " + str(float('%.3g' % auc_60_100))+ ")")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc=4)
    return figure

f = plot()
# f.savefig("/Users/lls/Documents/CODE/stored_files/radii_stuff/roc.pdf")
