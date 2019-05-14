import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/scripts')
from mlhalos import machinelearning as ml
from mlhalos import distinct_colours


# original

results = np.load("/Users/lls/Documents/CODE/stored_files/classification/classification_results.npy")
fpr, tpr, auc, threshold = ml.roc(results[:,2:4], results[:,1])


# with EPS label

pred_proba_EPS_label = np.load("/Users/lls/Documents/CODE/stored_files/with_EPS/predicted_probabilities"
                               ".npy")
true_labels_EPS_label = np.load("/Users/lls/Documents/CODE/stored_files/with_EPS/true_labels.npy")

fpr_w_EPS, tpr_w_EPS, auc_w_EPS, threshold = ml.roc(pred_proba_EPS_label, true_labels_EPS_label)


# plot

def plot():
    figure, ax = plt.subplots(figsize=(8,6))
    col = distinct_colours.get_distinct(1)
    ax.plot(fpr, tpr, color='k', label="original (auc " + str(float('%.3g' % auc)) + ")")
    ax.plot(fpr_w_EPS, tpr_w_EPS, color=col[0],
            label="EPS-label (auc " + str(float('%.3g' % auc_w_EPS)) + ")")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc=4)
    return figure

f = plot()
f.savefig("/Users/lls/Documents/CODE/stored_files/with_EPS/roc_with_EPS.pdf")

