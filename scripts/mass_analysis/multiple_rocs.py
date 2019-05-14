from mlhalos import machinelearning as ml
from scripts.EPS import EPS_predictions
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils import mass
from mlhalos import distinct_colours
from utils import classification_results as res


def multiple_rocs(y_proba, y_true, threshold_l, true_class=1, color='b', label="", with_EPS=False, fpr_EPS=None,
                  tpr_EPS=None):
    """Get ROC plot given predicted probability scores of classes and true classes for samples."""
    fpr, tpr, auc, threshold = ml.roc(y_proba, y_true, true_class=true_class)

    plt.scatter(fpr[threshold == threshold_l], tpr[threshold == threshold_l], marker='o', color=color, s=50)
    plt.plot(fpr, tpr, label=label + " (auc " + str(float('%.3g' % auc)) + ")", color=color)

    if with_EPS is True:
        plt.scatter(fpr_EPS, tpr_EPS, marker='^', color=color, s=50)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.legend(loc=4)


if __name__ == "__main__":
    ids, p, t = res.load_classification_results()
    p = np.load("/Users/lls/Documents/CODE/stored_files/all_out/predicted_probabilities.npy")
    t = np.load("/Users/lls/Documents/CODE/stored_files/all_out/true_labels.npy")

    threshold, fnr = ml.false_negative_rate(p, t)
    fn_EPS = EPS_predictions.get_subset_classified_particles_EPS(particles="false negatives")
    fnr_EPS = len(fn_EPS) / len(t[t == 1])
    threshold_limit = threshold[fnr < fnr_EPS][0]

    fp_EPS = EPS_predictions.get_subset_classified_particles_EPS(particles="false positives")
    tp_EPS = EPS_predictions.get_subset_classified_particles_EPS(particles="true positives")
    fpr_EPS = len(fp_EPS)/ len(t[t == -1])
    tpr_EPS = len(tp_EPS)/ len(t[t == 1])

    ids = np.load("/Users/lls/Documents/CODE/stored_files/all_out/classified_ids.npy")
    ids_in = ids[t == 1]
    ids_out = ids[t== -1]
    mass_ins = mass.get_halo_mass_each_particle(ids_in)
    bool = mass_ins < 7e12

    p_in = p[t == 1]
    t_in = t[t == 1]
    p_out = p[t == -1]
    t_out = t[t == -1]

    # small mass

    ids_small_in = ids_in[bool]
    ids_small_out = ids_out[bool]
    p_small_in = p_in[bool]
    t_small_in = t_in[bool]
    p_small = np.concatenate((p_small_in, p_out))
    t_small = np.concatenate((t_small_in, t_out))

    fp_EPS_small = EPS_predictions.get_subset_classified_particles_EPS(particles="false positives",
                                                                    ids_subset=ids_out)
    tp_EPS_small = EPS_predictions.get_subset_classified_particles_EPS(particles="true positives",
                                                                     ids_subset=ids_small_in)
    fpr_EPS_small = len(fp_EPS_small)/ len(t_small[t_small == -1])
    tpr_EPS_small = len(tp_EPS_small)/ len(t_small[t_small == 1])

    # high mass

    ids_high_in = ids_in[~bool]
    p_high_in = p_in[~bool]
    t_high_in = t_in[~bool]
    p_high = np.concatenate((p_high_in, p_out))
    t_high = np.concatenate((t_high_in, t_out))

    fp_EPS_high = EPS_predictions.get_subset_classified_particles_EPS(particles="false positives",
                                                                    ids_subset=ids_out)
    tp_EPS_high = EPS_predictions.get_subset_classified_particles_EPS(particles="true positives",
                                                                     ids_subset=ids_high_in)
    fpr_EPS_high = len(fp_EPS_high)/ len(t_high[t_high == -1])
    tpr_EPS_high = len(tp_EPS_high)/ len(t_high[t_high == 1])

    fpr, tpr, auc, threshold = ml.roc(p, t)

    c = distinct_colours.get_distinct(2)

    plt.figure(figsize=(8, 6))
    multiple_rocs(p, t, threshold_limit, label="all", with_EPS=True, fpr_EPS=fpr_EPS, tpr_EPS=tpr_EPS)
    multiple_rocs(p_small, t_small, threshold_limit, label=r"$M < 7 \times 10^{12} \mathrm{M_\odot}$", color=c[0],
                  with_EPS=True, fpr_EPS=fpr_EPS_small, tpr_EPS=tpr_EPS_small)
    multiple_rocs(p_high, t_high, threshold_limit, label=r"$M > 7 \times 10^{12} \mathrm{M_\odot}$", color=c[1],
                  with_EPS=True, fpr_EPS=fpr_EPS_high, tpr_EPS=tpr_EPS_high)
    #plt.axvline(fpr_EPS, ls='--', color='k', label="EPS")
    #plt.axhline(tpr_EPS, ls='--', color='k')

    # plt.legend(loc=4)

    th = plt.scatter([0],[0], color='k', marker='o', label='0.429 threshold')
    #EPS, = matplotlib.lines.Line2D([0],[0], linestyle="none", color='k', marker='^', label='EPS')
    EPS = plt.scatter([0],[0], color='k', marker='^', label='EPS')

    plt.legend(loc=4)
    #
    # legend1 = plt.legend([th, EPS], ["0.429 threshold", "EPS"],
    #                      loc=2)
    # plt.gca().add_artist(legend1)
    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.savefig("/Users/lls/Desktop/rocs_small_high_EPS.pdf")
