import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
#from mlhalos import plot
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


def plot_importances(imp, label="1000 trees, depth 2", m=None, width=None, title=r"Halos $12.5 < \log M \leq 13.5$"):
    if m is None:
        m = np.linspace(np.log10(3e10), np.log10(1e15), 50)[:-1]
        width = np.append(np.diff(m), np.diff(m)[-1])[:-1]

    plot.plot_importances_vs_mass_scale(imp, 10 ** m, width=width, label=label,
                                        title=title, subplots=1, figsize=(10, 5))
    plt.axvline(x=10 ** 13.5, color="grey", ls="--")
    plt.axvline(x=10**12.5, color="grey", ls="--")
    plt.legend(loc="best", fontsize=16)
    plt.subplots_adjust(bottom=0.15, top=0.9)
    # plt.ylim(0, 0.2)


traj = np.load("/Users/lls/Documents/mlhalos_files/regression/features_w_periodicity_fix/ics_density_contrasts.npy")
training_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/"
                       "ic_traj/nest_2000_lr006/training_ids.npy")
halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")

# ids_between_125_135 = np.where((np.log10(halo_mass[training_ids]) > 12.5) &
#                                         (np.log10(halo_mass[training_ids]) <= 13.5))[0]
#
# features_training = traj[training_ids[ids_between_125_135], :-1]
# truth_training = np.log10(halo_mass[training_ids[ids_between_125_135]])

training_ids_2 = np.random.choice(np.where((np.log10(halo_mass) > 12.5) &
                                        (np.log10(halo_mass) <= 13.5 ))[0],
                                  100000, replace=False)

features_training = traj[training_ids_2, :-1]
truth_training = np.log10(halo_mass[training_ids_2])

# clf_10 = GradientBoostingRegressor(n_estimators=1000, max_features=0.8, subsample=0.8, learning_rate=0.01,
#                                    max_depth=3, loss="lad")
# clf_10.fit(features_training, truth_training)
#
# plot_importances(clf_10.feature_importances_)
# plt.subplots_adjust(bottom=0.15, top=0.9)
# plt.savefig("/Users/lls/Desktop/log_m_125_135_bug_fixed_sklearng")


#### First compute the number of trees by setting an early stopping convergence criterion
# s.t. the algorithm stops if within the last 10 iterations it has not improved by 10^-4
# the score of the validation set

clf_10 = GradientBoostingRegressor(n_estimators=1000, max_features=0.8, subsample=0.8, learning_rate=0.01, max_depth=3,
                                   loss="lad", random_state=3, validation_fraction=0.1, n_iter_no_change=10, tol=0.0001)
clf_10.fit(features_training, truth_training)

print("This is the number of trees " + str(clf_10.n_estimators_))

gbm_base = GradientBoostingRegressor(n_estimators=clf_10.n_estimators_, loss="lad",learning_rate=0.01,
                                     validation_fraction=0.1, n_iter_no_change=10, tol=0.0001)
param_grid = {#"learning_rate": [0.04, 0.01, 0.008],
              "max_depth": [3, 5, 8],
              "max_features": ["sqrt", 0.3, 0.5, 0.8],
              "min_samples_leaf": [0.05, 0.01, 0.1]
              }
gbm = GridSearchCV(estimator=gbm_base, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1,
                   scoring="neg_mean_absolute_error")
gbm.fit(features_training, truth_training)

joblib.dump(gbm, "/Users/lls/Desktop/clf.pkl")


gbm_bestest2 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.01, loss='lad', max_depth=5, max_features=0.5,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=0.05,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=462, n_iter_no_change=10, presort='auto',
             random_state=None, subsample=1., tol=0.0001,
             validation_fraction=0.1, verbose=0, warm_start=False)

clf_10 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.01, loss='lad', max_depth=3, max_features=0.5,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=0.1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=400, n_iter_no_change=None, presort='auto',
             random_state=None, subsample=0.8, tol=0.0001,
             validation_fraction=0.1, verbose=0, warm_start=False)
#
# total_imps = np.zeros((classifier.n_estimators_, 49))
# # total_sum = np.zeros((self.n_features_,), dtype=np.float64)
# for i in range(len(classifier.estimators_)):
#     stage = classifier.estimators_[i]
#     stage_sum = np.sum(tree.tree_.compute_feature_importances(
#         normalize=False) for tree in stage) / len(stage)
#     total_imps[i] = stage_sum
#
# importances = total_sum / len(self.estimators_)
# importances /= importances.sum()

#
# a = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
#              learning_rate=0.01, loss='lad', max_depth=8, max_features=0.5,
#              max_leaf_nodes=None, min_impurity_decrease=0.0,
#              min_impurity_split=None, min_samples_leaf=0.01,
#              min_samples_split=2, min_weight_fraction_leaf=0.0,
#              n_estimators=462, n_iter_no_change=10, presort='auto',
#              random_state=None, subsample=0.8, tol=0.0001,
#              validation_fraction=0.1, verbose=0, warm_start=False)
# a.fit(features_training, truth_training)
# loss_a = np.zeros(a.n_estimators_,)
# for i, pred in enumerate(a.staged_predict(features_training)):
#      loss_a[i] = mae(truth_training, pred)
#
#
# b = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
#              learning_rate=0.01, loss='lad', max_depth=8, max_features=0.5,
#              max_leaf_nodes=None, min_impurity_decrease=0.0,
#              min_impurity_split=None, min_samples_leaf=0.01,
#              min_samples_split=2, min_weight_fraction_leaf=0.0,
#              n_estimators=462, n_iter_no_change=10, presort='auto',
#              random_state=None, subsample=0.5, tol=0.0001,
#              validation_fraction=0.1, verbose=0, warm_start=False)
# b.fit(features_training, truth_training)
# loss_b = np.zeros(b.n_estimators_,)
# for i, pred in enumerate(b.staged_predict(features_training)):
#      loss_b[i] = mae(truth_training, pred)
#
# plt.plot(np.arange(a.n_estimators_), loss_a, label="subsample 1.")
# plt.plot(np.arange(b.n_estimators_), loss_b, label="subsample 0.5")
