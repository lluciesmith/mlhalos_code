import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from mlhalos import plot
from sklearn.model_selection import GridSearchCV
import os; os.environ['KMP_DUPLICATE_LIB_OK']='True'
import lightgbm as lgb
from ..plots import plotting_functions as pf


################## FIRST BUILD THE TRAINING SET FROM ORIGINAL SIMULATION ##################

traj = np.load("/Users/lls/Documents/mlhalos_files/regression/features_w_periodicity_fix/ics_density_contrasts.npy")
halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")

training_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/"
                       "ic_traj/nest_2000_lr006/training_ids.npy")

features_training = traj[training_ids, :-1]
truth_training = np.log10(halo_mass[training_ids])

# Validation set from same simulation

all_ids = np.arange(256**3)[halo_mass > 0]
remaining_ids = all_ids[~np.in1d(all_ids, training_ids)]
validation_ids_same_sim = np.random.choice(remaining_ids, 10000, replace=False)

features_val_same_sim = traj[validation_ids_same_sim, :-1]
truth_val_same_sim = np.log10(halo_mass[validation_ids_same_sim])

# Validation set from different simulation

traj_val = np.load("/Users/lls/Documents/mlhalos_files/reseed50/features/density_constrasts.npy")
truth_val = np.load("/Users/lls/Documents/mlhalos_files/reseed50/features/halo_mass_particles.npy")

all_ids_diff_sim = np.arange(256**3)[truth_val > 0]
val_ids_diff_sim = np.random.choice(all_ids_diff_sim, 10000, replace=False)

features_val_diff_sim = traj_val[val_ids_diff_sim, :-1]
truth_val_diff_sim = np.log10(truth_val[val_ids_diff_sim])

# sklearn GBT

param_grid = {"max_depth": [3, 5, 8],
              "max_features": ["sqrt", 0.3, 0.5, 0.8],
              "min_samples_leaf": [0.05, 0.1, 0.3]
              }

gbm_base = GradientBoostingRegressor(n_estimators=500, subsample=0.8, learning_rate=0.05, loss="lad")
gbm_cv = GridSearchCV(estimator=gbm_base, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1,
                   scoring="neg_mean_absolute_error")
gbm_cv.fit(features_training, truth_training)

gbm_bestest = gbm_cv.best_estimator_
imp_sklearn = gbm_bestest.feature_importances_

# gbm_bestest = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
#              learning_rate=0.05, loss='lad', max_depth=8, max_features=0.8,
#              max_leaf_nodes=None, min_impurity_decrease=0.0,
#              min_impurity_split=None, min_samples_leaf=0.05,
#              min_samples_split=2, min_weight_fraction_leaf=0.0,
#              n_estimators=500, n_iter_no_change=None, presort='auto',
#              random_state=None, subsample=0.8, tol=0.0001,
#              validation_fraction=0.1, verbose=0, warm_start=False)
# gbm_bestest.fit(features_training, truth_training)


# compare importances to LGBM

lgb_train = lgb.Dataset(features_training, truth_training)
# lgb_eval = lgb.Dataset(features_val_diff_sim, truth_val_diff_sim, reference=lgb_train)
# lgb_eval = lgb.Dataset(features_val_same_sim, truth_val_same_sim, reference=lgb_train)

params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metric':'l1', 'num_leaves': 60,
          'learning_rate': 0.1, 'feature_fraction': 0.6, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': 0
          }
lgbm_algo = lgb.train(params, lgb_train, num_boost_round=1000)


###### Plot importances ######


def plot_importances(imp, label="1000 trees, depth 2", m=None, width=None, title=r"Halos $12.5 < \log M \leq 13.5$"):
    if m is None:
        m = np.linspace(np.log10(3e10), np.log10(1e15), 50)[:-1]
        width = np.append(np.diff(m), np.diff(m)[-1])[:-1]

    plot.plot_importances_vs_mass_scale(imp, 10 ** m, width=width, label=label,
                                        title=title, subplots=1, figsize=(10, 5))
    plt.legend(loc="best", fontsize=16)
    plt.subplots_adjust(bottom=0.15, top=0.9)
    # plt.ylim(0, 0.2)


imp_lgbm = lgbm_algo.feature_importance("gain")

m = np.linspace(np.log10(3e10), np.log10(1e15), 50)[:-1]
width = np.append(np.diff(10**m), np.diff(10**m)[-1])

plot_importances(gbm_bestest.feature_importances_, label="sklearn", title="All halos")
plt.bar(10**m, imp_lgbm/np.sum(imp_lgbm), width=2/3*width, label="LightGBM", color="pink", alpha=0.8)
plt.legend(loc="best")

###### Plot predictions for test set ######

testing_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/"
                       "ic_traj/nest_2000_lr006/testing_ids.npy")

features_testing = traj[testing_ids, :-1]
truth_testing = np.log10(traj[testing_ids])

pred_sklearn = gbm_bestest.predict(features_testing)
pred_lgbm = lgbm_algo.predict(features_testing)

bins_plotting = np.linspace(truth_testing.min(), truth_testing.max(), 15, endpoint=True)
pf.compare_two_violin_plots(pred_sklearn, truth_testing, pred_lgbm, truth_testing, bins_plotting, path=None,
                                  label1="sklearn", label2="LightGBM")
plt.legend(loc="best")
plt.savefig("/Users/lls/Desktop/violins_sklearn_lgbm.png")
