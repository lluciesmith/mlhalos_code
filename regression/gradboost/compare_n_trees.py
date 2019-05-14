"""
Compare mean absolute error of training and test set, importances and predictions
for two classifiers which differ by the number of trees

"""
import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

saving_path = "/share/data2/lls/regression/gradboost/ic_traj/compare_n_trees/"

# data

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
traj = np.load("/share/data2/lls/features_w_periodicity_fix/ics_density_contrasts.npy")

training_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/training_ids.npy")
traj_training = traj[training_ids, :]
log_halo_training = np.log10(halo_mass[training_ids])

testing_ids = np.load("/share/data1/lls/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")
log_halo_testing = np.log10(halo_mass[testing_ids])
traj_testing = traj[testing_ids, :]

del traj
del halo_mass


# training

# 300 TREES

# clf_300 = GradientBoostingRegressor(loss="lad", n_estimators=300, learning_rate=0.01, max_depth=5, criterion="mse",
#                                     random_state=3, max_features="sqrt")
# clf_300.fit(traj_training, log_halo_training)
# joblib.dump(clf_300, saving_path + "300_trees/clf.pkl")
# np.save(saving_path + "300_trees/imp_300.npy", clf_300.feature_importances_)
# pred_300 = clf_300.predict(traj_testing)
# np.save(saving_path + "300_trees/pred_300.npy", pred_300)
#
# mae_train_300 = np.zeros(len(clf_300.estimators_), )
# for i, y_pred in enumerate(clf_300.staged_predict(traj_training)):
#     mae_train_300[i] = mean_absolute_error(log_halo_training, y_pred)
# np.save(saving_path + "300_trees/mae_training_scores_300.npy", mae_train_300)
#
# mae_test_300 = np.zeros(len(clf_300.estimators_), )
# for i, y_pred in enumerate(clf_300.staged_predict(traj_testing)):
#     mae_test_300[i] = mean_absolute_error(log_halo_testing, y_pred)
# np.save(saving_path + "300_trees/mae_testing_scores_300.npy", mae_test_300)

# 600 trees

# clf_600 = GradientBoostingRegressor(loss="lad", n_estimators=600, learning_rate=0.01, max_depth=5, criterion="mse",
#                                     random_state=3, max_features="sqrt")
# clf_600.fit(traj_training, log_halo_training)
# joblib.dump(clf_600, saving_path + "600_trees/clf.pkl")
# np.save(saving_path + "600_trees/imp_600.npy", clf_600.feature_importances_)
# pred_600 = clf_600.predict(traj_testing)
# np.save(saving_path + "600_trees/pred_600.npy", pred_600)
#
# mae_train_600 = np.zeros(len(clf_600.estimators_), )
# for i, y_pred in enumerate(clf_600.staged_predict(traj_training)):
#     mae_train_600[i] = mean_absolute_error(log_halo_training, y_pred)
# np.save(saving_path + "600_trees/mae_training_scores_600.npy", mae_train_600)
#
# mae_test_600 = np.zeros(len(clf_600.estimators_), )
# for i, y_pred in enumerate(clf_600.staged_predict(traj_testing)):
#     mae_test_600[i] = mean_absolute_error(log_halo_testing, y_pred)
# np.save(saving_path + "600_trees/mae_testing_scores_600.npy", mae_test_600)

# 1000 trees

# clf_1000 = GradientBoostingRegressor(loss="lad", n_estimators=1000, learning_rate=0.01, max_depth=5, criterion="mse",
#                                     random_state=3, max_features="sqrt")
# clf_1000.fit(traj_training, log_halo_training)
# joblib.dump(clf_1000, saving_path + "1000_trees/clf.pkl")
# np.save(saving_path + "1000_trees/imp_1000.npy", clf_1000.feature_importances_)
# pred_1000 = clf_1000.predict(traj_testing)
# np.save(saving_path + "1000_trees/pred_1000.npy", pred_1000)
#
# mae_train_1000 = np.zeros(len(clf_1000.estimators_), )
# for i, y_pred in enumerate(clf_1000.staged_predict(traj_training)):
#     mae_train_1000[i] = mean_absolute_error(log_halo_training, y_pred)
# np.save(saving_path + "1000_trees/mae_training_scores_1000.npy", mae_train_1000)
#
# mae_test_1000 = np.zeros(len(clf_1000.estimators_), )
# for i, y_pred in enumerate(clf_1000.staged_predict(traj_testing)):
#     mae_test_1000[i] = mean_absolute_error(log_halo_testing, y_pred)
# np.save(saving_path + "1000_trees/mae_testing_scores_1000.npy", mae_test_1000)

# 1000 trees, learning rate = 1

clf_1000 = GradientBoostingRegressor(loss="lad", n_estimators=1000, learning_rate=1, max_depth=5, criterion="mse",
                                    random_state=3, max_features="sqrt")
clf_1000.fit(traj_training, log_halo_training)
joblib.dump(clf_1000, saving_path + "1000_trees/clf.pkl")
np.save(saving_path + "1000_trees/imp_1000.npy", clf_1000.feature_importances_)
pred_1000 = clf_1000.predict(traj_testing)
np.save(saving_path + "1000_trees/pred_1000.npy", pred_1000)

mae_train_1000 = np.zeros(len(clf_1000.estimators_), )
for i, y_pred in enumerate(clf_1000.staged_predict(traj_training)):
    mae_train_1000[i] = mean_absolute_error(log_halo_training, y_pred)
np.save(saving_path + "1000_trees/mae_training_scores_1000.npy", mae_train_1000)

mae_test_1000 = np.zeros(len(clf_1000.estimators_), )
for i, y_pred in enumerate(clf_1000.staged_predict(traj_testing)):
    mae_test_1000[i] = mean_absolute_error(log_halo_testing, y_pred)
np.save(saving_path + "1000_trees/mae_testing_scores_1000.npy", mae_test_1000)




# ###### plotting stuff on desktop #####
#
# # path = "/Users/lls/Documents/mlhalos_files/regression/gradboost/ic_traj/compare_n_trees/"
# #
# # # Importances
# #
# # m = 10**np.linspace(np.log10(3e10), np.log10(1e15), 50)
# # width = np.append(np.diff(m), np.diff(m)[-1])
# # imp_600 = np.load(path + "600_trees/imp_600.npy")
# # imp_300 = np.load(path + "300_trees/imp_300.npy")
# imp_1000 = np.load(path + "1000_trees/imp_1000.npy")
#
# plt.figure(figsize=(10, 5))
# plt.bar(m, imp_600, label="num trees=600", color="b",alpha=0.7, width=width*2/3, align="center")
# plt.bar(m, imp_1000, label="num trees=100", color="g",alpha=0.7, width=width*2/3, align="center")
# plt.xscale("log")
# plt.legend(loc="best")
# plt.xlabel("$M_{\mathrm{smoothing}} / \mathrm{M}_{\odot} $")
# plt.ylabel("Importance")
# #
# #
# # # Mae train and test
# #
# # mae_train_300 = np.load(path + "300_trees/mae_training_scores_300.npy")
# # mae_test_300 = np.load(path + "300_trees/mae_testing_scores_300.npy")
# mae_train_600 = np.load(path + "600_trees/mae_training_scores_600.npy")
# mae_test_600 = np.load(path + "600_trees/mae_testing_scores_600.npy")
# mae_train_1000 = np.load(path + "1000_trees/mae_training_scores_1000.npy")
# mae_test_1000 = np.load(path + "1000_trees/mae_testing_scores_1000.npy")
# #
# #
# plt.plot(np.arange(1, 1000), mae_train_1000[1:] - mae_train_1000[:-1], color="g", ls="--", label="training(num "
#                                                                                                  "trees=1000)")
# plt.plot(np.arange(1, 1000), mae_test_1000[1:] - mae_test_1000[:-1], color="g", label="testing(num trees=1000)")
# plt.plot(np.arange(1, 600), mae_train_600[1:] - mae_train_600[:-1], color="b", ls="--", label="training(num trees=600)")
# plt.plot(np.arange(1, 600), mae_test_600[1:] - mae_test_600[:-1], color="b", label="testing(num trees=600)")
# plt.xlabel("Number of trees")
# plt.ylabel("Mean absolute error")
# plt.legend(loc="best")
# #
# # # predictions
# #
# # log_halo_testing = np.log10(np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output"
# #                             "/even_radii_and_random/true_halo_mass.npy"))
# # pred_300 = np.load(path + "300_trees/pred_300.npy")
# # pred_600 = np.load(path + "600_trees/pred_600.npy")
# pred_1000 = np.load(path + "1000_trees/pred_1000.npy")
#
# bins_plotting = np.linspace(log_halo_testing.min(), log_halo_testing.max(), 15, endpoint=True)
for i in range(14):
    ids_RF = (log_halo_testing >= bins_plotting[i]) & (log_halo_testing <= bins_plotting[i + 1])
    mae_300 = mean_absolute_error(log_halo_testing[ids_RF], pred_random_400[ids_RF])
    mae_1000 = mean_absolute_error(log_halo_testing[ids_RF], pred_random[ids_RF])
    # ids_ic = (true_ic >= bins_plotting[i]) & (true_ic <= bins_plotting[i + 1])
    plt.figure()
    plt.hist(pred_random_400[ids_RF], bins=50, label="n trees=400, mae = %.2f" % mae_300, color="g", histtype="step",
             normed=True)
    plt.hist(pred_random[ids_RF], bins=50, label="n trees=1000, mae = %.2f" % mae_1000, color="b", histtype="step",
             normed=True)

    plt.axvline(x=bins_plotting[i], color="k")
    plt.axvline(x=bins_plotting[i + 1], color="k")
    plt.xlim(10,15)
    plt.xlabel("Predicted masses")
    plt.legend(loc="best")


def get_loss_vs_iterations(clf, features, truth, percent=False):
    mae = np.zeros(len(clf.estimators_), )
    for i, y_pred in enumerate(clf.staged_predict(features)):
        if percent is True:
            mae[i] = np.mean(np.abs((truth - y_pred) / truth)) * 100
        else:
            mae[i] = mean_absolute_error(truth, y_pred)
    return mae

mae_halos_training = np.zeros((14, 1000))
mae_halos_testing = np.zeros((14, 1000))

for i in range(14):
    ids_testing = (log_halo_testing >= bins_plotting[i]) & (log_halo_testing <= bins_plotting[i + 1])
    ids_training = (log_halo_training >= bins_plotting[i]) & (log_halo_training <= bins_plotting[i + 1])

    mae_halos_training[i] = get_loss_iterations(clf_random, traj_training[ids_training], log_halo_training[
        ids_training], percent=True)
    mae_halos_testing[i] = get_loss_iterations(clf_random, traj_testing[ids_testing], log_halo_testing[ids_testing],
                                               percent=True)

mid_bins = (bins_plotting[1:] + bins_plotting[:-1])/2

plt.figure()
for i in [400, 600, 999]:
    plt.scatter(mid_bins, mae_halos_training[:, i] - mae_halos_training[:, 200], label="tree " + str(i), s=5)
    plt.scatter(mid_bins, mae_halos_testing[:, i] - mae_halos_training[:, 200], marker="x", label="tree " + str(i), s=5)
plt.legend(loc="best")
plt.xlabel("log mass training")
plt.ylabel("MAE")

plt.figure()
for i in [400, 600, 999]:
    plt.scatter(mid_bins, mae_halos_testing[:, i]- mae_halos_training[:, 200], fmt="x", label="tree " + str(i), s=2)
plt.legend(loc="best")
plt.xlabel("log mass testing")
plt.ylabel("MAE")

for i in range(14):
    plt.figure()
    plt.plot(np.arange(1000), mae_halos_training[i], label="training")
    plt.plot(np.arange(1000), mae_halos_testing[i], label="testing")
    plt.legend(loc="best")
    plt.xlabel("Bin " + str(i))