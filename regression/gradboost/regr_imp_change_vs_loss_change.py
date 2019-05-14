"""


"""
import sys
sys.path.append("/home/lls/mlhalos_code")
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from mlhalos import machinelearning as ml
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from mlhalos import plot
from sklearn.externals import joblib


path = "/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/imp_change/"

training_ids = np.load("/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006"
                       "/training_ids.npy")
testing_ids = np.load("/share/data2/lls/regression/gradboost/randomly_sampled_training/ic_traj/nest_2000_lr006"
                       "/testing_ids.npy")

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
traj = np.load("/share/data2/lls/features_w_periodicity_fix/ics_density_contrasts.npy")
#
# traj = np.load("/Users/lls/Documents/mlhalos_files/regression/features_w_periodicity_fix/ics_density_contrasts.npy")
# halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")
#
# training_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/ic_traj"
#                        "/nest_2000_lr006/training_ids.npy")
#
# testing_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/ic_traj/nest_2000_lr006/testing_ids.npy")
t = np.random.choice(testing_ids, 100000, replace=False)
testing_features = traj[t]
true_test = np.log10(halo_mass[t])

training_features = traj[training_ids]
true_train = np.log10(halo_mass[training_ids])

# clf = GradientBoostingRegressor(n_estimators=1, max_depth=10, max_features=0.8, warm_start=True, subsample=0.8,
#                                 learning_rate=0.01)
#
# clf.fit(training_features, true_train)
# imp_0 = clf.feature_importances_
# pred_0 = clf.predict(testing_features)
# loss_0 = mean_absolute_error(true_test, pred_0)
#
#
# l1_norm = np.zeros(100,)
# loss = np.zeros(100,)
#
# m = np.linspace(np.log10(3e10), np.log10(1e15), 50)
# width = np.append(np.diff(m), np.diff(m)[-1])
# imp_first_trees = []
#
# # plt.figure()
# for i in range(100):
#     clf.n_estimators += 1
#     print(clf.n_estimators)
#     clf.fit(training_features, true_train)
#     print("Done fit")
#
#     imp_i = clf.feature_importances_
#     print("Done importances")
#     pred_i = clf.predict(testing_features)
#     loss_i = mean_absolute_error(true_test, pred_i)
#     print("Done loss")
#     if i in [20, 40, 80, 99]:
#         imp_first_trees.append(clf.feature_importances_)
#
#     l1_norm[i] = np.sum(abs(imp_i - imp_0))
#     loss[i] = abs(loss_i - loss_0)
#
#     imp_0 = imp_i
#     loss_0 = loss_i
#
# np.save(path + "l1_norm_100.npy", l1_norm)
# np.save(path + "loss_100.npy", loss)
# joblib.dump(clf, path + "clf_100.pkl")
# np.save(path + "imp_first_trees.npy", np.array(imp_first_trees))

clf = joblib.load(path + "clf_100.pkl")
imp_0 = clf.feature_importances_
loss_0 = np.load(path + "loss_100.npy")[-1]


# test the convergence of importances

l1_norm_2 = np.zeros(10,)
loss_2 = np.zeros(10,)

for i in range(10):
    clf.n_estimators += 20
    print(clf.n_estimators)
    clf.fit(training_features, true_train)
    print("Done fit")

    imp_i = clf.feature_importances_
    print("Done importances")
    pred_i = clf.predict(testing_features)
    loss_i = mean_absolute_error(true_test, pred_i)
    print("Done loss")

    l1_norm_2[i] = np.sum(abs(imp_i - imp_0))
    loss_2[i] = abs(loss_i - loss_0)

    imp_0 = imp_i
    loss_0 = loss_i

imp_300 = clf.feature_importances_

np.save(path + "l1_norm_300.npy", l1_norm_2)
np.save(path + "loss_300.npy", loss_2)
joblib.dump(clf, path + "clf_300.pkl")
np.save(path + "imp_300.npy", imp_300)



l1_norm_3 = np.zeros(15,)
loss_3 = np.zeros(15,)

for i in range(15):
    clf.n_estimators += 20
    print(clf.n_estimators)
    clf.fit(training_features, true_train)
    print("Done fit")

    imp_i = clf.feature_importances_
    print("Done importances")
    pred_i = clf.predict(testing_features)
    loss_i = mean_absolute_error(true_test, pred_i)
    print("Done loss")

    l1_norm_3[i] = np.sum(abs(imp_i - imp_0))
    loss_3[i] = abs(loss_i - loss_0)

    imp_0 = imp_i
    loss_0 = loss_i

imp_600 = clf.feature_importances_


np.save(path + "l1_norm_600.npy", l1_norm_3)
np.save(path + "loss_600.npy", loss_3)
joblib.dump(clf, path + "clf_600.pkl")
np.save(path + "imp_600.npy", imp_600)

l1_norm = np.load(path + "l1_norm_100.npy")
loss = np.load(path + "loss_100.npy")
p2 = np.poly1d(np.polyfit(np.log10(loss[3:]), np.log10(l1_norm[3:]), 1))

plt.figure()
plt.scatter(np.log10(loss), np.log10(l1_norm), s=2, color="k", label="0--100 trees")
plt.scatter(np.log10(abs(loss_2)), np.log10(l1_norm_2), s=2, color="r", label="100 -- 300 trees")
plt.scatter(np.log10(abs(loss_3)), np.log10(l1_norm_3), s=2, color="b", label="300 -- 500 trees")
# plt.scatter(np.log10(abs(loss_2[200:])), np.log10(l1_norm_2[200:]), s=2, color="g", label="300 -- 600 trees")
x = np.linspace(np.log10(abs(loss_3)).min(), np.log10(loss).max(), 100)
plt.legend(loc="best")
plt.plot(x, p2(x), "--", color="k")
plt.xlabel("$\log(\Delta L)$")
plt.ylabel("$\log$(l1 norm)")
plt.savefig(path + "fitted_loss_l1_norm.png")