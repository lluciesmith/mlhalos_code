"""
Binary classification

"""
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from mlhalos import machinelearning as ml
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from mlhalos import plot


def get_multiple_rocs(predicted, true, labels=[""], add_EPS=False, fpr_EPS=None, tpr_EPS=None, add_ellipsoidal=False,
                      fpr_ellipsoidal=None, tpr_ellipsoidal=None, label_EPS=None, label_ellipsoidal=None,
                      fontsize_labels=17):
    FPR = np.zeros((2, 50))
    TPR = np.zeros((2, 50))
    AUC = np.zeros((2, ))
    for i in range(2):
        FPR[i], TPR[i], AUC[i], threshold = ml.roc(predicted[i], true[i], true_class=1)

    fig = plot.roc_plot(FPR.transpose(), TPR.transpose(), AUC, labels=labels, figsize=None, add_EPS=add_EPS,
                        fpr_EPS=fpr_EPS, tpr_EPS=tpr_EPS, add_ellipsoidal=add_ellipsoidal,
                        fpr_ellipsoidal=fpr_ellipsoidal, tpr_ellipsoidal=tpr_ellipsoidal, label_EPS=label_EPS,
                        label_ellipsoidal=label_ellipsoidal, frameon=False,
                        fontsize_labels=fontsize_labels, cols=["#8856a7", "#7ea6ce" ])

    return FPR, TPR, AUC, fig


den_f = np.load("/Users/lls/Documents/CODE/stored_files/shear/shear_quantities/features/density_features.npy")
training_ids = np.load("/Users/lls/Documents/mlhalos_files/stored_files/density/50k_features_index.npy")
testing_index = np.random.choice(np.arange(256**3)[~np.in1d(np.arange(256 ** 3), training_ids)], 50000)
# testing_index = ~np.in1d(np.arange(256 ** 3), training_ids)
true_test = den_f[testing_index, -1]
true_train = den_f[training_ids, -1]
training_features = den_f[training_ids, :-1]
testing_features = den_f[testing_index, :-1]

path = "/Users/lls/Desktop/GBT_binary_class/lr_01_maxf_08_subs_06/"
clf = GradientBoostingClassifier(n_estimators=1, max_depth=10, learning_rate=0.1, max_features=0.8, warm_start=True,
                                 subsample=0.6)
clf.fit(training_features, true_train)
imp_0 = clf.feature_importances_
pred_0 = clf.predict_proba(testing_features)
fpr_0, tpr_0, auc_0, threshold_0 = ml.roc(pred_0, true_test, true_class=1)
loss_0 = clf.loss_(true_test, pred_0[:,1])


l1_norm = np.zeros(100,)
loss = np.zeros(100,)
auc = np.zeros(100,)

m = np.linspace(np.log10(3e10), np.log10(1e15), 50)
width = np.append(np.diff(m), np.diff(m)[-1])

plt.figure()
for i in range(100):
    clf.n_estimators += 1
    print(clf.n_estimators)
    clf.fit(training_features, true_train)
    print("Done fit")

    imp_i = clf.feature_importances_
    print("Done importances")
    print("Done predictions")
    loss_i = clf.loss_(true_test, clf.decision_function(testing_features))
    print("Done loss")
    pred_i = clf.predict_proba(testing_features)
    fpr_i, tpr_i, auc_i, threshold_i = ml.roc(pred_i, true_test, true_class=1)
    print("Auc is " + str(auc_i))
    print("Done auc")
    if i in [20, 40, 80, 99]:
        plt.bar(m, clf.feature_importances_, width=width * 2 / 3, label="Tree " + str(i+1), alpha=0.6)

    l1_norm[i] = np.sum(abs(imp_i - imp_0))
    loss[i] = abs(loss_i - loss_0)
    auc[i] = abs(auc_i - auc_0)

    imp_0 = imp_i
    loss_0 = loss_i
    auc_0 = auc_i

plt.xlabel("log smoothing mass")
plt.ylabel("Importance")
plt.legend()
plt.subplots_adjust(bottom=0.15)
plt.savefig(path + "imps.png")
joblib.dump(clf, path + "clf.pkl")

# compare ROC to RF

pred_density_RF = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/density_only/"
                       "predicted_den.npy")
true_density_RF = np.load("/Users/lls/Documents/CODE/stored_files/shear/classification/density_only/true_den.npy")

pred_all = np.array([pred_i, pred_density_RF])
true_all = np.array([true_test, true_density_RF])
fpr, tpr, auc, fig = get_multiple_rocs(pred_all, true_all, labels=["GBT", "RF"])
plt.savefig(path + "roc_vs_RF.png")


# score test vs train

score_test = np.zeros(clf.n_estimators,)
for i, y_pred in enumerate(clf.staged_decision_function(testing_features)):
    score_test[i] = clf.loss_(true_test, y_pred)

score_train = np.zeros(clf.n_estimators, )
for i, y_pred in enumerate(clf.staged_decision_function(training_features)):
    score_train[i] = clf.loss_(true_train, y_pred)

score_train -= score_train[0]
score_test -= score_test[0]

plt.figure()
plt.plot(np.arange(clf.n_estimators), score_train, label="score train")
plt.plot(np.arange(clf.n_estimators), score_test, label="score test")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.xlabel("N estimators")
plt.savefig(path + "scores.png")

# test the convergence of importances

plt.figure()
p3 = np.poly1d(np.polyfit(np.log10(loss)[1:], np.log10(l1_norm)[1:], 3))
p2 = np.poly1d(np.polyfit(np.log10(loss)[1:], np.log10(l1_norm)[1:], 2))
x = np.linspace(np.log10(loss[1:]).min(), np.log10(loss[1:]).max(), 100)
_ = plt.scatter(np.log10(loss)[1:], np.log10(l1_norm)[1:], s=2, color="k")
_ = plt.plot(x, p2(x), "-", color="k")
# _ = plt.plot(x, p3(x), "--", color="k")
plt.xlabel("$\log(\Delta L)$")
plt.ylabel("$\log$(l1 norm)")

l1_norm_2 = np.zeros(100,)
loss_2 = np.zeros(100,)

for i in range(100):
    clf.n_estimators += 1
    print(clf.n_estimators)
    clf.fit(training_features, true_train)
    print("Done fit")

    imp_i = clf.feature_importances_
    print("Done importances")
    loss_i = clf.loss_(true_test, clf.decision_function(testing_features))
    print("Done loss")

    l1_norm_2[i] = np.sum(abs(imp_i - imp_0))
    loss_2[i] = loss_i - loss_0

    imp_0 = imp_i
    loss_0 = loss_i

_ = plt.scatter(np.log10(abs(loss_2)), np.log10(l1_norm_2), s=2)
x = np.linspace(np.log10(abs(loss_2)).min(), np.log10(loss[1:]).max(), 100)
p2 = np.poly1d(np.polyfit(np.log10(abs(loss_2)), np.log10(l1_norm_2), 4))
_ = plt.plot(x, p2(x), "-")
_ = plt.plot(x, p3(x), "--", color="k")
plt.savefig(path + "fitted_loss_l1_norm.png")

l1_norm_3 = np.zeros(100,)
loss_3 = np.zeros(100,)

for i in range(100):
    clf.n_estimators += 1
    print(clf.n_estimators)
    clf.fit(training_features, true_train)
    print("Done fit")

    imp_i = clf.feature_importances_
    print("Done importances")
    loss_i = clf.loss_(true_test, clf.decision_function(testing_features))
    print("Done loss")

    l1_norm_3[i] = np.sum(abs(imp_i - imp_0))
    loss_3[i] = loss_i - loss_0

    imp_0 = imp_i
    loss_0 = loss_i


LOSS = np.concatenate((loss[1:], abs(loss_2), abs(loss_3)))
L1_NORM = np.concatenate((l1_norm[1:], l1_norm_2, l1_norm_3))

p2 = np.poly1d(np.polyfit(np.log10(LOSS), np.log10(L1_NORM), 2))

plt.figure()
plt.scatter(np.log10(loss)[1:], np.log10(l1_norm)[1:], s=2, color="k", label="0--100 trees")
plt.scatter(np.log10(abs(loss_2)), np.log10(l1_norm_2), s=2, color="r", label="100 -- 200 trees")
plt.scatter(np.log10(abs(loss_3)), np.log10(l1_norm_3), s=2, color="b", label="200 -- 300 trees")
x = np.linspace(np.log10(abs(LOSS)).min(), np.log10(LOSS).max(), 100)
plt.plot(x, p2(x), "-", color="k")

