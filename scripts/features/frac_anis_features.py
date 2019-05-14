import sys
sys.path.append('/home/lls/mlhalos_code')
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mlhalos import machinelearning as ml
import matplotlib as plt

fa = np.load("/home/lls/stored_files/shear/fractional_anisotropies.npy")
den_ell = np.load('/home/lls/stored_files/shear_no_rescaling/density_ellipticity_features.npy')
labels = den_ell[:, -1]
densities = den_ell[:, :50]

feat_den_fa = np.column_stack((densities, fa, labels))

index_training = np.load("/home/lls/stored_files/50k_features_index.npy")
training_features = feat_den_fa[index_training, :]

RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=20, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=1300, n_jobs=60,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

RF.fit(training_features[:, :-1], training_features[:, -1])

index_test = np.random.choice(len(feat_den_fa), 5000)
index_test = index_test[~np.in1d(index_test, index_training)]
test_features = feat_den_fa[index_test, :]

pred = RF.predict_proba(test_features[:, :-1])

np.save('/home/lls/stored_files/shear/predicted_test_sample.npy', pred)
np.save('/home/lls/stored_files/shear/true_test_sample.npy', test_features[:, -1])
#
# fig = ml.get_roc_curve(pred, test_features[:, -1])
# plt.savefig("/home/lls/stored_files/shear/roc_den_frac_anis.pdf")


