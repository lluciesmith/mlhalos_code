import sys
sys.path.append("/home/lls/mlhalos_code/scripts")
import numpy as np
from sklearn.externals import joblib


classifier = joblib.load("/share/data1/lls/shear_quantities/classifier_den+den_sub_ell+den_sub_prol/clf.pkl")

try:
    imp = classifier.feature_importances_
except:
    imp = classifier.best_estimator_.feature_importances_

np.save("/share/data1/lls/shear_quantities/classifier_den+den_sub_ell+den_sub_prol/importances.npy", imp)
