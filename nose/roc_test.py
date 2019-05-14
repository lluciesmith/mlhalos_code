from mlhalos import machinelearning
import numpy as np


def test_roc():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_score = np.array([0., 0.2, 0.5, 0.6, 0.7, 1.0])
    tpr, fpr, auc = machinelearning.roc(y_score, y_true, true_class=1)
    tpr_true = np.array([ 0.,  0.,  0.,  0.,  0., 0. ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
            0.  ,  0.  ,  0.25,  0.25,  0.25,  0.25,  0.25,  0.5 ,  0.5 ,
            0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ,
            0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.75,  0.75,  0.75,  0.75,  0.75,
            0.75,  0.75,  0.75,  0.75,  0.75])
    fpr_true = np.array([ 0. ,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
            0.5,  0.5,  0.5,  0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,
            1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,
            1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,
            1. ,  1. ,  1. ,  1. ,  1. ,  1. ])
    auc_true = 0.75
    assert np.allclose(tpr, tpr_true)
    assert np.allclose(fpr, fpr_true)
    assert auc == auc_true

