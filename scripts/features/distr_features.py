import numpy as np
import matplotlib.pyplot as plt


features_full_mass = np.load('/Users/lls/Documents/CODE/stored_files/all_out/features_full_mass_scale.npy')

features_training = np.load("/Users/lls/Documents/CODE/stored_files/all_out/50k_features.npy")


# check distribution of each features for all particles and a subset of 50000 random particles

for i in range(50):
    n0, b0 = np.histogram(features_full_mass[:,i], bins=20)
    n01, b01 = np.histogram(features_training[:,i], bins=b0)
    plt.plot(b0[:-1], n01 / n0)
    plt.title("Feature " + str(i))
    plt.xlabel("Feature value")
    plt.ylabel("train/all ratio")
    plt.savefig("/Users/lls/Documents/CODE/stored_files/all_out/feature_distributions/train_all_50k_f_full_mass"
                "/feature_" + str(i) + ".png")
    plt.clf()


# Plot only in range [-3,3] (3sigma)

for i in range(50):
    n0, b0 = np.histogram(features_full_mass[:,i], bins=20)
    n01, b01 = np.histogram(features_training[:,i], bins=b0)
    plt.plot(b0[:-1], n01 / n0)
    plt.title("Feature " + str(i))
    plt.xlabel("Feature value")
    plt.ylabel("train/all ratio")
    plt.xlim(-3,3)
    plt.savefig("/Users/lls/Documents/CODE/stored_files/all_out/feature_distributions/train_all_50k_f_full_mass"
                "/3sigma/feature_" + str(i) + ".png")
    plt.clf()


# Try to get errorbars giving the sample size --> standard error is sample std. deviation / number of samples ?

for i in range(50):
    n0, b0 = np.histogram(features_full_mass[:,i], bins=20)
    n01, b01 = np.histogram(features_training[:,i], bins=b0)
    plt.errorbar(b0[:-1], n01/n0, yerr=1/n01)
    plt.title("Feature " + str(i))
    plt.xlabel("Feature value")
    plt.ylabel("train/all ratio")
    plt.savefig("/Users/lls/Documents/CODE/stored_files/all_out/feature_distributions/train_all_50k_f_full_mass"
                "/w_errorbars/feature_" + str(i) + ".png")
    plt.clf()

# In hypatia:

# r1 = []
# for j in range(10):
#
#     a = []
#     for i in range(50):
#         n0, b0 = np.histogram(features_full_mass[:, i], bins=20)
#         n01, b01 = np.histogram(features_training[:, i], bins=b0)
#         r = np.zeros((20,))
#         r[n0!=0] = n01[n0!=0]/n0[n0!=0]
#         r[n0==0] = 0
#         a.append(r)
#
#     r1.append(np.array(a))
#
# r1= np.array(r1)
#
# std_features = []
# for k in range(r1.shape[1]):
#     std = np.std(r1[:, k, :], axis=0)
#     std_features.append(std)
#
# std_features = np.array(std_features)
