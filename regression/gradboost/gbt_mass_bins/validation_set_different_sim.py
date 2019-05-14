import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from mlhalos import parameters


################## FIRST BUILD THE TRAINING SET FROM ORIGINAL SIMULATION ##################

traj = np.load("/Users/lls/Documents/mlhalos_files/regression/features_w_periodicity_fix/ics_density_contrasts.npy")
halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")

# all_ids = np.where((np.log10(halo_mass) > 12.5) & (np.log10(halo_mass) <= 13.5))[0]
# training_ids = np.random.choice(all_ids, 100000, replace=False)

all_ids = np.arange(256**3)[halo_mass > 0]
training_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/gradboost/random_sampled_training/"
                       "ic_traj/nest_2000_lr006/training_ids.npy")

features_training = traj[training_ids, :-1]
truth_training = np.log10(halo_mass[training_ids])


################## CREATE A VALIDATION SET FROM THE SAME SIMULATION ##################


remaining_ids = all_ids[~np.in1d(all_ids, training_ids)]
validation_ids_same_sim = np.random.choice(remaining_ids, 100000, replace=False)

features_val_same_sim = traj[validation_ids_same_sim, :-1]
truth_val_same_sim = np.log10(halo_mass[validation_ids_same_sim])



################## LOAD SIMULATION TO USE FOR VALIDATION SET ##################

# ic_validation = parameters.InitialConditionsParameters(
#     initial_snapshot="/Users/lls/Documents/mlhalos_files/reseed50/IC.gadget3",
#     final_snapshot="/Users/lls/Documents/mlhalos_files/reseed50/snapshot_099", load_final=True)

traj_val = np.load("/Users/lls/Documents/mlhalos_files/reseed50/features/density_constrasts.npy")
truth_val = np.load("/Users/lls/Documents/mlhalos_files/reseed50/features/halo_mass_particles.npy")

all_ids_diff_sim = np.arange(256**3)[truth_val > 0]
val_ids_diff_sim = np.random.choice(all_ids_diff_sim, 10000, replace=False)

features_val_diff_sim = traj_val[val_ids_diff_sim, :-1]
truth_val_diff_sim = np.log10(truth_val[val_ids_diff_sim])


################## HYPERPARAMETER: MAX FEATURES ##################

def test_scores_for_max_features(hyperparameter_values):
    trainining_score = []
    val_same_sim = []
    val_diff_sim = []

    for hyperparameter_value in hyperparameter_values:
       model = GradientBoostingRegressor(max_features=hyperparameter_value, n_estimators=150, learning_rate=0.05,
                                         max_depth=3)
       model.fit(features_training, truth_training)

       train_pred = model.predict(features_training)
       mae_train = mae(truth_training, train_pred)
       trainining_score.append(mae_train)

       val_pred_same_sim = model.predict(features_val_same_sim)
       mae_same_sim = mae(truth_val_same_sim, val_pred_same_sim)
       val_same_sim.append(mae_same_sim)

       validation_pred = model.predict(features_val_diff_sim)
       mae_diff_sim = mae(truth_val_diff_sim, validation_pred)
       val_diff_sim.append(mae_diff_sim)

    return trainining_score, val_same_sim, val_diff_sim


max_features = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
train_results, test_same_sim, test_diff_sim = test_scores_for_max_features(max_features)

plt.plot(max_features, train_results, label="train")
plt.plot(max_features, test_same_sim, label="validation same sim")
plt.plot(max_features, test_diff_sim, label="validation diff sim")
plt.xlabel("Number of features at node/total number of features")
plt.ylabel("Mean absolute error")
plt.legend(loc="best")
plt.subplots_adjust(bottom=0.15)
plt.savefig("/Users/lls/Desktop/mae_vs_max_features_full_mass_range.png")
plt.clf()


################## HYPERPARAMETER: LEARNING RATE ##################

def test_scores_for_learning_rate(hyperparameter_values):
    trainining_score = []
    val_same_sim = []
    val_diff_sim = []

    for hyperparameter_value in hyperparameter_values:
       model = GradientBoostingRegressor(learning_rate=hyperparameter_value)
       model.fit(features_training, truth_training)

       train_pred = model.predict(features_training)
       mae_train = mae(truth_training, train_pred)
       trainining_score.append(mae_train)

       val_pred_same_sim = model.predict(features_val_same_sim)
       mae_same_sim = mae(truth_val_same_sim, val_pred_same_sim)
       val_same_sim.append(mae_same_sim)

       validation_pred = model.predict(features_val_diff_sim)
       mae_diff_sim = mae(truth_val_diff_sim, validation_pred)
       val_diff_sim.append(mae_diff_sim)

    return trainining_score, val_same_sim, val_diff_sim

learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
train_results_lr, test_same_sim_lr, test_diff_sim_lr = test_scores_for_learning_rate(learning_rates)

plt.plot(learning_rates, train_results_lr, label="train")
plt.plot(learning_rates, test_same_sim_lr, label="validation same sim")
plt.plot(learning_rates, test_diff_sim_lr, label="validation diff sim")
plt.xlabel("Learning rate")
plt.ylabel("Mean absolute error")
plt.legend(loc="best")
plt.subplots_adjust(bottom=0.15)
plt.savefig("/Users/lls/Desktop/mae_vs_learning_rate_full_mass_range.png")
plt.clf()


################## HYPERPARAMETER: NUMBER OF ESTIMATORS ##################

def test_scores_for_num_estimators(hyperparameter_values):
    trainining_score = []
    val_same_sim = []
    val_diff_sim = []

    for hyperparameter_value in hyperparameter_values:
       model = GradientBoostingRegressor(n_estimators=hyperparameter_value)
       model.fit(features_training, truth_training)

       train_pred = model.predict(features_training)
       mae_train = mae(truth_training, train_pred)
       trainining_score.append(mae_train)

       val_pred_same_sim = model.predict(features_val_same_sim)
       mae_same_sim = mae(truth_val_same_sim, val_pred_same_sim)
       val_same_sim.append(mae_same_sim)

       validation_pred = model.predict(features_val_diff_sim)
       mae_diff_sim = mae(truth_val_diff_sim, validation_pred)
       val_diff_sim.append(mae_diff_sim)

    return trainining_score, val_same_sim, val_diff_sim

n_estimators = [1, 10, 20, 50, 100, 200, 400]
train_results_est, test_same_sim_est, test_diff_sim_est = test_scores_for_num_estimators(n_estimators)

plt.plot(n_estimators, train_results_est, label="train")
plt.plot(n_estimators, test_same_sim_est, label="validation same sim")
plt.plot(n_estimators, test_diff_sim_est, label="validation diff sim")
plt.xlabel("Number of trees")
plt.ylabel("Mean absolute error")
plt.legend(loc="best")
plt.savefig("/Users/lls/Desktop/mae_vs_number_of_trees_full_mass_range.png")
plt.clf()


################## HYPERPARAMETER: MAX DEPTH ##################

def test_scores_for_max_depth(hyperparameter_values):
    trainining_score = []
    val_same_sim = []
    val_diff_sim = []

    for hyperparameter_value in hyperparameter_values:
       model = GradientBoostingRegressor(max_depth=hyperparameter_value)
       model.fit(features_training, truth_training)

       train_pred = model.predict(features_training)
       mae_train = mae(truth_training, train_pred)
       trainining_score.append(mae_train)

       val_pred_same_sim = model.predict(features_val_same_sim)
       mae_same_sim = mae(truth_val_same_sim, val_pred_same_sim)
       val_same_sim.append(mae_same_sim)

       validation_pred = model.predict(features_val_diff_sim)
       mae_diff_sim = mae(truth_val_diff_sim, validation_pred)
       val_diff_sim.append(mae_diff_sim)

    return trainining_score, val_same_sim, val_diff_sim

max_depths = np.linspace(1, 10, 5, endpoint=True)
train_results_depth, test_same_sim_depth, test_diff_sim_depth = test_scores_for_max_depth(max_depths)

plt.plot(max_depths, train_results_depth, label="train")
plt.plot(max_depths, test_same_sim_depth, label="validation same sim")
plt.plot(max_depths, test_diff_sim_depth, label="validation diff sim")
plt.xlabel("Max depth")
plt.ylabel("Mean absolute error")
plt.legend(loc="best")
plt.savefig("/Users/lls/Desktop/mae_vs_max_depth_full_mass_range.png")
plt.clf()


################## HYPERPARAMETER: MIN SAMPLES LEAF ##################

def test_scores_for_min_samples_leaf(hyperparameter_values):
    trainining_score = []
    val_same_sim = []
    val_diff_sim = []

    for hyperparameter_value in hyperparameter_values:
       model = GradientBoostingRegressor(min_samples_leaf=hyperparameter_value)
       model.fit(features_training, truth_training)

       train_pred = model.predict(features_training)
       mae_train = mae(truth_training, train_pred)
       trainining_score.append(mae_train)

       val_pred_same_sim = model.predict(features_val_same_sim)
       mae_same_sim = mae(truth_val_same_sim, val_pred_same_sim)
       val_same_sim.append(mae_same_sim)

       validation_pred = model.predict(features_val_diff_sim)
       mae_diff_sim = mae(truth_val_diff_sim, validation_pred)
       val_diff_sim.append(mae_diff_sim)

    return trainining_score, val_same_sim, val_diff_sim

min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results_leafs, test_same_sim_leafs, test_diff_sim_leafs = test_scores_for_min_samples_leaf(min_samples_leafs)

plt.plot(min_samples_leafs, train_results_leafs, label="train")
plt.plot(min_samples_leafs, test_same_sim_leafs, label="validation same sim")
plt.plot(min_samples_leafs, test_diff_sim_leafs, label="validation diff sim")
plt.xlabel("Min samples leaf")
plt.ylabel("Mean absolute error")
plt.legend(loc="best")
plt.savefig("/Users/lls/Desktop/mae_vs_min_samples_leaf_full_mass_range.png")
plt.clf()

