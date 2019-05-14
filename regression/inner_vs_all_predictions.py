import matplotlib.pyplot as plt
import numpy as np

from mlhalos import parameters
from regression.plots import plotting_functions as pf

# INNER PARTICLES

# bins_plotting = [10, 11, 12, 13, 14, 15]
true_mass_test_inner = np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/inner_rad"
                         "/true_halo_mass.npy")
predicted_mass_inner = np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/inner_rad"
                         "/predicted_halo_mass.npy")
log_true_mass_inner = np.log10(true_mass_test_inner)
log_predicted_mass_inner = np.log10(predicted_mass_inner)

bins_plotting = np.linspace(log_true_mass_inner.min(), log_true_mass_inner.max(), 10, endpoint=True)
pf.get_violin_plot_single_prediction(bins_plotting, log_predicted_mass_inner, log_true_mass_inner, return_mean=False,
                                     label_distr="inner")
plt.clf()


# ALL PARTICLES

all_true_mass_test = np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output"
                             "/balanced_training_set/true_halo_mass.npy")
all_predicted_mass = np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output"
                         "/balanced_training_set/predicted_halo_mass.npy")
all_log_true_mass = np.log10(all_true_mass_test)
all_log_predicted_mass = np.log10(all_predicted_mass)

bins_plotting = np.linspace(all_log_true_mass.min(), all_log_true_mass.max(), 10, endpoint=True)
pf.get_violin_plot_single_prediction(bins_plotting, all_log_predicted_mass, all_log_true_mass, return_mean=False,
                                     label_distr="All")
plt.savefig("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/balanced_training_set/"
            "violins.png")
plt.clf()

# look at only inner radii particles

testing_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/balanced_training_set"
                      "/testing_ids.npy")

radii_properties_in = np.load("/Users/lls/Documents/mlhalos_files/stored_files/radii_stuff/radii_properties_in_ids.npy")
radii_properties_out = np.load("/Users/lls/Documents/mlhalos_files/stored_files/radii_stuff/radii_properties_out_ids.npy")
fraction = np.concatenate((radii_properties_in[:,2],radii_properties_out[:,2]))
ids_in_halo = np.concatenate((radii_properties_in[:,0],radii_properties_out[:,0]))
inner_ids = ids_in_halo[fraction < 0.3]
inner_ids = inner_ids.astype("int")

ids_inner_tested = np.in1d(testing_ids, inner_ids)
predicted_all_inner = all_log_predicted_mass[ids_inner_tested]
true_all_inner = all_log_true_mass[ids_inner_tested]

pf.get_violin_plot_single_prediction(bins_plotting, predicted_all_inner, true_all_inner, return_mean=False,
                                     label_distr="inner from ALL training")
plt.clf()


# Importances

ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE", load_final=True)
f_imp_inner = np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/balanced_training_set/f_imp.npy")

pf.importances_plot(f_imp_inner, initial_parameters=ic)
plt.savefig("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/balanced_training_set/imp_plot.png")
