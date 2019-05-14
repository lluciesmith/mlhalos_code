import numpy as np
from regression.plots import plotting_functions as pf
from mlhalos import parameters


if __name__ == "__main__":
    # path_shear = "/Users/lls/Documents/mlhalos_files/regression/shear/"
    path_shear = "/Users/lls/Documents/mlhalos_files/shear_period_fix/"

    all_log_true_mass = np.load(path_shear + "true_halo_mass.npy")
    bins_plotting = np.linspace(all_log_true_mass.min(), all_log_true_mass.max(), 15, endpoint=True)

    shear_predicted_mass = np.load(path_shear + "predicted_halo_mass.npy")
    all_log_predicted_mass = np.log10(shear_predicted_mass)

    pf.get_violin_plot_single_prediction(bins_plotting, all_log_predicted_mass, all_log_true_mass,
                                         return_mean=False, label_distr="All")
    # plt.savefig(path + "violins.png")


    ####################### Importances #######################

    f_imp_shear = np.load(path_shear + "f_imp.npy")
    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/mlhalos_files", load_final = True)
    pf.importances_plot(f_imp_shear, initial_parameters=ic, label=["Density", "Ellipticity", "Prolateness"], save=False,
                        subplots=3, figsize=(6.9, 5.9))


    ####################### shear+density vs density only #######################

    # path_density = "/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/even_radii_and_random/"
    path_density = "/Users/lls/Documents/mlhalos_files/den_only_periodicity_fix/"

    log_den_true = np.load("/Users/lls/Documents/mlhalos_files/lowz_density/true_mass_test_set.npy")
    log_den_predicted = np.load(path_density + "predicted_log_halo_mass.npy")
    # log_den_true = np.log10(den_true)
    # log_den_predicted = np.log10(den_predicted)

    # 2D HISTOGRAM

    pf.compare_2d_histograms(all_log_true_mass, all_log_predicted_mass, log_den_true, log_den_predicted,
                             title1="density+shear", title2="density", save_path=None)

    # VIOLINS

    pf.compare_violin_plots(all_log_predicted_mass, all_log_true_mass, log_den_predicted, log_den_true,
                            bins_plotting, label1 ="den+shear", label2 = "density", color1="b", color2="r",
                            path=None, saving_name="violins_shear_vs_den.pdf")

    # Inner radii particles only

    # testing_ids = np.load("/Users/lls/Documents/mlhalos_files/regression/in_halos_only/log_m_output/even_radii_and_random/testing_ids.npy")
    #
    # radii_properties_in = np.load("/Users/lls/Documents/mlhalos_files/stored_files/radii_stuff/radii_properties_in_ids.npy")
    # radii_properties_out = np.load("/Users/lls/Documents/mlhalos_files/stored_files/radii_stuff/radii_properties_out_ids.npy")
    # fraction = np.concatenate((radii_properties_in[:,2],radii_properties_out[:,2]))
    # ids_in_halo = np.concatenate((radii_properties_in[:,0],radii_properties_out[:,0]))
    # inner_ids = ids_in_halo[fraction < 0.3]
    # inner_ids = inner_ids.astype("int")
    #
    # ids_inner_tested = np.in1d(testing_ids, inner_ids)
    # predicted_all_inner = all_log_predicted_mass[ids_inner_tested]
    # true_all_inner = all_log_true_mass[ids_inner_tested]
    #
    # violin_plots_density_vs_shear(all_log_predicted_mass[ids_inner_tested], all_log_true_mass[ids_inner_tested],
    #                               log_den_predicted[ids_inner_tested], log_den_true[ids_inner_tested], bins_plotting

