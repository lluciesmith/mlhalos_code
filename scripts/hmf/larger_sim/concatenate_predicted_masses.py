import numpy as np

# pred_masses_ps = [np.load("/share/data1/lls/sim200/PS/PS_predicted_masses_" + str(i) + ".npy") for i in range(1000)]
# all_pred_masses_ps = np.concatenate(pred_masses_ps)
# np.save("/share/data1/lls/sim200/ALL_PS_predicted_masses.npy", all_pred_masses_ps)
#
# pred_masses_st = [np.load("/share/data1/lls/sim200/ST/ST_predicted_masses_" + str(i) + ".npy") for i in range(1000)]
# all_pred_masses_st = np.concatenate(pred_masses_st)
# np.save("/share/data1/lls/sim200/ALL_ST_predicted_masses.npy", all_pred_masses_st)


# # Concatenate volume sharp-k results
#
# a = np.array(np.array_split(np.arange(512**3), 1000, axis=0))
#
# pred_masses_ps = [np.lib.format.open_memmap("/share/data1/lls/sim200/volume_sharp_k/"
#                                             "PS/PS_predicted_masses_" + str(i) + ".npy", mode="r",
#                                             shape=(len(a[i]), 910))
#                   for i in range(1000)]
# all_pred_masses_ps = np.concatenate(pred_masses_ps)
# np.save("/share/data1/lls/sim200/volume_sharp_k/ALL_PS_predicted_masses.npy", all_pred_masses_ps)
#
# pred_masses_st = [np.lib.format.open_memmap("/share/data1/lls/sim200/volume_sharp_k/"
#                                             "ST/ST_predicted_masses_" + str(i) + ".npy", mode="r",
#                                             shape=(len(a[i]), 910))
#                   for i in range(1000)]
# all_pred_masses_st = np.concatenate(pred_masses_st)
# np.save("/share/data1/lls/sim200/volume_sharp_k/ALL_ST_predicted_masses.npy", all_pred_masses_st)


# Concatenate volume sharp-k results for the cut mass ranges

a = np.array(np.array_split(np.arange(512**3), 1000, axis=0))

pred_masses_ps = [np.lib.format.open_memmap("/share/data1/lls/sim200/volume_sharp_k/cut_at_m_15/"
                                            "PS/PS_predicted_masses_" + str(i) + ".npy", mode="r",
                                            shape=(len(a[i]), 910))
                  for i in range(1000)]
all_pred_masses_ps = np.concatenate(pred_masses_ps)
np.save("/share/data1/lls/sim200/volume_sharp_k/cut_at_m_15/ALL_PS_predicted_masses.npy", all_pred_masses_ps)

pred_masses_st = [np.lib.format.open_memmap("/share/data1/lls/sim200/volume_sharp_k/cut_at_m_15/"
                                            "ST/ST_predicted_masses_" + str(i) + ".npy", mode="r",
                                            shape=(len(a[i]), 910))
                  for i in range(1000)]
all_pred_masses_st = np.concatenate(pred_masses_st)
np.save("/share/data1/lls/sim200/volume_sharp_k/cut_at_m_15/ALL_ST_predicted_masses.npy", all_pred_masses_st)

