"""
Plot trajectories

"""
import pynbody
import numpy as np
from scripts.hmf import hmf_tests as ht
from mlhalos import parameters
import matplotlib.pyplot as plt
from scripts.hmf import number_modes as nm
from mlhalos import distinct_colours


# Original smoothing scales

ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/CODE")

m_bins = 10**np.arange(10, 15, 0.0033).view(pynbody.array.SimArray)
m_bins.units = "Msol h^-1"
r = ht.pynbody_m_to_r(m_bins, ic.initial_conditions)
k_smoothing = 1/r

boxsize = pynbody.array.SimArray([50])
k_min = 2*np.pi/boxsize

k_smoothing = 1/r
index_above_k_min = np.where(k_smoothing >= k_min)[0]

k_final = k_smoothing[index_above_k_min]
t = np.load("/Users/lls/Documents/CODE/stored_files/traj_examples/traj_example_above_k_min.npy")

index = [35, 43, 106,
         55, 98]

col = distinct_colours.get_distinct(7)



# Large box stuff

ic_200 = parameters.InitialConditionsParameters(initial_snapshot="/Users/lls/Documents/CODE/sim200/sim200.gadget3",
                                                path="/Users/lls/Documents/CODE/")
m_bins_large = 10 ** np.arange(10, 16, 0.0033).view(pynbody.array.SimArray)
m_large = m_bins_large[::2]
r_large = ht.pynbody_m_to_r(m_large, ic_200.initial_conditions)
k_large = 1/r_large

boxsize_large = pynbody.array.SimArray([200])
k_min_large = 2*np.pi/boxsize_large

index_above_k_min_large = np.where(k_large >= k_min_large)[0]

k_final_large = k_large[index_above_k_min_large]
# t_0 = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/subbox_62155961/trajectories/trajectories_ids_8.npy")
t_0 = np.load("/Users/lls/Documents/CODE/stored_files/hmf/sim200/trajectories/trajectories_ids_63.npy")

ind_l = np.random.choice(len(t_0), 5)

# Plot

fig, (ax1, ax3) = plt.subplots(2, sharex=True, figsize=(14,9))
# ax1 = fig.add_subplot(111)

ax2 = ax1.twiny()

[ax1.plot(k_final/k_min, t[ind]) for ind in index]
# ax1.set_xlabel(r"$k /\left( 2\pi/L \right)$")
ax1.set_ylabel(r"$\delta + 1$")
ax1.set_ylim(0.98, 1.025)
# ax1.set_ylim(0.96, 1.06)
ax1.set_xlim(1, 3)
# ax1.get_xticklabels()

[ax3.plot(k_final_large/k_min, t_0[ind_l, index_above_k_min_large]) for ind_l in index]
ax3.set_xlabel(r"$k /\left( 2\pi/L \right)$")
ax3.set_ylabel(r"$\delta + 1$")
ax3.set_ylim(ax1.get_ylim())
ax3.set_xlim(1, 3)

modes = ax3.get_xticks()
k_sm_ax2 = modes * (2*np.pi/boxsize)
m_ax2 = ht.pynbody_r_to_m(1/k_sm_ax2, ic_200.initial_conditions)
m1_ax2 = ["%.2e" % z for z in m_ax2]
ax2.set_xlim(ax3.get_xlim())
ax2.set_xticklabels(m1_ax2)
ax2.set_xlabel(r"$M [M_{\odot}/h]$")


# ax4 = ax3.twiny()
#
# [ax3.plot(k_final_large/k_min, t_0[ind, index_above_k_min_large]) for ind in index]
# ax3.set_xlabel(r"$k /\left( 2\pi/L \right)$")
# ax3.set_ylabel(r"$\delta + 1$")
# ax3.set_ylim(0.96, 1.06)
# ax3.set_xlim(1, 10)

fig.subplots_adjust(hspace=0)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.subplots_adjust(top=0.9)

# modes_large = ax3.get_xticks()
# k_sm_ax3 = modes_large * (2*np.pi/boxsize)
# m_ax4 = ht.pynbody_r_to_m(1/k_sm_ax3, ic_200.initial_conditions)
# m1_ax4 = ["%.2e" % z for z in m_ax4]
# ax4.set_xlim(ax3.get_xlim())
# ax4.set_xticklabels(m1_ax4)
# ax4.set_xlabel(r"$M [M_{\odot}/h]$")
#
# fig.subplots_adjust(hspace=0.3)


# plot big box stuff


plt.tight_layout()

# plt.show()


fig, (ax1, ax3) = plt.subplots(2, figsize=(14,9))
# ax1 = fig.add_subplot(111)

# ax2 = ax1.twiny()
ax1.plot(k_final/k_min, t[index[0]], label="small")
[ax1.plot(k_final/k_min, t[ind]) for ind in index[1:]]
ax1.set_xlabel(r"$k /\left( 2\pi/L \right)$")
ax1.set_ylabel(r"$\delta + 1$")
ax1.set_ylim(0.96, 1.04)
# ax1.set_ylim(0.96, 1.06)
ax1.set_xlim(1, 10)
# ax1.set_legend()
# ax1.get_xticklabels()

ax3.plot(k_final_large[t_0[ind_l[0], index_above_k_min_large]!=0]/k_min_large, t_0[ind_l[0]][t_0[ind_l[0],
                                                                                           index_above_k_min_large]!=0], label="large")
[ax3.plot(k_final_large[t_0[ind_ll, index_above_k_min_large]!=0]/k_min_large, t_0[ind_ll, index_above_k_min_large][t_0[ind_ll, index_above_k_min_large]!=0]) for\
        ind_ll in ind_l]
ax3.set_xlabel(r"$k /\left( 2\pi/L \right)$")
ax3.set_ylabel(r"$\delta + 1$")
ax3.set_ylim(0.98, 1.03)
ax3.set_xlim(1, 10)
# ax3.get_legend()

#plt.legend(loc="best")
fig.subplots_adjust(hspace=0.3)



# plot number of modes at each k_scale

num_emp = nm.get_number_of_modes_below_k_scale(ic, k_final)
plt.plot(k_final/k_min, num_emp)
plt.plot(k_final[::20]/k_min, num_emp[::20])
# plt.plot(k_final[::50]/k_min, num_emp[::50])

plt.yscale("log")
plt.xlabel(r"k $/\left( 2\pi/L \right)$ [h Mpc$^{-1}$ a$^{-1}$]")
plt.ylabel("Number of modes")


