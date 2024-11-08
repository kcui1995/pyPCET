import numpy as np
import matplotlib.pyplot as plt
from pyPCET import pyPCET
from pyPCET.functions import fit_poly8 
from pyPCET.units import massH, massD

# define temperature, electronic coupling, reaction free energy, and reorganization energy 
T = 298
Vel = 0.0434
dG = -0.50
Lambda = 1.00


# double well potentials calculated from first principles 
rp = np.array([-0.614,-0.550,-0.485,-0.420,-0.356,-0.291,-0.226,-0.162,-0.097,-0.032,0.032,0.097,0.162,0.226,0.291,0.356,0.420,0.485,0.550,0.614])
E_LES = np.array([5.283,4.534,4.061,3.794,3.673,3.646,3.680,3.688,3.634,3.646,3.602,3.513,3.392,3.257,3.138,3.078,3.140,3.413,4.022,5.144])
E_LEPT = np.array([4.847,4.134,3.706,3.495,3.452,3.509,3.620,3.801,3.949,4.073,4.157,4.194,4.189,4.157,4.128,4.144,4.270,4.597,5.250,6.411])

ReacProtonPot = fit_poly8(rp, E_LES) 
ProdProtonPot = fit_poly8(rp, E_LEPT)

# set up system and do a calculation
system = pyPCET(ReacProtonPot, ProdProtonPot, dG, Lambda, Vel=Vel)
system.calculate(massH, T=T)


#===========================================================
# Plot proton vibrational wave functions
#===========================================================

fig = plt.figure(figsize=(9,4.5))
gs = fig.add_gridspec(ncols=2, wspace=0)
ax1,ax2 = gs.subplots(sharex=True, sharey=True)

Evib_reactant, wfc_reactant = system.get_reactant_proton_states()
Evib_product, wfc_product = system.get_product_proton_states()
rp = system.rp
NStates_to_show = 6

# align the zero-point energy of the reactant and product states in this plot
if Evib_product[0] < Evib_reactant[0]:
    dEr = 0
    dEp = -Evib_product[0] + Evib_reactant[0]
else:
    dEr = Evib_product[0] - Evib_reactant[0]
    dEp = 0

ax1.plot(-rp, ReacProtonPot(rp)+dEr, 'b', lw=2)
ax1.plot(-rp, ProdProtonPot(rp)+dEp, 'r', lw=2)
scale_wfc = 0.06        # we will plot wave functions and energies in the same plot, this factor scales the wave function for better visualization

for i, (Ei, wfci) in enumerate(zip(Evib_reactant[:NStates_to_show], wfc_reactant[:NStates_to_show])):
    # change the sign of the vibrational wave functions for better visualization
    # make the largest amplitude positive
    sign = 1 if np.abs(np.max(wfci)) > np.abs(np.min(wfci)) else -1
    ax1.plot(-rp, Ei+dEr+scale_wfc*sign*wfci, 'b-', lw=1, alpha=(1-0.12*i))
    ax1.fill_between(-rp, Ei+dEr+scale_wfc*sign*wfci, Ei+dEr, color='b', alpha=0.4)

ax1.set_xlim(-0.8,0.8)
ax1.set_ylim(0,1.3)
ax1.set_xlabel(r'$r_{\rm p}\ /\ \rm\AA$', fontsize=16)
ax1.set_ylabel(r'$E$ / eV', fontsize=16)
ax1.tick_params(labelsize=14)


ax2.plot(-rp, ReacProtonPot(rp)+dEr, 'b', lw=2)
ax2.plot(-rp, ProdProtonPot(rp)+dEp, 'r', lw=2)

for i, (Ei, wfci) in enumerate(zip(Evib_product[:NStates_to_show], wfc_product[:NStates_to_show])):
    sign = 1 if np.abs(np.max(wfci)) > np.abs(np.min(wfci)) else -1
    ax2.plot(-rp, Ei+dEp+scale_wfc*sign*wfci, 'r-', lw=1, alpha=(1-0.12*i))
    ax2.fill_between(-rp, Ei+dEp+scale_wfc*sign*wfci, Ei+dEp, color='r', alpha=0.4)


ax2.set_xlim(-0.65,0.75)
ax2.set_ylim(0,1.3)
ax2.set_xlabel(r'$r_{\rm p}\ /\ \rm\AA$', fontsize=16)
ax2.set_xticks(np.arange(-0.6,0.8,0.2))
ax2.tick_params(labelsize=14)

plt.tight_layout()
plt.show()


#===========================================================
# Analyze the contribution of each pair of vibronic states 
#===========================================================

Pu = system.get_reactant_state_distributions()
Suv = system.get_proton_overlap_matrix()
dGuv = system.get_reaction_free_energy_matrix()
dGa_uv = system.get_activation_free_energy_matrix() 
kuv = system.get_kinetic_contribution_matrix()
k_tot = system.get_total_rate_constant()
percentage_contribution = kuv/k_tot

# print a table for these quantities
NStates_to_show = 4
print('\n' + '='*130)
print('(u, v)\t\tP_u\t\t\t|S_uv|^2\t\tDelta G_uv / eV\t\tDelta G^#_uv / eV\t% Contrib.')
print('-'*130)
for u in range(NStates_to_show):
    for v in range(NStates_to_show):
        print(f'({u:d}, {v:d})\t\t{Pu[u]:.3e}\t\t{Suv[u,v]*Suv[u,v]:.3e}\t\t{dGuv[u,v]:+.3f}\t\t\t{dGa_uv[u,v]:.3f}\t\t\t{percentage_contribution[u,v]:.3f}')

print('='*130 + '\n')


#===========================================================
# Calculate rate constants for H and D, and the KIE 
#===========================================================

k_tot_H = system.calculate(massH, T)
k_tot_D = system.calculate(massD, T)

print(f'At {T:d}K, k_tot(H) = {k_tot_H:.2e} s^-1')
print(f'At {T:d}K, k_tot(D) = {k_tot_D:.2e} s^-1')
print(f'At {T:d}K, KIE = {k_tot_H/k_tot_D:.2f}')
