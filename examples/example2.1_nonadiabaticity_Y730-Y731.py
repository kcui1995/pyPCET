import numpy as np
import matplotlib.pyplot as plt
from pyPCET import kappa_coupling 
from pyPCET.functions import bspline 
from scipy.interpolate import CubicSpline
from pyPCET.units import kcal2eV, massH


# double well potentials and electronic coupling read from a file
# In this file, all energies are in kcal/mol
rp_read, E_Reac, E_Prod, Vel = np.loadtxt('Y730_Y731_config1_env.dat', unpack=True)
E_Reac *= kcal2eV
E_Prod *= kcal2eV
Vel *= kcal2eV

# Spline the proton potential
# this works better for smooth data
ReacProtonPot = bspline(rp_read, E_Reac) 
ProdProtonPot = bspline(rp_read, E_Prod)
Vel_rp = CubicSpline(rp_read, Vel)

# define a finer rp grid
rp = np.linspace(np.min(rp_read), np.max(rp_read), 256)

# setup the system and perform the calculation
system = kappa_coupling(rp, ReacProtonPot(rp), ProdProtonPot(rp), Vel_rp(rp)) 
system.calculate(massH)


#===========================================================
# Print nonadiabaticity data 
#===========================================================

tau_e, tau_p, p, kappa = system.get_nonadiabaticity_parameters()
V_sc, V_nad, V_ad = system.get_vibronic_couplings()

print(f'tau_p = {tau_p:.3e} s')
print(f'tau_e = {tau_e:.3e} s')
print(f'p = {p:.3e}')
print(f'kappa = {kappa:.3f}')

print(f'V_ad = {V_ad:.2e} eV = {V_ad/kcal2eV:.2e} kcal/mol')
print(f'V_nad = {V_nad:.2e} eV = {V_nad/kcal2eV:.2e} kcal/mol')
print(f'V_sc = {V_sc:.2e} eV = {V_sc/kcal2eV:.2e} kcal/mol')


#===========================================================
# Plot the proton potentials 
#===========================================================

shifted_pot_reactant, shifted_Evib_reactant, wfc_reactant = system.get_reactant_proton_states()
shifted_pot_product, shifted_Evib_product, wfc_product = system.get_product_proton_states()

# get the proton coordinate, energy, electronic coupling, and slopes at the crossing point
# for plot use only
rp_crossing = system.rp_crossing
E_crossing = system.E_crossing 
Vel_crossing = system.Vel_crossing
slope_Reac = system.slope_reac
slope_Prod = system.slope_prod

plt.plot(rp, shifted_pot_reactant, 'b', lw=2)
plt.plot(rp, shifted_pot_product, 'r', lw=2)

plt.plot(rp_crossing, E_crossing, 'o', ms=5, mew=2, mfc='k', mec='k')

tmp_x = np.linspace(rp_crossing-0.1, rp_crossing+0.1, 100)
plt.plot(tmp_x, slope_Reac*(tmp_x-rp_crossing)+E_crossing, 'k--', lw=1.5)
plt.plot(tmp_x, slope_Prod*(tmp_x-rp_crossing)+E_crossing, 'k--', lw=1.5)


# plot proton vibrational wave functions
scale_wfc = 0.06
NStates_to_show = 1

# change the sign of the vibrational wave functions for better visualization
# make the largest amplitude positive
sign = 1 if np.abs(np.max(wfc_reactant[0])) > np.abs(np.min(wfc_reactant[0])) else -1
plt.plot(rp, shifted_Evib_reactant[0]+scale_wfc*sign*wfc_reactant[0], 'b-', lw=1, alpha=1)
plt.fill_between(rp, shifted_Evib_reactant[0]+scale_wfc*sign*wfc_reactant[0], shifted_Evib_reactant[0], color='b', alpha=0.4)

sign = 1 if np.abs(np.max(wfc_product[0])) > np.abs(np.min(wfc_product[0])) else -1
plt.plot(rp, shifted_Evib_product[0]+scale_wfc*sign*wfc_product[0], 'r-', lw=1, alpha=1)
plt.fill_between(rp, shifted_Evib_product[0]+scale_wfc*sign*wfc_product[0], shifted_Evib_product[0], color='r', alpha=0.4)


plt.xlim(-1.5,2.0)
plt.ylim(0,2.0)
plt.xlabel(r'$r_{\rm p}\ /\ \rm\AA$', fontsize=18)
plt.ylabel(r'$E$ / eV', fontsize=18)
plt.xticks(np.arange(-1.5,2.0,0.5), fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.savefig('Proton_pot_w_slope_Y730_Y731.pdf')
plt.clf()


#===========================================================
# Plot adiabatic proton potentials
#===========================================================

plt.plot(rp, shifted_pot_reactant, 'b', lw=2)
plt.plot(rp, shifted_pot_product, 'r', lw=2)

Eg, Ee = system.get_adiabatic_proton_potentials()

plt.plot(rp, Eg, 'k--', lw=1.5)
plt.plot(rp, Ee, 'k--', lw=1.5)

plt.xlim(-1.5,2.0)
plt.ylim(-1.0,4.0)
plt.xlabel(r'$r_{\rm p}\ /\ \rm\AA$', fontsize=18)
plt.ylabel(r'$E$ / eV', fontsize=18)
plt.xticks(np.arange(-1.5,2.0,0.5), fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.savefig('Proton_pot_adiabatic_Y730_Y731.pdf')

