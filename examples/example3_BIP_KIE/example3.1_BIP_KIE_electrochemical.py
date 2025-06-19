import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyPCET import pyPCET
from pyPCET.functions import fit_poly6, fit_poly8
from pyPCET.units import massH, massD
from pyPCET.units import kB, kcal2eV, A2Bohr, Ha2eV
from pyPCET.electrochemistry import Fermi_distribution
try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


#=========================================================================================
# Define the thermodynamic parameters, taken from 
# Huynh et. al. ACS Cent. Sci. 2017, 3, 372−380
#=========================================================================================

# R values sampled in calculations
Rs = np.arange(2.37,2.92,0.05)

eta = 0                     # unit in V
Lambda = 21.4*kcal2eV
Vel = 1*kcal2eV             # Vel is not needed for KIE calculation, use a default value of 1 kcal/mol
T = 298.15

# beta' and rho_M parameters in Eqs. (S2) and (S3) are also not needed for KIE calculation, set them equal to 1
beta = 1                    # unit in A^-1
rho_M = 1                   # unit in eV^-1

# In this work, Delta G will depend on the energy level of the electrode state (relative to the Fermi level) epsilon, and the overpotential eta 
# Delta G_a/c = Delta G^0 +/- epsilon -/+ e*eta (a: anodic, c: cathodic)
# Delta G^0 is the reaction free energy at equilibrium potential (eta = 0), which should be 0 by definition
# we will first set Delta G = 0 and then update it for actual calculations (see below). 
DeltaG = 0

NStates = 20                # how many states to include in rate constant calculation
NStates_to_show = 7         # how many states to plot/print


keff = 0.0443               # effective force constant for proton donor-acceptor motion, unit in a.u.
R_eq = 2.58                 # equilibrium proton donor-acceptor distance


#=========================================================================================
# Read data from files
#=========================================================================================

# define rainbow color for plotting proton potentials
import colorsys
hues = np.linspace(0.0,0.8,len(Rs))
colors = [colorsys.hls_to_rgb(hue,0.5,0.85) for hue in hues]

ReacProtonPot_R = []
ProdProtonPot_R = []

rp = np.linspace(-1.0, 1.0, 256)

# read proton potentials from .csv files
# the proton potentials are digitized from Figure S39 of
# Huynh et. al. ACS Cent. Sci. 2017, 3, 372−380
for i, R in enumerate(Rs):
    dat_red = pd.read_csv(f'proton_potentials/Reduced_BIP_{R:.2f}A.csv', sep=', ', header=0, engine='python')
    dat_ox = pd.read_csv(f'proton_potentials/Oxidized_BIP_{R:.2f}A.csv', sep=', ', header=0, engine='python')
    rp_red_tmp = dat_red['x']
    E_red_tmp = dat_red['y']
    rp_ox_tmp = dat_ox['x']
    E_ox_tmp = dat_ox['y']

    # the energy unit in the .csv files is kcal/mol, convert it to eV
    E_red_tmp *= kcal2eV
    E_ox_tmp *= kcal2eV

    # smooth the data by fitting to polynomials
    # for these data set, fitting to polynomial is better than spline
    # 8th order polynomial gives the best fit, except for the reduced BIP at R = 2.37A
    # 6th order polynomial is used for that set of data
    if i == 0:
        ReacProtonPot_R.append(fit_poly6(rp_red_tmp, E_red_tmp))
    else:
        ReacProtonPot_R.append(fit_poly8(rp_red_tmp, E_red_tmp))

    ProdProtonPot_R.append(fit_poly8(rp_ox_tmp, E_ox_tmp))

# plot the proton potentials
fig = plt.figure(figsize=(5,7))
gs = fig.add_gridspec(2, hspace=0)
ax1,ax2 = gs.subplots(sharex=True, sharey=True)

for i, R in enumerate(Rs):
    ax1.plot(rp, ReacProtonPot_R[i](rp), '-', lw=2, color=colors[i])
    ax2.plot(rp, ProdProtonPot_R[i](rp), '-', lw=2, color=colors[i])

ax2.set_xlim(-1,1)
ax2.set_ylim(0,2.2)
ax2.set_xlabel(r'$r_{\rm p}\ /\ \rm\AA$',fontsize=16)
ax1.set_ylabel(r'$E$ / eV', fontsize=16)
ax2.set_ylabel(r'$E$ / eV', fontsize=16)
ax2.set_xticks(np.arange(-1.0,1.5,0.5))
ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig('Proton_potentials.png', dpi=300)
plt.clf()


#=========================================================================================
# Calculate the KIE of electrochemical PCET of BIP at different R and eta = 0
# The standard rate constant is approximated as the anodic rate constant at eta = 0
#=========================================================================================


# we sample 100 different epsilon values from -2 eV to 2 eV to perform the numerical intergation over electrode states
epsilons = np.linspace(-2, 2, 101)

kH_R = np.zeros(len(Rs))
kD_R = np.zeros(len(Rs))

for i,R in enumerate(Rs):
    print(f'Calculating... R = {R:.2f}A')
    
    # create two instances of the pyPCET object for H and D, respectively
    systemH = pyPCET(ReacProtonPot_R[i], ProdProtonPot_R[i], DeltaG=DeltaG, Lambda=Lambda, Vel=Vel, NStates=NStates, rmin=-1.0, rmax=1.0)
    systemD = pyPCET(ReacProtonPot_R[i], ProdProtonPot_R[i], DeltaG=DeltaG, Lambda=Lambda, Vel=Vel, NStates=NStates, rmin=-1.0, rmax=1.0)

    kH_epsilon = np.zeros(len(epsilons))
    kD_epsilon = np.zeros(len(epsilons))
    for j,epsilon in enumerate(epsilons):
        # update Delta G for a given epsilon
        # we only calculate the anodic rate constant here
        dG_anodic = epsilon - eta 
        systemH.set_parameters(DeltaG=dG_anodic)
        systemD.set_parameters(DeltaG=dG_anodic)
        kH_epsilon[j] = systemH.calculate(mass=massH, T=T, reuse_saved_proton_states=True)
        kD_epsilon[j] = systemD.calculate(mass=massD, T=T, reuse_saved_proton_states=True)

        # plot the wave functions and print the state contrtbutions for epsilon = 0
        # plot for proton and print for both H and D

        if epsilon == 0.0:
            fig = plt.figure(figsize=(9,4.5))
            gs = fig.add_gridspec(ncols=2, wspace=0)
            ax1,ax2 = gs.subplots(sharex=True, sharey=True)

            Evib_reactant, wfc_reactant = systemH.get_reactant_proton_states()
            Evib_product, wfc_product = systemH.get_product_proton_states()
            rp = systemH.rp

            # align the zero-point energy of the reactant and product states in this plot
            if Evib_product[0] < Evib_reactant[0]:
                dEr = 0
                dEp = -Evib_product[0] + Evib_reactant[0]
            else:
                dEr = Evib_product[0] - Evib_reactant[0]
                dEp = 0

            ax1.plot(rp, ReacProtonPot_R[i](rp)+dEr, 'b', lw=2)
            scale_wfc = 0.06        # we will plot wave functions and energies in the same plot, this factor scales the wave function for better visualization

            for ii, (Ei, wfci) in enumerate(zip(Evib_reactant[:NStates_to_show], wfc_reactant[:NStates_to_show])):
                # change the sign of the vibrational wave functions for better visualization
                # make the largest amplitude positive
                sign = 1 if np.abs(np.max(wfci)) > np.abs(np.min(wfci)) else -1
                ax1.plot(rp, Ei+dEr+scale_wfc*sign*wfci, 'b-', lw=1, alpha=(1-0.12*ii))
                ax1.fill_between(rp, Ei+dEr+scale_wfc*sign*wfci, Ei+dEr, color='b', alpha=0.4)

            ax2.plot(rp, ProdProtonPot_R[i](rp)+dEp, 'r', lw=2)

            for ii, (Ei, wfci) in enumerate(zip(Evib_product[:NStates_to_show], wfc_product[:NStates_to_show])):
                sign = 1 if np.abs(np.max(wfci)) > np.abs(np.min(wfci)) else -1
                ax2.plot(rp, Ei+dEp+scale_wfc*sign*wfci, 'r-', lw=1, alpha=(1-0.12*ii))
                ax2.fill_between(rp, Ei+dEp+scale_wfc*sign*wfci, Ei+dEp, color='r', alpha=0.4)

            ax2.set_xlim(-1.0,1.0)
            ax2.set_ylim(0,1.3)
            ax1.set_xlabel(r'$r_{\rm p}\ /\ \rm\AA$', fontsize=16)
            ax1.set_ylabel(r'$E$ / eV', fontsize=16)
            ax2.set_xlabel(r'$r_{\rm p}\ /\ \rm\AA$', fontsize=16)
            ax2.set_xticks(np.arange(-0.8,1.2,0.4))
            ax1.tick_params(labelsize=14)
            ax2.tick_params(labelsize=14)

            plt.tight_layout()
            plt.savefig(f'Proton_states_H_R{R:.2f}.png', dpi=300)
            plt.clf()

            # print a table for these quantities
            # write to a file
            with open(f'rate_constant_contribution_R{R:.2f}A.log', 'w') as outfp:
                # for H
                Pu = systemH.get_reactant_state_distributions()
                Suv = systemH.get_proton_overlap_matrix()
                dGuv = systemH.get_reaction_free_energy_matrix()
                dGa_uv = systemH.get_activation_free_energy_matrix()
                kuv = systemH.get_kinetic_contribution_matrix()
                k_tot = systemH.get_total_rate_constant()
                percentage_contribution = kuv/k_tot

                outfp.write(f'\nR = {R:.2f}A, epsilon = 0, eta = 0\n')
                outfp.write('\nH\n' + '='*125 + '\n')
                outfp.write('(u, v)\t\tP_u\t\t\t|S_uv|^2\t\tDelta G_uv / eV\t\tDelta G^#_uv / eV\t% Contrib.\n')
                outfp.write('-'*125 + '\n')
                for u in range(NStates_to_show):
                    for v in range(NStates_to_show):
                        outfp.write(f'({u:d}, {v:d})\t\t{Pu[u]:.3e}\t\t{Suv[u,v]*Suv[u,v]:.3e}\t\t{dGuv[u,v]:+.3f}\t\t\t{dGa_uv[u,v]:.3f}\t\t\t{percentage_contribution[u,v]*100:.1f}\n')
                outfp.write('='*125 + '\n\n')

                # for D
                Pu = systemD.get_reactant_state_distributions()
                Suv = systemD.get_proton_overlap_matrix()
                dGuv = systemD.get_reaction_free_energy_matrix()
                dGa_uv = systemD.get_activation_free_energy_matrix()
                kuv = systemD.get_kinetic_contribution_matrix()
                k_tot = systemD.get_total_rate_constant()
                percentage_contribution = kuv/k_tot

                outfp.write('\nD\n' + '='*125 + '\n')
                outfp.write('(u, v)\t\tP_u\t\t\t|S_uv|^2\t\tDelta G_uv / eV\t\tDelta G^#_uv / eV\t% Contrib.\n')
                outfp.write('-'*125 + '\n')
                for u in range(NStates_to_show):
                    for v in range(NStates_to_show):
                        outfp.write(f'({u:d}, {v:d})\t\t{Pu[u]:.3e}\t\t{Suv[u,v]*Suv[u,v]:.3e}\t\t{dGuv[u,v]:+.3f}\t\t\t{dGa_uv[u,v]:.3f}\t\t\t{percentage_contribution[u,v]*100:.1f}\n')
                outfp.write('='*125 + '\n\n')

    # calculate the anodic rate constant according to Eq. (S2) in the paper
    kH_R[i] = simps(rho_M/beta*(1-Fermi_distribution(epsilons, T=T))*kH_epsilon, epsilons)
    kD_R[i] = simps(rho_M/beta*(1-Fermi_distribution(epsilons, T=T))*kD_epsilon, epsilons)


# Print PCET rate constants for H and D at each R to a file
with open('kPCET_data.log', 'w') as outfp:
    outfp.write('# R_PT/A\tk_H/s^-1\tk_D/s^-1\n')
    for i,R in enumerate(Rs):
        outfp.write(f'{R:.2f}\t\t{kH_R[i]:.4e}\t{kD_R[i]:.4e}\n')


#=========================================================================================
# Thermally average the PCET rate constant over R
#=========================================================================================

# the integration should be from 0 to infinity, but in reality we perform the integral in the interval that the integrand reaches zero at both limits
R_fine_grid = np.linspace(2.0, 3.0, 200) 

# Interpolate k(R)
# In this calculation, at the smallest sampled R = 2.37A, k_PCET * P(R) is non-zero, 
# so we need to extrapolate k_PCET data to properly perform the integration from 0 to infinity
# !!!NOTE!!! Always check if k_PCET * P(R) reaches zero at the limit of your sampled R values

# Different interpolation and extrapolation method has been tested, they give similar KIE  
# Interpolate log(k) then take the exponential
kH_fine_grid = np.exp(interp1d(Rs, np.log(kH_R), kind='quadratic', fill_value='extrapolate')(R_fine_grid))
kD_fine_grid = np.exp(interp1d(Rs, np.log(kD_R), kind='quadratic', fill_value='extrapolate')(R_fine_grid))

# Directly interpolate k
#kH_fine_grid = interp1d(Rs, kH_R, kind='quadratic', fill_value='extrapolate')(R_fine_grid)
#kD_fine_grid = interp1d(Rs, kD_R, kind='quadratic', fill_value='extrapolate')(R_fine_grid)

# calculate P(R)
def PR(R, R0, keff, T):
    ER = 0.5*keff*(R-R0)*(R-R0)*A2Bohr*A2Bohr*Ha2eV
    kBT = kB*T
    return (np.exp(-ER/kBT))

PR = PR(R_fine_grid, R_eq, keff, T) 
Z = simps(PR, R_fine_grid)
PR /= Z

# perform thermal average and print the final results
Rmax_H = R_fine_grid[find_peaks(PR*kH_fine_grid)[0]]
Rmax_D = R_fine_grid[find_peaks(PR*kD_fine_grid)[0]]

ave_kH = simps(PR*kH_fine_grid, R_fine_grid)
ave_kD = simps(PR*kD_fine_grid, R_fine_grid)

print()
print(f'Dominant R for H = {Rmax_H[0]:.2f}A')
print(f'Dominant R for D = {Rmax_D[0]:.2f}A')
print(f'k_H_tot = {ave_kH:.4e} s^-1')
print(f'k_D_tot = {ave_kD:.4e} s^-1')
print(f'KIE = {ave_kH/ave_kD:.2f}')
