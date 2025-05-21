import numpy as np
import matplotlib.pyplot as plt
from pyPCET import pyPCET
from pyPCET.functions import bspline 
from pyPCET.units import massH, massD
from pyPCET.units import kcal2eV
from scipy.integrate import simps
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


#=========================================================================================
# Define the thermodynamic parameters, taken from 
# Zhong et. al. J. Am. Chem. Soc. 2025, 147, 4459−4468 
#=========================================================================================

# R values sampled in calculations
Rs = np.arange(2.42,3.2,0.1)

DeltaG = +0.0*kcal2eV          # For PCET between two tyrosines, DeltaG = 0
Lambda = 18.86*kcal2eV
Vel = 0.8*kcal2eV
T = 298

NStates = 7                    # how many states to include in rate constant calculation
NStates_to_show = 4            # how many states to plot/print


#=========================================================================================
# Read data from files
#=========================================================================================

# define rainbow color for plotting proton potentials
import colorsys
hues = np.linspace(0.0,0.8,len(Rs))
colors = [colorsys.hls_to_rgb(hue,0.5,0.85) for hue in hues]

ReacProtonPot_R = []
ProdProtonPot_R = []

rp = np.linspace(-1.5, 1.5, 256)
rlim = np.zeros(len(Rs))

# read proton potentials from .dat files
for i, R in enumerate(Rs):
    rp_tmp, E_reac_tmp, E_prod_tmp = np.loadtxt(f'proton_potentials/R{R:.2f}_potential.dat', usecols=(0,1,2), unpack=True)
    rlim[i] = rp_tmp[-1]

    # the energy unit in the .dat files is kcal/mol, convert it to eV
    E_reac_tmp *= kcal2eV
    E_prod_tmp *= kcal2eV

    # smooth the data by splining
    ReacProtonPot_R.append(bspline(rp_tmp, E_reac_tmp))
    ProdProtonPot_R.append(bspline(rp_tmp, E_prod_tmp))

# plot the proton potentials
fig = plt.figure(figsize=(8,4))
gs = fig.add_gridspec(ncols=2, wspace=0)
ax1,ax2 = gs.subplots(sharex=True, sharey=True)

for i, R in enumerate(Rs):
    ax1.plot(rp, ReacProtonPot_R[i](rp)/kcal2eV, '-', lw=2, color=colors[i])
    ax2.plot(rp, ProdProtonPot_R[i](rp)/kcal2eV, '-', lw=2, color=colors[i])

ax2.set_xlim(-1.2,1.2)
ax2.set_ylim(0,100)
ax1.set_xlabel(r'$r_{\rm p} (\rm\AA)$',fontsize=16)
ax2.set_xlabel(r'$r_{\rm p} (\rm\AA)$',fontsize=16)
ax1.set_ylabel(r'$E$ / (kcal/mol)', fontsize=16)
ax2.set_xticks(np.arange(-1.0,1.5,0.5))
ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig('Proton_potentials.png', dpi=300)
plt.clf()


#=========================================================================================
# Calculate the PCET rate constant between Y356 and Y731 
#=========================================================================================

kH_R = np.zeros(len(Rs))

for i,R in enumerate(Rs):
    print(f'Calculating... R = {R:.2f}A')
    system = pyPCET(ReacProtonPot_R[i], ProdProtonPot_R[i], DeltaG=DeltaG, Lambda=Lambda, Vel=Vel, NStates=NStates, rmin=-rlim[i], rmax=rlim[i], NGridiPot=512)

    kH_R[i] = system.calculate(mass=massH, T=T)

    # plot the wave functions and print the state contrtbutions for each R 

    fig = plt.figure(figsize=(9,4.5))
    gs = fig.add_gridspec(ncols=2, wspace=0)
    ax1,ax2 = gs.subplots(sharex=True, sharey=True)

    system.calculate(mass=massH, T=T)
    Evib_reactant, wfc_reactant = system.get_reactant_proton_states()
    Evib_product, wfc_product = system.get_product_proton_states()
    rp = system.rp

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

    ax2.set_xlim(-1.2,1.2)
    ax2.set_ylim(0,2.5)
    ax1.set_xlabel(r'$r_{\rm p}\ /\ \rm\AA$', fontsize=16)
    ax1.set_ylabel(r'$E$ / eV', fontsize=16)
    ax2.set_xlabel(r'$r_{\rm p}\ /\ \rm\AA$', fontsize=16)
    ax2.set_xticks(np.arange(-1.0,1.5,0.5))
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig(f'Proton_states_H_R{R:.2f}.png', dpi=300)
    plt.clf()

    # print a table for these quantities
    # write to a file
    with open(f'rate_constant_contribution_R{R:.2f}A.log', 'w') as outfp:
        Pu = system.get_reactant_state_distributions()
        Suv = system.get_proton_overlap_matrix()
        dGuv = system.get_reaction_free_energy_matrix()
        dGa_uv = system.get_activation_free_energy_matrix()
        kuv = system.get_kinetic_contribution_matrix()
        k_tot = system.get_total_rate_constant()
        percentage_contribution = kuv/k_tot

        outfp.write(f'\nR = {R:.2f}A\n')
        outfp.write('(u, v)\t\tP_u\t\t\t|S_uv|\t\tDelta G_uv / eV\t\tDelta G^#_uv / eV\t% Contrib.\n')
        outfp.write('-'*125 + '\n')
        for u in range(NStates_to_show):
            for v in range(NStates_to_show):
                outfp.write(f'({u:d}, {v:d})\t\t{Pu[u]:.3e}\t\t{np.abs(Suv[u,v]):.3e}\t\t{dGuv[u,v]:+.3f}\t\t\t{dGa_uv[u,v]:.3f}\t\t\t{percentage_contribution[u,v]*100:.1f}\n')
        outfp.write('='*125 + '\n\n')



# Print PCET rate constants for H at each R to a file
with open('kPCET_data.log', 'w') as outfp:
    outfp.write('# R_PT/A\tk_H/s^-1\n')
    for i,R in enumerate(Rs):
        outfp.write(f'{R:.2f}\t\t{kH_R[i]:.4e}\n')


#=========================================================================================
# Thermally average the PCET rate constant over R
#=========================================================================================

color1 = '#ff7700'
color2 = (0.1,0.6,0.2)

# the integration should be from 0 to infinity, but in reality we perform the integral in the interval that the integrand reaches zero at both limits
R_fine_grid = np.linspace(2.0, 4.0, 500) 

# Fit k(R) by fitting logk(R) to a quadratic function
# !!!NOTE!!! Always check if k_PCET * P(R) reaches zero at the limit of your sampled R values

def quadratic(x, a, b, c):
    return a*x*x + b*x + c

a1, b1, c1 = curve_fit(quadratic, Rs, np.log(kH_R))[0]
a2, b2, c2 = curve_fit(quadratic, Rs, np.log(kD_R))[0]

kH_fine_grid = np.exp(quadratic(R_fine_grid, a1, b1, c1))

# P(R) is calculated from umbrella sampling
# Read the umbrella sampling data from file and fit logP(R) to a 4th order polynomial
R_tmp, PR_tmp = np.loadtxt('p_R_umbrella_A.dat', unpack=True)

def poly4(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

params = curve_fit(poly4, R_tmp, np.log(PR_tmp))[0]
PR = np.exp(poly4(R_fine_grid, *params))

# re-normalize the distribution
Z = simps(PR, R_fine_grid)
PR /= Z
R_eq = R_fine_grid[find_peaks(PR)[0]]

# perform thermal average and print the final results
Rmax_H = R_fine_grid[find_peaks(PR*kH_fine_grid)[0]]


ave_kH = simps(PR*kH_fine_grid, R_fine_grid)

print()
print(f'Dominant R for H = {Rmax_H[0]:.2f}A')
print(f'k_H_tot = {ave_kH:.4e} s^-1')


#=========================================================================================
# plot k(R), P(R), k(R)*P(R) for H
#=========================================================================================

plt.figure(figsize=(4.8,4.5))

plt.title('H', fontsize=20)
plt.plot(R_fine_grid, kH_fine_grid/np.max(kH_fine_grid), '-', lw=2, color=color1, label=r'$k_{\rm PCET}(R)$')
plt.plot(R_fine_grid, PR/np.max(PR), '-', lw=2, color=color2, label='$P(R)$')
plt.plot(R_fine_grid, PR*kH_fine_grid/np.max(PR*kH_fine_grid), '-', lw=2, color='k', label=r'$P(R)k_{\rm PCET}(R)$')

plt.axvline(x=R_eq, linewidth=1.5, color='darkgray', linestyle=(0, (3, 3)))
plt.axvline(x=Rmax_H[0], linewidth=1.5, color='k', linestyle=(0, (3, 3)))

plt.legend(fontsize=16,frameon=False, loc=1)
plt.xlim(2.10,3.7)
plt.ylim(0,1.2)
plt.xlabel(r'$R$ / $\rm \AA$', fontsize=18)
plt.xticks(np.arange(2.25,3.75,0.25), fontsize=16)
plt.yticks([])


plt.tight_layout()
plt.savefig('kR-PR.png', dpi=300)
plt.clf()
