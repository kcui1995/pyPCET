import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyPCET import pyPCET
from pyPCET.functions import fit_poly6, fit_poly8
from pyPCET.units import massH, massD
from pyPCET.units import kB, kcal2eV, A2Bohr, Ha2eV
from scipy.integrate import simps
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


#=========================================================================================
# Define the thermodynamic parameters
# for photochemical oxidation of BIP
#=========================================================================================

# R values sampled in calculations
Rs = np.arange(2.37,2.92,0.05)

DeltaG = -5.0*kcal2eV
Lambda = 21.4*kcal2eV
Vel = 1*kcal2eV             # Vel is not needed for KIE calculation, use a default value of 1 kcal/mol
T = 298.15

NStates = 20                # how many states to include in rate constant calculation
NStates_to_show = 7         # how many states to plot/print

keff = 0.0443               # effective force constant for proton donor-acceptor motion, unit in a.u.
R_eq = 2.58                 # equilibrium proton donor-acceptor distance


#=========================================================================================
# Read data from files
#=========================================================================================

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


#=========================================================================================
# Calculate the KIE of electrochemical PCET of BIP at different R
#=========================================================================================

# for photochemical PCET, the integration over epsilon is not needed 

kH_R = np.zeros(len(Rs))
kD_R = np.zeros(len(Rs))

for i,R in enumerate(Rs):
    print(f'Calculating... R = {R:.2f}A')
    system = pyPCET(ReacProtonPot_R[i], ProdProtonPot_R[i], DeltaG=DeltaG, Lambda=Lambda, Vel=Vel, NStates=NStates, rmin=-1.0, rmax=1.0)

    kH_R[i] = system.calculate(mass=massH, T=T)
    kD_R[i] = system.calculate(mass=massD, T=T)

    # Plot of the wave functions and print of the contributions are omitted in this example
    # They are the same as in the electrochemical case because the same proton potentials are used

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
