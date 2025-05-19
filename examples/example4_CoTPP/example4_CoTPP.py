import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyPCET import pyPCET
from pyPCET.functions import fit_poly6, fit_poly8
from pyPCET.units import massH, massD
from pyPCET.units import kB, kcal2eV, A2Bohr, Ha2eV
from pyPCET.electrochemistry import EDL_model, Fermi_distribution 
from scipy.integrate import simps
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

#=========================================================================================
# Define the thermodynamic parameters, taken from 
# Hutchison et. al. ACS Catat. 2024, 19, 14363–14372.
#=========================================================================================

# !!!NOTE!!! This script takes ~18h to run. 

# Donor-Acceptor distance values sampled in calculations
Rs = np.array([3.057,3.157,3.207,3.257,3.307,3.357,3.379,3.407,3.457,3.507,3.557,3.607,3.657,3.757,3.857,3.957,4.057,4.157,4.257])

Lambda = 0.83               # Units: eV. Inner sphere contribution only.
Vel = 0.10                  # Units: eV. Vel is not needed for KIE calculation, use a default value of 1 kcal/mol
T = 300                     # Units: K

#rho_M is the electronic density of states for a pristine graphene slab from a periodic planewave DFT calculation.
rho_M = np.genfromtxt('graphene_DOS_norm_gauss.csv', delimiter=',', skip_header=1) # unit in N_{states} eV^-1 atom^–1
epsilons = rho_M[:,0]      #Define the electronic energy levels that we will numerically integrate over
rho_DOS = rho_M[:,1]


# In this work, Delta G0 is the free energy change associated with the reaction 1/2 H_2 + CoTPP --> CoHTPP. 
# Further corrections will be made based on the applied electrode potential with the EDL model.

DeltaG0_H = 0.55 #eV
DeltaG0_D = 0.55-0.009296397 #eV

NStates = 10              # how many states to include in rate constant calculation
NStates_to_show = 9        # how many states to plot/print

#==========================================================================================
# Define the electric double layer model and non-bonded parameters used for the work terms
#==========================================================================================
dIHL = 3.6                  # angstrom
dOHL = 3.5                  # angstrom
eps_IHL = 2.7
eps_st = 78.0
eps_op = 1.78
dipole = 'calculate'
rho_water = 0.9970470       # g/cm^3
m_water = 18.01528          # g/mol
c_ions = 0.5                # mol/L
C_EDL = 15                  # microFarad/cm^2
PZFCvsSHE = 0.04            # V


potentials_vs_SHE = np.arange(-1.0,0.0,0.2)
colors = ['r', 'darkorange', 'g', 'b', 'purple']

fig = plt.figure(figsize=(6,3.5))

for i, EvsSHE in enumerate(potentials_vs_SHE):
    EDL_potential_drop = EDL_model(EvsSHE, dIHL, dOHL, eps_IHL, eps_st, eps_op, dipole, rho_water, m_water, c_ions, C_EDL, PZFCvsSHE, print_data=False)
    R = np.arange(0,10,0.1)
    plt.plot(R, EDL_potential_drop(R), '-', label=f'$E = {EvsSHE:.1f}$V', lw=1.5, color=colors[i])

plt.axvline(x=dIHL, linewidth=1.5, color='k', linestyle=(0, (3, 3)))
plt.axvline(x=dIHL+dOHL, linewidth=1.5, color='k', linestyle=(0, (3, 3)))

plt.legend(loc=4, frameon=True, framealpha=1, fontsize=14)
plt.xlim(0,10)
plt.xlabel(r'$R\ /\ \rm\AA$', fontsize=16)
plt.ylabel(r'$\phi(R,E)$ / V', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

plt.savefig('EDL_model.png', dpi=300)

# This defines the Buckingham potentials using the coefficients determined from fitting DFT data.
# Reactant: CoTPP + H_3O^+ non-bonded interaction
def reactant_work(Rs):
    return kcal2eV*(692272.09*np.exp((-1*Rs)/0.25730094) - 3699.5922/(Rs**6)) #  Units: eV

# Product: CoHTPP + H_2O non-bonded interaction
def product_work(Rs):
    return kcal2eV*(54305.39*np.exp((-1*Rs)/0.39709137) - 19539.245/(Rs**6)) # Units: eV

# Define a general Tafel equation where the prefactor is alpha*F/RT and b is ln[TOF(E0)]
# This is used in fitting the ln[k_H](E) curve to obtain the transfer coefficient, alpha
def Tafel(E_appl,prefactor,b):
    return -1*prefactor*E_appl + b

#=========================================================================================
# Read data from files
#=========================================================================================

ReacProtonPot_R = []
ProdProtonPot_R = []

# read proton potentials from .csv files
# the proton potentials are directly from the published work
for i, R in enumerate(Rs):
    dat_react = pd.read_csv(f'proton_potentials/rDA_{R:.3f}_R.csv', sep=',', header=0, engine='python')
    dat_prod = pd.read_csv(f'proton_potentials/rDA_{R:.3f}_P.csv', sep=',', header=0, engine='python')
    rp_react_tmp = dat_react['x']
    E_react_tmp = dat_react['Reactant']
    rp_prod_tmp = dat_prod['x']
    E_prod_tmp = dat_prod['Product']

    # the energy unit in the .csv files is Hartree atomic units, convert it to eV
    E_react_tmp *= Ha2eV
    E_prod_tmp *= Ha2eV

    # smooth the data by fitting to polynomials
    # for these data set, fitting to polynomial is better than spline
    # 6th order polynomial is used for that set of data
    ReacProtonPot_R.append(fit_poly6(rp_react_tmp, E_react_tmp))

    ProdProtonPot_R.append(fit_poly6(rp_prod_tmp, E_prod_tmp))


#=========================================================================================
# Calculate the KIE of electrochemical PCET of CoTPP at different R and applied potentials E 
#=========================================================================================


kH_R = np.zeros(len(Rs))
kD_R = np.zeros(len(Rs))

RTF = 8.31446*T/96485.33 #Units: J/C = V
pH = 0 #This helps convert from the RHE to SHE scale because the potential drop is defined on the SHE scale

# Build the applied potential list
E_appl_list = np.arange(-0.7,-0.49,0.01)

# Extract extrema of sampled proton-donor acceptor distances
R_min = Rs[0]
R_max = Rs[-1]

# Build the proton donor-accceptor distance grid based on the extrema of the sampled distances
R_fine_grid = np.linspace(R_min, R_max, 200) 


DeltaGR = np.zeros((2,len(Rs)))
kHD_of_E = np.zeros((3,len(E_appl_list)))

# Loop over applied potential
for n,E_appl in enumerate(E_appl_list):
    
    # Pre-generate the potential drop function
    EDL_potential_drop = EDL_model(E_appl, dIHL, dOHL, eps_IHL, eps_st, eps_op, dipole, rho_water, m_water, c_ions, C_EDL, PZFCvsSHE, print_data=False)
    
    # Loop over proton-donor acceptor distances
    for i,R in enumerate(Rs):
        react_work = reactant_work(R) + EDL_potential_drop(R)
        prod_work = product_work(R)
        DeltaW = prod_work - react_work

        # Correct the Delta G0 value with applied potential and work terms
        DeltaGH = DeltaG0_H + E_appl + prod_work - react_work + RTF*np.log(10)*pH
    
        # Correct the Delta G0 value with applied potential and work terms
        # The last term accounts for offsets in the pK_W for H_2O and D_2O which affect the potential scale
        DeltaGD = DeltaG0_D + E_appl + prod_work - react_work + RTF*np.log(10)*pH - RTF*np.log(10)*(14.0-14.87)
        
    
        print(f'Calculating... R = {R:.3f}A')
        systemH = pyPCET(ReacProtonPot_R[i], ProdProtonPot_R[i], DeltaG=DeltaGH, Lambda=Lambda, Vel=Vel, NStates=NStates, rmin=-1.5, rmax=1.5)
        systemD = pyPCET(ReacProtonPot_R[i], ProdProtonPot_R[i], DeltaG=DeltaGD, Lambda=Lambda, Vel=Vel, NStates=NStates, rmin=-1.5, rmax=1.5)

        kH_epsilon = np.zeros(epsilons.shape[0])
        kD_epsilon = np.zeros(epsilons.shape[0])
        DOS = np.zeros(epsilons.shape[0])
        for j,epsilon in enumerate(epsilons):
            # update Delta G for a given epsilon
            DeltaGH = DeltaG0_H + E_appl + prod_work - react_work - epsilon + RTF*np.log(10)*pH
        
            DeltaGD = DeltaG0_D + E_appl + prod_work - react_work - epsilon + RTF*np.log(10)*pH - RTF*np.log(10)*(14.0-14.87)
    
            systemH.set_parameters(DeltaG=DeltaGH)
            kH_epsilon[j] = systemH.calculate(mass=massH, T=T)
            systemD.set_parameters(DeltaG=DeltaGD)
            kD_epsilon[j] = systemD.calculate(mass=massD, T=T)
            
            if E_appl == -0.66:
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

                    outfp.write(f'\nR = {R:.2f}A, epsilon = 0.005, E_appl = –0.66\n')
                    outfp.write('\nH\n' + '='*125 + '\n')
                    outfp.write('(u, v)\t\tP_u\t\t\t|S_uv|^2\t\tDelta G_uv / eV\t\tDelta G^#_uv / eV\t% Contrib.\n')
                    outfp.write('-'*125 + '\n')
                    for u in range(NStates_to_show):
                        for v in range(NStates_to_show):
                            outfp.write(f'({u:d}, {v:d})\t\t{Pu[u]:.3e}\t\t{Suv[u,v]*Suv[u,v]:.3e}\t\t{dGuv[u,v]:+.3f}\t\t\t{dGa_uv[u,v]:.3f}\t\t\t{percentage_contribution[u,v]*100:.1f}\n')
                    outfp.write('='*125 + '\n\n')

                    # for D
                    systemD.calculate(mass=massD, T=T)
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
            # Multiply the DOS at a given epsilon by the Fermi-Dirac distribution at a given temperature
            DOS[j] = rho_DOS[j]*Fermi_distribution(epsilon, T=T)
    
        # Numerical integration over the electrode energy levels
        kH_R[i] = simps(DOS*kH_epsilon, epsilons)
        kD_R[i] = simps(DOS*kD_epsilon, epsilons)


    # Print PCET rate constants for H and D at each R to a file
    with open('kPCET_data.log', 'w') as outfp:
        outfp.write('# E / V\t R_PT/A\tk_H/s^-1\tk_D/s^-1\n')
        for i,R in enumerate(Rs):
            outfp.write(f'{E_appl:.2f}\t\t{R:.3f}\t\t{kH_R[i]:.4e}\t{kD_R[i]:.4e}\n')

#=========================================================================================
# Calculate P(R)
#=========================================================================================

    # Calculate the potential dependent concentration of proton donors at point R
    kBT = kB*T
    c0 = 1 #molar, for pH = 0 conditions
    W_of_R = reactant_work(R_fine_grid)+ EDL_potential_drop(R_fine_grid)
    PR = (np.exp(-1*(W_of_R)/kBT))*c0

#=========================================================================================
# Thermally average the PCET rate constant over R
#=========================================================================================

    # the integration should be from 0 to infinity, but in reality we perform the integral in the interval that the integrand reaches zero at both limits

    # Directly interpolate k
    kH_fine_grid = interp1d(Rs, kH_R, kind='linear', fill_value='extrapolate')(R_fine_grid)
    kD_fine_grid = interp1d(Rs, kD_R, kind='linear', fill_value='extrapolate')(R_fine_grid)

    # perform thermal average and print the final results
    Rmax_H = R_fine_grid[find_peaks(PR*kH_fine_grid)[0]]
    Rmax_D = R_fine_grid[find_peaks(PR*kD_fine_grid)[0]]

    ave_kH_of_E_appl = simps(PR*kH_fine_grid, R_fine_grid)
    ave_kD_of_E_appl = simps(PR*kD_fine_grid, R_fine_grid)

    print()
    print(f'Applied Potential= {E_appl:.2f} V vs SHE')
    print(f'Dominant R for H = {Rmax_H[0]:.2f}A')
    print(f'Dominant R for D = {Rmax_D[0]:.2f}A')   
    print(f'k_H_tot = {ave_kH_of_E_appl:.4e} s^-1')
    print(f'k_D_tot = {ave_kD_of_E_appl:.4e} s^-1')
    print(f'KIE = {ave_kH_of_E_appl/ave_kD_of_E_appl:.2f}')

    kHD_of_E[0][n] = E_appl
    kHD_of_E[1][n] = np.log(ave_kH_of_E_appl)
    kHD_of_E[2][n] = np.log(ave_kD_of_E_appl)

Tafel_params_H, covH = curve_fit(Tafel,kHD_of_E[0],kHD_of_E[1])
Tafel_params_D, covD = curve_fit(Tafel,kHD_of_E[0],kHD_of_E[2])

# Remove the factor of F/RT from the Tafel prefactor
alphaH = Tafel_params_H[0]*RTF
alphaD = Tafel_params_D[0]*RTF

print('The transfer coefficient for protons is: ' + str(alphaH))
print('The transfer coefficient for deuterons is: ' + str(alphaD))

