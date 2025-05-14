# pyPCET
Python implementation of the nonadiabatic PCET theory. This module is used to calculate the vibronically nonadiabatic PCET rate constant using the following equation:
```math
k_{\rm PCET} = \frac{2\pi}{\hbar}|V_{\rm el}|^2\sum_{\mu,\nu}P_{\mu}|S_{\mu\nu}|^2\frac{1}{\sqrt{4\pi\lambda k_{\rm B}T}}\exp\left(-\frac{(\Delta G^{\rm o}_{\mu\nu}+\lambda)^2}{4\lambda k_{\rm B}T}\right)
```
where $`V_{\rm el}`$ is the electronic coupling between reactant and product electronic states, $`P_{\mu}`$ is the Boltzmann population of the reactant vibronic states, $`S_{\mu\nu}`$ is the overlap integral between the proton vibrational wave functions associated with the reactant and product electronic states, $`\lambda`$ is the reorganization energy, $`\Delta G^{\rm o}_{\mu\nu}`$ is the reaction free energy for vibronic states $\mu$ and $\nu$: 
```math
\Delta G^{\rm o}_{\mu\nu} = \Delta G^{\rm o} + \varepsilon_\nu - \varepsilon_\mu
```
Here $\Delta G^{\rm o}$ is the PCET reaction free energy, $\varepsilon_\mu$ and $\varepsilon_\nu$ are the energies of reactant and product vibronic states $\mu$ and $\nu$, relative to their respective ground vibronic states. 

In general, this rate constant depends on the proton donor-acceptor separation $R$, and the overall PCET rate constant should be calculated as an average over $R$. This is not implemented in this module. However, users can easily write a outer loop to perform such average, as in examples 3 and 4. 

## Installation 
To use the pyPCET module, simply download the code and add it to your `$PYTHONPATH` variable.

## Documentation

### I. Initialization

#### Required Quantities
To calculate the PCET rate constant using the vibronically nonadiabatic PCET theory, we need the following physical quantities: 

1. `ReacProtonPot` (2D array or function): proton potential of the reactant state
2. `ProdProtonPot` (2D array or function): proton potential of the product state

The input of the proton potential can be either a 2D array or a callable function. If these inputs are 2D arrays, a fitting will be performed to create a callable function for subsequent calculations. By default, the proton potentials will be fitted to an 8th-order polynormial. The 2D array should have shape (N, 2), the first row is the proton position in Angstrom, and the second row is the potential energy in eV. If these inputs are functions, they must only take one argument, which is the proton position in Angstrom. The unit of the returned proton potentials should be eV

3. `DeltaG` (float): reaction free energy of the PCET process in eV. This should be the free energy difference between electronic states, i.e., ZPEs should not be included
4. `Lambda` (float): reorganization energy of the PCET reaction in eV
5. `Vel` (float): electronic coupling between reactant and product electronic states in eV, default = 0.0434 eV = 1 kcal/mol


#### Example
The following code set up an `pyPCET` object for rate constant calculation. 
```python
import numpy as np
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
E_Reac = np.array([5.283,4.534,4.061,3.794,3.673,3.646,3.680,3.688,3.634,3.646,3.602,3.513,3.392,3.257,3.138,3.078,3.140,3.413,4.022,5.144])
E_Prod = np.array([4.847,4.134,3.706,3.495,3.452,3.509,3.620,3.801,3.949,4.073,4.157,4.194,4.189,4.157,4.128,4.144,4.270,4.597,5.250,6.411])

ReacProtonPot = fit_poly8(rp, E_Reac) 
ProdProtonPot = fit_poly8(rp, E_Prod)

# set up system
system = pyPCET(ReacProtonPot, ProdProtonPot, dG, Lambda, Vel=Vel)
```


#### Other Parameters
Other parameters that can be modified during the initialization are

6. `NStates` (int): number of proton vibrational states to be calculated, default = 10. One should test the convergence with respect to this quantity. 
7. `NGridPot` (int): number of grid points used for FGH calculation, default = 256
8. `Smooth` (string): method to smooth the proton potential if given as 2Darray, possible choices are 'fit_poly6', 'fit_poly8', 'bspline', default = 'fit_poly6' 

When initializing, The program will automatically determine the ranges of proton position to perform subsequent calculations. Users could fine tune these ranges by parseing additional inputs `rmin`, `rmax`. 


### II. Calculation
#### PCET Rate Constant
In a typical calculation of the vibronically nonadiabatic PCET rate constant, we first need to solve the 1D Schrödinger equations for the proton moving in the proton potentials associated with the reactant and product electronic states. This calculation yields the proton vibrational energy levels and wave functions, which in turn determine the Boltzmann population of the reactant vibronic states, $`P_{\mu}`$, the overlap integral between the proton vibrational wave functions associated with the reactant and product electronic states, $`S_{\mu\nu}`$, as well as the reaction free energy for vibronic states $\mu$ and $\nu$, $`\Delta G^{\rm o}_{\mu\nu}`$. We will calculate $`P_{\mu}`$, $`S_{\mu\nu}`$, and $`\Delta G^{\rm o}_{\mu\nu}`$ for all $`\mu`$ and $`\nu`$ from 0 to `NStates`. These quantities will be fed into the rate constant expression to give the final results. 

All these steps have been integrated in the method `pyPCET.calculate`.  This method takes two parameters, the mass of the particle, which should be set to the mass of the proton or deuterium, and the temperature of the system. It returns the calculated rate constant at the given condition. Follow by the previous example, we can calculate the PCET rate constant and the kinetic isotope effect (KIE) by: 
```python
from pyPCET.units import massH, massD

k_tot_H = system.calculate(massH, T)
k_tot_D = system.calculate(massD, T)
KIE = k_tot_H/k_tot_D
```

#### Analysis of the Result
The calculated proton vibrational energy levels and wave functions can be accessed through
```python
Evib_reactant, wfc_reactant = system.get_reactant_proton_states()
Evib_product, wfc_product = system.get_product_proton_states()
```
$`P_{\mu}`$, $`S_{\mu\nu}`$, and $`\Delta G^{\rm o}_{\mu\nu}`$ can be accessed using the methods
```python
Pu = system.get_reactant_state_distributions()
Suv = system.get_proton_overlap_matrix()
dGuv = system.get_reaction_free_energy_matrix()
```

The activation free energy or free energy barrier associated with the transition between reactant state $\mu$ and product state $\nu$ is defined as
```math
\Delta G^\ddagger_{\mu\nu} = \frac{(\Delta G^{\rm o}_{\mu\nu}+\lambda)^2}{4\lambda}
```
This quantity can be calculated by
```python
dGa_uv = system.get_activation_free_energy_matrix() 
```

The contribution of a given $`(\mu,\nu)`$ pair to the total rate constant is given by
```math
k_{\mu\nu} = \frac{2\pi}{\hbar}|V_{\rm el}|^2 P_{\mu}|S_{\mu\nu}|^2\frac{1}{\sqrt{4\pi\lambda k_{\rm B}T}}\exp\left(-\frac{(\Delta G^{\rm o}_{\mu\nu}+\lambda)^2}{4\lambda k_{\rm B}T}\right)
```

```math
{\rm \%\ Contrib.} = \frac{k_{\mu\nu}}{k_{\rm PCET}}
```

These quantities can be obtained from the code using
```python
kuv = system.get_kinetic_contribution_matrix()
k_tot = system.get_total_rate_constant()
percentage_contribution = kuv/k_tot
```

### III. Probing Nonadiabaticity
The rate constant expression coded in the `pyPCET` module is only applicable for PCET reactions that are both vibronically and electronically nonadiabatic. The module `kappa_coupling` contains methods to probe the nonadiabaticity of a PCET reaction. 

#### Required Quantities
To start an analysis, we need the following quantities:

1. `rp` (1D array): proton coordinate along the proton axis in A
2. `ReacProtonPot` (1D array): Reactant proton potential as a function of rp in eV
3. `ProdProtonPot` (1D array): Product proton potential as a function of rp in eV
4. `Vel` (float or 1D array): electronic coupling as a function of rp or a constant in eV

The input of the proton coodtinate and the proton potentials can only be 1D arrays, and their length must be equal. The electronic coupling can be either an 1D array or a number. If `Vel` is given as a number, we assume it is constant along the proton coordinate. Note that the programm will not automatically interpolate or fit the input data. The users should provide data on a dense grid of the proton coordinate to ensure numerical accuracy in subsequent calculations. The `fit_poly6`, `fit_poly8`, and `bspline` functions in `functions.py` can be used for interpolation or fitting purpose. 

In the following example, we read calculated proton potentials and electronic coupling from a file, and then spline the data using the `bspline` function in `functions.py`, 

```python
from pyPCET import kappa_coupling 
from pyPCET.functions import bspline 
from scipy.interpolate import CubicSpline
from pyPCET.units import kcal2eV

# double well potentials and electronic coupling read from a file
# In this file, all energies are in kcal/mol
rp_read, E_Reac, E_Prod, Vel = np.loadtxt('Y356_Y731_config2_env.dat', unpack=True)
E_Reac *= kcal2eV
E_Prod *= kcal2eV
Vel *= kcal2eV

# Spline the proton potential
ReacProtonPot = bspline(rp_read, E_Reac) 
ProdProtonPot = bspline(rp_read, E_Prod)
Vel_rp = CubicSpline(rp_read, Vel)

# define a finer rp grid
rp = np.linspace(-0.8, 0.8, 256)

# setup the system and perform the calculation
system = kappa_coupling(rp, ReacProtonPot(rp), ProdProtonPot(rp), Vel_rp(rp))
```

#### Calculation and Analysis of the Results
After creating a `kappa_coupling` object, one can perform the nonadiabaticity analysis simply using

```python
from pyPCET.units import massH
system.calculate(massH)
```

The program will calculate the effective proton tunneling time $`\tau_{\rm p}`$, the electronic transition time $`\tau_{\rm e}`$, the adiabaticity parameter $`p = \tau_{\rm p}/\tau_{\rm e}`$, the prefactor $`\kappa`$ which enters the general expression of the vibronic coupling in a semiclassical formalism, the vibronic coupling in the general form $`V_{\rm \mu\nu}^{(\rm sc)}`$, and the vibronic coupling in the nonadiabatic and adiabatic limits $`V_{\rm \mu\nu}^{(\rm nad)}`$ and $`V_{\rm \mu\nu}^{(\rm ad)}`$. Note that we only implemented the analsysis for the ground vibronic states (i.e., $`\mu=0`$, $`\nu=0`$). 

The users can access the calculated quantities via

```python
tau_e, tau_p, p, kappa = system.get_nonadiabaticity_parameters()
V_sc, V_nad, V_ad = system.get_vibronic_couplings()
```

At a given temperature, the reaction is vibronically nonadiabatic if $`V_{\rm \mu\nu}^{(\rm sc)} \ll k_{\rm B}T`$, or vibronically adiabatic if $`V_{\rm \mu\nu}^{(\rm sc)} \gg k_{\rm B}T`$. An electronically nonadiabatic reaction is characterized by $`p \ll 1`$, $`\kappa < 1`$, and $`V_{\rm \mu\nu}^{(\rm sc)} \approx V_{\rm \mu\nu}^{(\rm nad)}`$, whereas an electronically adiabatic reaction is characterized by $`p \gg 1`$, $`\kappa \approx 1`$, and $`V_{\rm \mu\nu}^{(\rm sc)} \approx V_{\rm \mu\nu}^{(\rm ad)}`$. 

### IV. Useful Scripts
In the folder `scripts` we provide several useful scripts that can help generate the input quantities for PCET rate constant calculation or nonadiabaticity analysis. Please refer to the README file in that folder for detailed documentation. 
