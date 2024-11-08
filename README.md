# pyPCET
Python implementation of the nonadiabatic PCET theory. This module is used to calculate the vibronically nonadiabatic PCET rate constant using the following equation:
```math
k_{\rm PCET} = \frac{2\pi}{\hbar}|V_{\rm el}|^2\sum_{\mu,\nu}P_{\mu}|S_{\mu\nu}|^2\frac{1}{\sqrt{4\pi\lambda k_{\rm B}T}}\exp\left(-\frac{(\Delta G^{\rm o}_{\mu\nu}+\lambda)^2}{4\lambda k_{\rm B}T}\right)
```
where $`V_{\rm el}`$ is the electronic coupling between reactant and product electronic states, $`P_{\mu}`$ is the Boltzmann population of the reactant vibronic states, $`S_{\mu\nu}`$ is the overlap integral between the proton vibrational wave functions associated with the reactant and product electronic states, $`\lambda`$ is the reorganization energy, $`\Delta G^{\rm o}_{\mu\nu}`$ is the reaction free energy for vibronic states $\mu$ and $\nu$: 
```math
\Delta G^{\rm o}_{\mu\nu} = \Delta G^{\rm o} + \varepsilon_\nu - \varepsilon_\mu
```
Here $\Delta G^{\rm o}$ is the PCET reaction free energy, $\varepsilon_\mu$ and $\varepsilon_\nu$ are the energies of reactant and product vibronic states $\mu$ and $\nu$. 

In general, this rate constant depends on the proton donor-acceptor separation $R$, and the overall PCET rate constant should be calculated as an average over $R$. This is not implemented in this module. However, users can easily write their own code to perform such average. 

## Installation 
To use the kinetic model, simply download the code and add it to your `$PYTHONPATH` variable.

## Documentation

### Initialization

#### Required Quantities
To calculate the PCET rate constant using the vibronically nonadiabatic PCET theory, we need the following physical quantities: 

1. `ReacProtonPot` (2D array or function): proton potential of the reactant state
2. `ProdProtonPot` (2D array or function): proton potential of the product state

The input of proton potentials can be either a 2D array or a callable function. If these inputs are 2D array, a fitting will be performed to create a callable function for subsequent calculations. By default, the proton potentials will be fitted to an 8th-order polynormial. The 2D array should have shape (N, 2), the first row is the proton position in Angstrom, and the second row is the potential energy in eV. If these inputs are functions, they must only take one argument, which is the proton position in Angstrom. The unit of the returned proton potentials should be eV

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
E_LES = np.array([5.283,4.534,4.061,3.794,3.673,3.646,3.680,3.688,3.634,3.646,3.602,3.513,3.392,3.257,3.138,3.078,3.140,3.413,4.022,5.144])
E_LEPT = np.array([4.847,4.134,3.706,3.495,3.452,3.509,3.620,3.801,3.949,4.073,4.157,4.194,4.189,4.157,4.128,4.144,4.270,4.597,5.250,6.411])

ReacProtonPot = fit_poly8(rp, E_LES) 
ProdProtonPot = fit_poly8(rp, E_LEPT)

# set up system
system = pyPCET(ReacProtonPot, ProdProtonPot, dG, Lambda, Vel=Vel)
```


#### Other Parameters
Other parameters that can be modified during the initialization are

6. `NStates` (int): number of proton vibrational states to be calculated, default = 10. One should test the convergence with respect to this quantity. 
7. `NGridPot` (int): number of grid points used for FGH calculation, default = 128
8. `NGridLineshape` (int): number of grip points used to calculate spectral overlap integral, defaut = 500
9. `FitOrder` (int): order of polynomial to fit the proton potential, default = 8, This is only useful when some of the proton potentials, `ReacProtonPot` and `ProdProtonPot`, are provided as 2D arrays. Another possible value for this is 6.

When initializing, The program will automatically determine the ranges of proton position to perform subsequent calculations. Users could fine tune these ranges by parseing additional inputs `rmin`, `rmax`. 


### Calculation
#### PCET Rate Constant
In a typical calculation of the vibronically nonadiabatic PCET rate constant, we first need to solve the 1D Schrödinger equations for the proton moving in the proton potentials associated with the reactant and product electronic states. This calculation yields the proton vibrational energy levels and wave functions, which in turn determine the Boltzmann population of the reactant vibronic states, $`P_{\mu}`$, the overlap integral between the proton vibrational wave functions associated with the reactant and product electronic states, $`S_{\mu\nu}`$, as well as the reaction free energy for vibronic states $\mu$ and $\nu$, $`\Delta G^{\rm o}_{\mu\nu}`$. We will calculate $`P_{\mu}`$, $`S_{\mu\nu}`$, and $`\Delta G^{\rm o}_{\mu\nu}`$ for all $`\mu`$ and $`\nu`$ from 0 to `NStates`. These quantities will be fed into the rate constant expression to give the final results. 

All these steps have been integrated in the method `pyPCET.calculate`.  This method takes two parameters, the mass of the particle, which should be set to the mass of the proton or deuterium, and the temperature of the system. It returns the calculated rate constant at the given condition. Follow by the previous example, we can calculate the PCET rate constant and the kinetic isotope effect (KIE) by: 
```python
from pyPCEnT.units import massH, massD

k_tot_H = system.calculate(massH, T)
k_tot_D = system.calculate(massD, T)
KIE = k_tot_H/k_tot_D
```

#### Analyze the Result
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
`Pu` is a 1D array with a length of `NStates`, where as `Suv` and `dGuv` are `NStates`$\times$`NStates` matrices. 

The activation free energy or free energy barrier associated with the transition between reactant state $\mu$ and product state $\nu$ is defined as
```math
\Delta G^\ddagger_{\mu\nu} = -\frac{(\Delta G^{\rm o}_{\mu\nu}+\lambda)^2}{4\lambda}
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
