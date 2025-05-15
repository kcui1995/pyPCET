import numpy as np
from .FGH_1D import FGH_1D
from .functions import *
from .units import * 
from scipy.special import gamma
from scipy.integrate import simps


class kappa_coupling(object):
    """
    This class is used to probe the nonadiabaticity of a PCET reaction
    using the semiclassical formalism derived by Georgievskii, Y.; Stuchebrukhov, A. J. Chem. Phys. 113, 10438–10450 (2000)
    For a given pair of reactant and product proton potentials, it calculates the proton tunnling time, electron transition time, adiabaticity parameter
    and the vibronic coupling between mu=0 and nu=0 states in the general form and in nondiabatic and adiabatic limits
    !!!NOTE!!! Currently the calculation can only be applied to mu=0 and nu=0 vibronic states
    """
    def __init__(self, rp, ReacProtonPot, ProdProtonPot, Vel, NStates=10):
        """
        Input Quantities
        rp (1D array): proton coordinate along the proton axis in A
        ReacProtonPot (1D array): Reactant proton potential as a function of rp in eV
        ProdProtonPot (1D array): Product proton potential as a function of rp in eV
        Vel (float or 1D array): electronic coupling as a function of rp or a constant in eV
        NStates (int): number of proton vibrational states to be calculated (only the ground state will be used), default = 10
        """
    
        if not (isarray(rp) and isarray(ReacProtonPot) and isarray(ProdProtonPot)):
            raise TypeError("'rp', 'ReacProtonPot', and 'ProdProtonPot' must be 1D arrays.")
        if isarray(Vel):
            pass
        elif isnumber(Vel):
            Vel = np.ones(len(rp))*Vel
        else:
            raise TypeError("'Vel' must be a number or an 1D array.")

        if len(ReacProtonPot) != len(rp) or len(ProdProtonPot) != len(rp) or len(Vel) != len(rp):
            raise ValueError("'rp', 'ReacProtonPot', 'ProdProtonPot', and 'Vel' must have the same dimension. ")

        if np.abs(np.log2(len(rp))-int(np.log2(len(rp)))) > 1e-4:
            raise ValueError("The number of grid points should be some interger power of 2. ")

        self.rp = rp
        self.ReacProtonPot = ReacProtonPot
        self.ProdProtonPot = ProdProtonPot
        self.Vel = Vel
        self.NStates = NStates


    def calc_proton_vibrational_states(self, mass=massH):
        """
        Calculate the proton vibrational states (energies and wave functions)
        corresponding to ReacProtonPot and ProdProtonPot respectively. 
        The FGH_1D code implemented by Maxim Secor is used
        """
        # calculate proton potentials on a grid, self.rp
        # the FGH code requires atomic units, so the units are converted
        rp_in_Bohr = self.rp*A2Bohr
        ngrid = len(rp_in_Bohr)
        sgrid = rp_in_Bohr[-1] - rp_in_Bohr[0]
        dx = sgrid/(ngrid-1)
        E_reac_in_Ha = self.ReacProtonPot*eV2Ha
        E_prod_in_Ha = self.ProdProtonPot*eV2Ha

        # calculate the proton vibrational energies and wave fucntions for the reactant
        eigvals_reac, eigvecs_reac = FGH_1D(ngrid, sgrid, E_reac_in_Ha, mass)
        self.ReacProtonEnergyLevels = eigvals_reac[:self.NStates]*Ha2eV

        # the output wave functions are normalized such that \sum_i \Psi_i^2 = 1 where i is the index of grid points
        # the correct normalization is that \int \Psi^2 dr = 1
        # the normalized wave functions has unit of A^-1/2
        unnormalized_wfcs_reac = np.transpose(eigvecs_reac)[:self.NStates]
        normalized_wfcs_reac = np.array([wfci/np.sqrt(simps(wfci*wfci, self.rp)) for wfci in unnormalized_wfcs_reac])
        self.ReacProtonWaveFunctions = normalized_wfcs_reac

        # calculate the proton vibrational energies and wave fucntions for the product
        eigvals_prod, eigvecs_prod = FGH_1D(ngrid, sgrid, E_prod_in_Ha, mass)
        self.ProdProtonEnergyLevels = eigvals_prod[:self.NStates]*Ha2eV

        unnormalized_wfcs_prod = np.transpose(eigvecs_prod)[:self.NStates]
        normalized_wfcs_prod = np.array([wfci/np.sqrt(simps(wfci*wfci, self.rp)) for wfci in unnormalized_wfcs_prod])
        self.ProdProtonWaveFunctions = normalized_wfcs_prod


    def analyze_proton_potentials(self, mass=massH):
        """
        This function analyze the proton potentials and prepare for a coupling calculation by doing the following:
        Calculate the proton vibrational ZPE of the reactant and the product
        Shift the reactant and product proton potentials so that their ZPE levels are aligned
        Find the crossing point of the shifted potential curves
        Calculate the slopes of the proton potential curves at the crossing point
        """

        self.calc_proton_vibrational_states(mass)
        ZPE_reac = self.ReacProtonEnergyLevels[0]
        ZPE_prod = self.ProdProtonEnergyLevels[0]

        # align the zero-point energy of the reactant and product states in this plot
        if ZPE_prod < ZPE_reac:
            dEr = 0
            dEp = -ZPE_prod + ZPE_reac
        else:
            dEr = ZPE_prod - ZPE_reac
            dEp = 0

        self.ShiftedReacProtonPot = self.ReacProtonPot + dEr
        self.ShiftedProdProtonPot = self.ProdProtonPot + dEp
        self.ShiftedReacProtonEnergyLevels = self.ReacProtonEnergyLevels + dEr
        self.ShiftedProdProtonEnergyLevels = self.ProdProtonEnergyLevels + dEp

        # find the crossing point of the shifted proton potential curves
        deltaE = self.ShiftedReacProtonPot - self.ShiftedProdProtonPot 

        def find_root(y,x):
            for i in range(1,len(x)):
                if y[i]*y[i-1] <= 0:
                    return i, (x[i]+x[i-1])/2

        rp_crossing_index, self.rp_crossing = find_root(deltaE, self.rp)
        self.E_crossing = (self.ShiftedReacProtonPot[rp_crossing_index] + self.ShiftedReacProtonPot[rp_crossing_index-1] + \
                           self.ShiftedProdProtonPot[rp_crossing_index] + self.ShiftedProdProtonPot[rp_crossing_index-1])/4
        self.Vel_crossing = (self.Vel[rp_crossing_index] + self.Vel[rp_crossing_index-1])/2 

        # calculate the slopes at the crossing point using the three-point central difference formula
        # slope in unit eV/A
        self.slope_reac = ( self.ShiftedReacProtonPot[rp_crossing_index] - self.ShiftedReacProtonPot[rp_crossing_index-1]) / (self.rp[rp_crossing_index] - self.rp[rp_crossing_index-1])
        self.slope_prod = ( self.ShiftedProdProtonPot[rp_crossing_index] - self.ShiftedProdProtonPot[rp_crossing_index-1]) / (self.rp[rp_crossing_index] - self.rp[rp_crossing_index-1])


    def calculate(self, mass=massH):
        """
        Calculate the proton tunnling time, electron transition time, adiabaticity parameter
        and the vibronic coupling between mu=0 and nu=0 states in the general form and in nondiabatic and adiabatic limits
        """
        self.analyze_proton_potentials(mass)

        # Calculate tau_e and tau_p 

        # energy of the aligned diabatic reactant and product vibrational levels
        E0 = self.ShiftedReacProtonEnergyLevels[0]
        # proton tunneling velocity in A/s
        vt = np.sqrt(2*(self.E_crossing-E0)*eV2Ha/mass)*Bohr2A/au2s

        self.tau_p = self.Vel_crossing/(np.abs(self.slope_reac-self.slope_prod)*vt)
        self.tau_e = hbar/self.Vel_crossing
        self.p = self.tau_p/self.tau_e
        self.kappa = np.sqrt(2*np.pi*self.p)*np.exp(self.p*np.log(self.p)-self.p)/gamma(self.p+1)


        # Generarte adiabatic proton potential

        self.AdiabaticProtonPotGS = 0.5*(self.ShiftedReacProtonPot + self.ShiftedProdProtonPot - np.sqrt((self.ShiftedProdProtonPot - self.ShiftedReacProtonPot)**2 + 4*self.Vel_crossing**2))
        self.AdiabaticProtonPotES = 0.5*(self.ShiftedReacProtonPot + self.ShiftedProdProtonPot + np.sqrt((self.ShiftedProdProtonPot - self.ShiftedReacProtonPot)**2 + 4*self.Vel_crossing**2))

        # Calculate the V_ad, V_nad, and V_sc 

        # Calculate the tunneling spliting on the adiabatic proton potential 
        # run FGH on the adiabatic proton potential
        # convert all units to a.u.
        rp_in_Bohr = self.rp*A2Bohr
        Eg_au = self.AdiabaticProtonPotGS*eV2Ha
        ngrid = len(rp_in_Bohr)
        sgrid = rp_in_Bohr[-1] - rp_in_Bohr[0]
        dx = sgrid/(ngrid-1)

        # calculate the proton vibrational energies and wave fucntions for the ground adiabatic state
        eigvals, eigvecs = FGH_1D(ngrid, sgrid, Eg_au, mass)
        self.AdiabaticGSProtonEnergyLevels = eigvals[:self.NStates]
        
        unnormalized_wfcs_adia = np.transpose(eigvecs)[:self.NStates]
        normalized_wfcs_adia = np.array([wfci/np.sqrt(simps(wfci*wfci, self.rp)) for wfci in unnormalized_wfcs_adia])
        self.AdiabaticGSProtonWaveFunctions = normalized_wfcs_adia

        tunneling_splitting = (eigvals[1]-eigvals[0])*Ha2eV
        self.V_ad = 0.5*tunneling_splitting
        
        S00 = simps(self.ReacProtonWaveFunctions[0]*self.ProdProtonWaveFunctions[0], self.rp)
        self.V_nad = self.Vel_crossing*S00
        self.V_sc = self.kappa*self.V_ad


    def get_reactant_proton_states(self):
        """
        returns shifted proton potential, shifted proton vibrational energy levels, and wave functions of the reactant state
        """
        return self.ShiftedReacProtonPot, self.ShiftedReacProtonEnergyLevels, self.ReacProtonWaveFunctions

    def get_product_proton_states(self):
        """
        returns shifted proton potential, shifted proton vibrational energy levels, and wave functions of the product state
        """
        return self.ShiftedProdProtonPot, self.ShiftedProdProtonEnergyLevels, self.ProdProtonWaveFunctions

    def get_adiabatic_proton_potentials(self):
        """
        returns the ground and excited state adiabatic proton potentials
        """
        return self.AdiabaticProtonPotGS, self.AdiabaticProtonPotES

    def get_ground_adiabatic_proton_states(self):
        """
        returns proton vibrational energy levels and wave functions associated with the ground adiabatic electronic state
        """
        return self.AdiabaticGSProtonEnergyLevels, self.AdiabaticGSProtonWaveFunctions

    def get_nonadiabaticity_parameters(self):
        """
        returns the parameters tau_e, tau_p, p, and kappa
        """
        return self.tau_e, self.tau_p, self.p, self.kappa

    def get_vibronic_couplings(self):
        """
        return the vibronic coupling between mu=0 and nu=0 states in the semi-classical formulism and in nonadiabatic and adiabatic limits
        """
        return self.V_sc, self.V_nad, self.V_ad

