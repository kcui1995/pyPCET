import numpy as np
from scipy.constants import N_A, elementary_charge
from scipy.optimize import fsolve
from .units import cm2A, A2Bohr, Bohr2A, eV2Ha, Ha2eV, kB
from .functions import isnumber, isarray


def EDL_model(EvsSHE, dIHL, dOHL, eps_IHL, eps_st, eps_op, dipole, rho_solvent, m_solvent, c_ions, C_EDL, PZFCvsSHE, T=298.15, eta_Kirkwood=1, g_Kirkwood=2.4, print=True):
    """
    Model the electric double layer (EDL) near an electrode,
    This function will return a callable function describing the potential drop in the EDL at the given potential
    The returned function takes one parameter: distance from the electrode R (in A), and returns the potential drop in V

    Input quantities:
    EvsSHE (float): applied potential vs. SHE (in V)
    dIHL (float): thickness of the inner Helmholtz layer (in angstrom)
    dOHL (float): thickness of the outer Helmholtz layer (in angstrom)
    eps_IHL (float): relative dielectric constant of the inner Helmholtz layer (2.7 for water)
    eps_st (float): relative static dielectric constant for bulk solvent (78.0 for water)
    eps_op (float): relative optical dielectric constant for bulk solvent (1.78 for water)
    dipole (float or string): dipole moment of the solvent molecule (in Debye), if dipole == 'calculate', 
                              calculate dipole moment of the solvent molecule using Onsager-Kirkwood-Frohlich equation
    rho_solvent (float): density of the solvent (in g/cm^3) (0.997)
    m_solvent (float): molar mass of the solvent (in g/mol)
    c_ions (float): molar concentration of electrolyte ions (in mol/L)
    C_EDL (float): capacitance of the EDL (in microFarad/cm^2)
    PZFCvsSHE (float): potential of zero free charge of the electrode vs. SHE (in V)
    T (float): temperature in K, default = 298.15
    """
    
    # convert the potential to vs. PZFC
    EvsPZFC = EvsSHE - PZFCvsSHE
    if print:
        print(f"E vs. SHE = {EvsSHE:.2f} V")
    
    # calculate surface charge density sigma_M on the electrode from the capacitance and potential
    # C_EDL given in microFarad / cm^2
    # calculated sigma_M is in atomic units (elementary charge/Bohr^-2)
    # 1e6 converts Coulomb to microCoulomb
    sigma_M = (C_EDL * EvsPZFC) / (elementary_charge*1e6 * (cm2A*A2Bohr)**2)
    
    if print:
        print(f"sigma_M = {sigma_M:.6e} a.u.")
    
    # number density of the solvent in atomic unit calculated from the molar mass and density of the solvent
    # in atomic units (Bohr^-3)
    n_solvent_au = rho_solvent/m_solvent*N_A / (cm2A*A2Bohr)**3

    # solvent dipole moment in atomic units 
    if dipole == 'calculate':
        if eta_Kirkwood == 1:
            eta_Kirkwood = 1
        else:
            eta_Kirkwood = 2*(2*eps_st+eps_op)/(3*g_Kirkwood*eps_st)
        dipole_au = 3/(2+eps_op) * np.sqrt(3*kB*T*eV2Ha*(eps_st-eps_op)*eta_Kirkwood/(8*np.pi*n_solvent_au))
        if print:
            print(f"dipole = {dipole_au:.6f} a.u.")
    elif isnumber(dipole):
        Debye2au = 1/2.541746473
        dipole_au = dipole*Debye2au 
    else:
        raise ValueError("'dipole' must be a number (in Debye) or 'calculate'. ")
    
    # number density of electrolyte ions in a.u. (Bohr^-3)
    # 1000 converts L to m^2, 1e10 converts m to angstrom
    n_ions_au = (c_ions * N_A * 1000) / (1e10*A2Bohr)**3
    dIHL_Bohr = dIHL * A2Bohr
    dOHL_Bohr = dOHL * A2Bohr
    
    # calculate the electrostatic potential at the OHP from surface charge density
    # In atomic units, the vacuum permitivity = 1/4pi
    # phi_OHP in V and phi_OHP_au in atomic units
    phi_OHP_au = 2*kB*T*eV2Ha * np.arcsinh(sigma_M / np.sqrt((2*kB*T*eV2Ha * eps_st * n_ions_au) / np.pi))
    phi_OHP = phi_OHP_au * Ha2eV
    
    if print:
        print(f"phi_OHP = {phi_OHP:.6f} V")
    
    def Langevin(u):
        return 1/np.tanh(u) - 1/u
    
    # relative dielectric constant of the outer Helmholtz layer as a function of the electric field in OHL
    def eps_OHL(E_OHL_au):
        # E_OHL_au in atomic units
        return eps_op + (4*np.pi*(2+eps_op)) / (3*E_OHL_au) * n_solvent_au * dipole_au * Langevin(((2+eps_op) * dipole_au * E_OHL_au) / (2*kB*T*eV2Ha))
    
    def equation(E_OHL_au):
        return E_OHL_au * ((dIHL_Bohr * eps_OHL(E_OHL_au) / eps_IHL) + dOHL_Bohr) - EvsPZFC*eV2Ha + phi_OHP_au
    
    # Solve the equation for E_OHL_au
    E_solution_au = fsolve(equation, x0=0.1)
    
    #print(f"E_solution_au: {E_solution_au}")
    
    # electric field in IHL and OHL
    E_OHL_au = E_solution_au[np.argsort(np.abs(E_solution_au))[-1]]
    E_OHL = E_OHL_au * Ha2eV / Bohr2A
    
    E_IHL_au = E_OHL_au * eps_OHL(E_OHL_au) / eps_IHL
    E_IHL = E_IHL_au * Ha2eV / Bohr2A
    
    if print:
        print(f"eps_OHL = {eps_OHL(E_OHL_au):.6f}")
        print(f"E_OHL = {E_OHL:.6f} V/A")
        print(f"E_IHL = {E_IHL:.6f} V/A")
        print()
    
    # screen constant in the diffuse layer in A^-1
    kappa = np.sqrt((8*np.pi*n_ions_au) / (eps_st*kB*T*eV2Ha)) / Bohr2A

    def EDL_potential_drop(R):
        """
        return this function
        It takes the distance from the electrode R (in A) as the parameter and returns the potential drop in V
        """
        if isnumber(R):
            if R <= dIHL:
                return EvsPZFC - R * E_IHL
            elif (dIHL < R) and (R <= dIHL + dOHL):
                return EvsPZFC - dIHL * E_IHL - (R - dIHL) * E_OHL
            else:
                return 4*kB*T * np.arctanh(np.tanh(phi_OHP / (4*kB*T)) * np.exp(-kappa * (R - dIHL - dOHL)))
        elif isarray(R):
            result = np.zeros(len(R))
            for i,Ri in enumerate(R):
                if Ri <= dIHL:
                    result[i] = EvsPZFC - Ri * E_IHL
                elif (dIHL < Ri) and (Ri <= dIHL + dOHL):
                    result[i] = EvsPZFC - dIHL * E_IHL - (Ri - dIHL) * E_OHL
                else:
                    result[i] = 4*kB*T * np.arctanh(np.tanh(phi_OHP / (4*kB*T)) * np.exp(-kappa * (Ri - dIHL - dOHL)))
            return result
        else:
            return none


    return EDL_potential_drop


def Fermi_distribution(E, E_Fermi=0, T=298.15):
    """
    Fermi distribution of electrode energy levels E at a given temperature
    """
    # E_Fermi is the Fermi level, which is often used as the reference for zero energy
    return 1/(np.exp((E-E_Fermi)/kB/T) + 1)

