import numpy as np

# define several unit convertion factors and physical constants

kB = 1.3806e-23/1.6022e-19    # Boltzmann constant in eV/K
h = 6.6261e-34/1.6022e-19    # Planck's constant in eV*s
hbar = h/2/np.pi    # hbar in eV*s
c = 2.99792458e8/1e-10   # speed of light in Angstrom/s
massH = 1836.15   # in atomic unit (m_e)
massD = 3671.48   # in atomic unit (m_e)

Ha2eV = 27.2114
Ha2kcal = 27.2114*96.485/4.184
kcal2Ha = 1/Ha2kcal
eV2Ha = 1/Ha2eV
eV2kcal = eV2Ha*Ha2kcal 
kcal2eV = 1/eV2kcal

A2Bohr = 1/0.529177
A2nm = 0.1
A2cm = 1e-8
Bohr2A = 1/A2Bohr
cm2A = 1/A2cm
nm2A = 1/A2nm

wn2eV = h*c*A2cm # convert wave number in cm^-1 to eV
eV2wn = 1/wn2eV

Da2me = 1822.888486209
me2Da = 1/Da2me

au2s = 2.418884326e-17 #covert atomic unit of time to s
