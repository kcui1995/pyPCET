import numpy as np
from ase.io import read, write
from optparse import OptionParser


def read_Gaussian_freq_job(xyzfile, logfile):
    atoms = read(xyzfile)
    natoms = len(atoms)
    # number of vibrational degrees of freedom, assume a nonlinear molecule
    nDOFvib = 3*natoms-6

    # when HPmodes is set in Gaussian input file, the normal modes are printed 5 modes per group
    ngroup = int(nDOFvib/5)
    nresidue = nDOFvib%5
    if nresidue != 0:
        ngroup += 1 

    logfp = open(logfile)
    lines = logfp.readlines()
    index_optimized_struct = 0
    index_freq_output = 0
    for i,line in enumerate(lines):
        if 'Standard orientation' in line:
            index_optimized_struct = i
        if line.startswith(' Harmonic frequencies'):
            # when HPmodes is set, the high precision normal modes are printed first 
            # we only need the line index that corresponds to the high precision outputs
            index_freq_output = i
            break

    #print(index_optimized_struct, index_freq_output)

    # read optimized structure
    # update the coordinate in the Atoms object if they are not the same as the optimized structure
    # this is because Gaussian can rotate the geometry during the calculation
    new_poses = np.zeros([natoms,3])
    i = index_optimized_struct + 5
    for j in range(natoms):
        new_poses[j] = [float(dat) for dat in lines[i+j].split()[3:]]
    atoms.set_positions(new_poses)

    write('optimized_geometry.xyz', atoms)

    freqs = np.zeros(nDOFvib)
    reduced_masses = np.zeros(nDOFvib)
    force_constants = np.zeros(nDOFvib)
    normal_modes = np.zeros([nDOFvib,3*natoms])

    # start reading normal modes
    i = index_freq_output + 4
    for ig in range(ngroup):
        mode_indices = [int(dat)-1 for dat in lines[i].split()]
        freqs[mode_indices] = [float(dat) for dat in lines[i+2].split()[2:]]
        reduced_masses[mode_indices] = [float(dat) for dat in lines[i+3].split()[3:]]
        force_constants[mode_indices] = [float(dat) for dat in lines[i+4].split()[3:]]

        for j in range(3*natoms):
            normal_modes[mode_indices, j] = [float(dat) for dat in lines[i+7+j].split()[3:]]

        i += (7+natoms*3) 

    logfp.close()

    # In Gaussian output, the frequencies are in cm-1, reduced masses in amu
    # force constants in mDyne/A
    # The printed normal modes by Gaussian are the Cartesian displacements, not the mass-weighted Cartesian displacements 

    return atoms, freqs, reduced_masses, force_constants, normal_modes


def calc_keff(atoms, donor_index, acceptor_index, reduced_masses, force_constants, normal_modes):
    # input force constants in mDyne/A, which is unit used in Gassuain outputs

    # convert the unit of force constant from mDyne/A to au
    # 1 au = 8.2387235038 mDyne, 1 Bohr = 0.529177 A
    scale = 8.2387235038/0.529177
    force_constants /= scale

    poses = atoms.get_positions()
    masses = atoms.get_masses()

    #M = np.array([masses,masses,masses])
    #diag_sqrt_mass = np.diag(np.sqrt(M.transpose().reshape(3*len(atoms))))
    #normal_modes = np.matmul(normal_modes, diag_sqrt_mass)

    # calculate the unit vector connect the proton donor and acceptor
    eDA = poses[acceptor_index] - poses[donor_index]
    eDA /= np.linalg.norm(eDA)

    nDOFvib = len(force_constants)
    weights = np.zeros(nDOFvib)

    for imode in range(nDOFvib):
        lAi = normal_modes[imode, acceptor_index*3:acceptor_index*3+3]
        lDi = normal_modes[imode, donor_index*3:donor_index*3+3]
        weights[imode] = np.inner(eDA, lAi - lDi) 
        #weights[imode] = np.inner(eDA, lAi/np.sqrt(masses[acceptor_index]) - lDi/np.sqrt(masses[donor_index]))
 
    effective_force_constant = 1/(np.sum(weights*weights/force_constants))
    effective_reduced_mass = 1/(np.sum(weights*weights/reduced_masses))

    # calculate effective proton DA vibrational frequency in cm-1
    Da2au = 1822.888486209
    au2s = 2.418884326e-17
    c = 29979245800	# in cm/s
    effective_frequency = np.sqrt(effective_force_constant/effective_reduced_mass/Da2au)/au2s/c/2/np.pi

    return effective_force_constant, effective_reduced_mass, effective_frequency


parser = OptionParser()

parser.add_option("--xyz", type="string", dest="xyzfile", help="select an xyz file for the molecule")
parser.add_option("--log", type="string", dest="logfile", help="select the log file of a Gaussian frequency calculation")
parser.add_option("-D", type="int", dest="donor_index", help="atomic index (start from 0) of the proton donor")
parser.add_option("-A", type="int", dest="acceptor_index", help="atomic index (start from 0) of the proton acceptor")

(options, args) = parser.parse_args()

xyzfile = options.xyzfile
logfile = options.logfile
donor_index = options.donor_index 
acceptor_index = options.acceptor_index

atoms, freq, reduced_masses, force_constants, normal_modes = read_Gaussian_freq_job(xyzfile, logfile)
effective_force_constant, effective_reduced_mass, effective_frequency = calc_keff(atoms, donor_index, acceptor_index, reduced_masses, force_constants, normal_modes)


print(f"Effective force constant in a.u.: {effective_force_constant:.4f}")
print(f"Effective reduced mass in amu: {effective_reduced_mass:.3f}")
print(f"Effective frequency in cm-1: {effective_frequency: .2f}")
