import numpy as np
from ase.io import read, write
import os

struct1 = read('averaged_optH_reac.xyz')
struct2 = read('averaged_optH_prod.xyz')

# atomic index (start with 0) of the proton in these xyz files
proton_index = 1

# define the state (reactant or product), charge and multiplicity
state = "reactant"
charge = 0
multiplicity = 1

pos1 = struct1.get_positions()
pos2 = struct2.get_positions()

# calculating the proton axis, which passes through the two optimized proton positions in the xyz files
rPT = pos2[proton_index] - pos1[proton_index]
dPT = np.linalg.norm(rPT)
nPT = rPT/dPT
center = 0.5*(pos1[proton_index] + pos2[proton_index]) 


# change the level of theory or add empirical dispersion/implicit solvent as needed, 
heading = """%chk={state}.chk
%nprocshared=24
%mem=60GB
# Nosymm B3LYP/6-31+g(d,p)

{state} proton potential

{charge} {multiplicity}
"""

# number of grip points
N = 20
# To effectively generate the proton potential, we need to move the proton very close to the donor or the acceptor
# the default scan range is 1.7 times the distance between the equilibrium proton positions on its donor and acceptor
# However, for very small R values, this is not sufficient. The scan range for this case is set to be 1 A
dp_center = np.linspace(np.min([-0.5,-1.7*dPT/2]),np.max([0.5,1.7*dPT/2]),N)
print(dp_center)

for i in range(N):
    os.mkdir('%02d'%i)
    outfp = open(f'%02d/{state}_sp.gjf'%i, 'w')
    outfp.write(heading.format(state=state, charge=charge, multiplicity=multiplicity))

    rp = dp_center[i]*nPT+center

    struct_tmp = struct1.copy()
    pos_tmp = struct_tmp.get_positions()
    pos_tmp[proton_index] = rp
    struct_tmp.set_positions(pos_tmp)

    for atom in struct_tmp:
        x,y,z = pos_tmp[atom.index]
        outfp.write('%s      %.8f   %.8f   %.8f\n'%(atom.symbol, x, y, z))
    outfp.write('\n')
    outfp.close()
