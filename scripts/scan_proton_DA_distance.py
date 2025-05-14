import numpy as np
from ase.io import read
import os



# xyz files for the reactant and the product
reac_xyz = "reac.xyz"
prod_xyz = "prod.xyz"

# atomic indices (start from 0) of the proton donor and acceptor in the xyz files
Hdonor = 0
Hacceptor = 2

# charge and multiplicity of the reactant and product
# here we assume the reactant is a neutral singlet and the product is a doublet with +1 charge
reac_charge = 0
reac_multiplicity = 1
prod_charge = 1
prod_multiplicity = 2



# change the level of theory or add empirical dispersion/implicit solvent as needed, 
heading = """%chk={state}.chk
%nprocshared=24
%mem=80GB
# B3LYP/6-31+g(d,p) opt=(ModRedundant)

{state}

{charge} {multiplicity}
"""

ending_fix_bond = """{Hdonor}   {Hacceptor}   ={R:.2f}   B
{Hdonor}   {Hacceptor}   F
"""

# read fully optimized reactant and product structure from xyz files
# the order of the atomx in these files must be the same
reac_struct = read(reac_xyz)
prod_struct = read(prod_xyz)

# define the range of proton donor-acceptor distance R
R = np.linspace(2.4,2.8,9)

for Ri in R:
    os.mkdir(f'R{Ri:.2f}A')
    os.mkdir(f'R{Ri:.2f}A/reac_opt/')
    os.mkdir(f'R{Ri:.2f}A/prod_opt/')

    # copy the reactant and product structure for further modification
    tmp_reac = reac_struct.copy()
    tmp_prod = prod_struct.copy()

    # set the distance between proton donor and proton acceptor to Ri
    tmp_reac.set_distance(Hdonor, Hacceptor, Ri, fix=0)
    tmp_prod.set_distance(Hdonor, Hacceptor, Ri, fix=1)

    # read the atomic symbol and positions from the modified structure
    symbols = tmp_reac.symbols
    reac_pos = tmp_reac.get_positions()
    prod_pos = tmp_prod.get_positions()

    outfp_reac_opt = open(f'R{Ri:.2f}A/reac_opt/reac_opt.gjf', "w")
    outfp_prod_opt = open(f'R{Ri:.2f}A/prod_opt/prod_opt.gjf', "w")

    # write the heading in file
    outfp_reac_opt.write(heading.format(state="reactant", charge=reac_charge, multiplicity=reac_multiplicity))
    outfp_prod_opt.write(heading.format(state="product", charge=prod_charge, multiplicity=prod_multiplicity))

    for i in range(len(reac_struct)):
        outfp_reac_opt.write(f"{symbols[i]:2s}       {reac_pos[i,0]: 3.6f}    {reac_pos[i,1]: 3.6f}    {reac_pos[i,2]: 3.6f}\n")
        outfp_prod_opt.write(f"{symbols[i]:2s}       {prod_pos[i,0]: 3.6f}    {prod_pos[i,1]: 3.6f}    {prod_pos[i,2]: 3.6f}\n")

    outfp_reac_opt.write("\n")
    outfp_prod_opt.write("\n")

    outfp_reac_opt.write(ending_fix_bond.format(Hdonor=Hdonor+1, Hacceptor=Hacceptor+1, R=Ri))
    outfp_prod_opt.write(ending_fix_bond.format(Hdonor=Hdonor+1, Hacceptor=Hacceptor+1, R=Ri))
    
    outfp_reac_opt.write("\n")
    outfp_prod_opt.write("\n")

    outfp_reac_opt.close()
    outfp_prod_opt.close()

