# Some Useful Scipts

## Prepare Gaussian Input Files for Scaning Proton Donor-Acceptor Distance
The script `scan_proton_DA_distance.py` can be used to generate Gaussian input files for constrained optimization of the reactant and product structure with a fixed proton donor-acceptor distance $`R`$. It requires the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/index.html) package, which is very useful for visualize and manipulate structures. 

To use this code, you should have the fully optimized reactant and product structure in hand, the atomic order in these two structure files must be the same. More detailed instruction are annotated in the script. 

## Align and Average Structure
The script `align_average_midpoint_minRMSD.py` is used to align the reactant and product structures of a PCET reaction and calculate the averaged geometry. It first overlays the reactant and product structures so the proton donor and acceptor are superimposed and then minimizes the RMSD of the remaining nuclei while maintaining this constraint. Then the average structure is generated by averaging all the Cartesian coordinates of the nuclei. 

The following arguments are needed:

"-r" or "--reactant-xyz": select xyz-file with the reactant structure

"-p" or "--product-xyz": select xyz-file with the reactant structure

"--r-donor": proton donor atom index (starts from 1) in the reactant structure

"--r-acceptor": proton acceptor atom index (starts from 1) in the reactant structure

"--p-donor": proton donor atom index (starts from 1) in the product structure

"--p-acceptor": proton acceptor atom index (starts from 1) in the product structure

For example, if the xyz files for the reactant and product are “reac.xyz“ and “prod.xyz“, respectively, and the indices of the proton donor and acceptor are 1 and 2 in both files, respectively, then you can run the script by:

```bash
>>> python3 align_average_midpoint_minRMSD.py -r reac.xyz -p prod.xyz --r-donor 1 --r-acceptor 2 --p-donor 1 --p-acceptor 2 
```

## Optimize Proton Position on the Averaged Structure 
From the averaged structure, you need to optimize the proton on the donor for the reactant, and then optimize the proton on the acceptor for the product, while all other nuclei remain fixed to the average structure. Currently, we don't have a script for this task.

>[!NOTE]
> Make sure to set `Nosymm` in the Gaussian input file when performing these constrained optimizations, otherwise Gaussian will automatically rotate the molecule, and the subsequent script will not work. 

## Prepare Gaussian Input Files for Scaning Proton Potential at each R
After optimizing the proton positions on the averaged structures, the script `scan_proton_coord.py` can be used to generate the proton axis that passes through the two optimized proton positions, and prepare Gaussian input files for single point calculations with the proton moving along the axis. You also need to install [ASE](https://wiki.fysik.dtu.dk/ase/index.html) to use this script. 

The detailed instruction of this code is annotated in the script. Note that you will need to run this script separately for the reactant and product states. 

>[!NOTE]
> This script is designed for the case where the reactant and product diabatic electronic states can be obtained by simply changing the charge of the overall system. For example, in homogeneous electrochemical PCET or photoexcited PCET with an external photoreceptor, the charge of the molecule will be different for the reactant and product.


## Calculate Effective Force Constant for Proton Donor-Acceptor Motion using Gaussian
The script `calc_keff.py` is used to calculate the effective force constant for the proton donor-acceptor motion, which in turn will be used to calculate the proton donor-acceptor distance distribution function $`P(R)`$. 

To use this code, you will need a xyz file for the molecular structure, and the log file for a Gaussian frequency job. You also need to install the [ASE](https://wiki.fysik.dtu.dk/ase/index.html). 

>[!NOTE]
> Make sure to set `#P` and `freq=HPmodes` in your Gaussian calculation. The script is looking for some specific text in the log file and will not work properly without these settings. 

Options:

“--xyz“: xyz file of the molecule

“--log“: log file of the Gaussian frequency job

“-D“: atomic index (start from 0) of the proton donor

“-A“: atomic index (start from 0) of the proton acceptor

For example, if the xyz for the molecule is “mol.xyz“, the log file of the Gaussian frequency job is “freq.log“, and the atomic indices of the proton donor and acceptor are 0 and 1, respectively, then you can run the script as:

```bash
>>> python3 calc_keff.py --xyz mol.xyz --log freq.log -D 0 -A 1
```

It will print the effective force constant for the proton donor-acceptor mode in a.u., the effective reduced mass for this mass in amu, and the effective frequency in cm<sup>-1</sup>. 
