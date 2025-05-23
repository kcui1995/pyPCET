import sys
import re
import math
import os
import numpy
from optparse import OptionParser
from scipy.optimize import minimize_scalar

# Conversions
radians = math.pi/180

parser = OptionParser()

parser.add_option("-r", "--reactant-xyz", type="string",dest="rea",
                  help="select xyz-file with the reactant structure")
parser.add_option("-p", "--product-xyz", type="string",dest="pro",
                  help="select xyz-file with the product structure")

parser.add_option("-o", "--output-xyz", type="string",dest="output_name",
                  default="AVERAGE_STRUCTURE.xyz",
                  help="select the name of xyz-file with the average structure (default: %default)")

parser.add_option("--r-donor", type="int",dest="rdo",
                  help="donor atom index in the reactant structure")
parser.add_option("--r-acceptor", type="int",dest="rac",
                  help="acceptor atom index in the reactant structure")

parser.add_option("--p-donor", type="int",dest="pdo",
                  help="donor atom index in the product structure")
parser.add_option("--p-acceptor", type="int",dest="pac",
                  help="acceptor atom index in the product structure")

(options, args) = parser.parse_args()

# Donor [do] and Acceptor [ac] atom numbers

rdo = options.rdo
rac = options.rac

print ("\nXYZ file with the reactant structure: " + options.rea)
print   ("XYZ file with the product  structure: " + options.pro)

try:
   print("\nDonor index in the reactant structure: " + str(rdo))
except:
   rdo = int(input('\nEnter donor atom index in the reactant structure: '))

try:
   print("Acceptor index in the reactant structure: " + str(rac))
except:
   rac = int(input('Enter acceptor atom index in the reactant structure: '))

pdo = options.pdo
pac = options.pac

try:
   print ("\nDonor index in the product structure: " + str(pdo))
except:
   pdo = int(input('\nEnter donor atom index in the product structure: '))

try:
   print ("Acceptor index in the product structure: " + str(pac))
except:
   pac = int(input('Enter acceptor atom index in the product structure: '))

print ("\nOutput file with the averaged structure: " + options.output_name + "\n")

# (index starts from 0 in python arrays)
rdo = rdo - 1
rac = rac - 1
pdo = pdo - 1
pac = pac - 1

realist = options.rea.split(".")
realist_length = len(realist)
realist_range = realist_length - 1
reabasename = ""
for i in range(realist_range):
   reabasename += realist[i]
reanew = reabasename + "_DA_along_Z." + realist[realist_length-1]

prolist = options.pro.split(".")
prolist_length = len(prolist)
prolist_range = prolist_length - 1
probasename = ""
for i in range(prolist_range):
   probasename += prolist[i]
proali = probasename + "_DA_along_Z." + prolist[prolist_length-1]
prorot = probasename + "_DA_along_Z_aligned." + prolist[prolist_length-1]


### REACTANT System

frea = open(options.rea)
lines = frea.readlines()
natom = int(lines[0])
posr = numpy.zeros((3,natom))
catom = []
for i in range(natom):
   xyz=lines[i+2].split()
   catom.append(xyz[0])
   posr[0][i]=float(xyz[1])
   posr[1][i]=float(xyz[2])
   posr[2][i]=float(xyz[3])
frea.close()

posr_new = numpy.zeros((3,natom))
vecar = numpy.zeros((3,1))

hr = [0.0,0.0,0.0]
for i in range(3):
   hr[i] = posr[i][rdo]

for j in range(natom):
   for i in range(3):
      posr[i][j] = posr[i][j] - hr[i]

rmagr = math.sqrt((posr[0][rdo]-posr[0][rac])**2 + (posr[1][rdo]-posr[1][rac])**2 + (posr[2][rdo]-posr[2][rac])**2)
rda_distance = rmagr

vecr = [0.0,0.0,0.0]
for i in range(3):
   vecr[i] = (posr[i][rdo]-posr[i][rac])/rmagr

# angle of projection with x-axis
costhrx = vecr[0]/math.sqrt(vecr[0]**2 + vecr[1]**2)
sinthrx = vecr[1]/math.sqrt(vecr[0]**2 + vecr[1]**2)

# angle with z axis
costhrz = vecr[2]
sinthrz = math.sin(math.acos(costhrz))

rrx = numpy.zeros((3,3))
rrz = numpy.zeros((3,3))

rrx[0][0] = costhrx
rrx[1][1] = costhrx
rrx[0][1] = sinthrx
rrx[1][0] = -sinthrx
rrx[2][2] = 1.0e0

rrz[0][0] = costhrz
rrz[2][2] = costhrz
rrz[0][2] = -sinthrz
rrz[2][0] = sinthrz
rrz[1][1] = 1.0e0

posr_new = numpy.dot(numpy.dot(rrz,rrx),posr)

# translate along the Z-axis so that a midpoint between donor and acceptor is at the origin
shift = posr_new[2][rac]/2
for i in range(natom):
   posr_new[2][i] = posr_new[2][i] - shift

fr = open(reanew,"w")
fr.write(str(natom)+"\n")
fr.write("Reactant with D and A atoms along the Z-axis and midpoint at (0,0,0): DA distance %10.6f \u212B\n" % rda_distance)
for i in range(natom):
   s = "%-2s %15.8f %15.8f %15.8f\n" % (catom[i],posr_new[0][i],posr_new[1][i],posr_new[2][i])
   fr.write(s)
fr.close()


### PRODUCT System
fpro = open(options.pro)
lines = fpro.readlines()
natom = int(lines[0])
posr = numpy.zeros((3,natom))
catom = []
for i in range(natom):
   xyz = lines[i+2].split()
   catom.append(xyz[0])
   posr[0][i] = float(xyz[1])
   posr[1][i] = float(xyz[2])
   posr[2][i] = float(xyz[3])
fpro.close()

posr_new = numpy.zeros((3,natom))
vecar = numpy.zeros((3,1))

hr = [0.0,0.0,0.0]
for i in range(3):
   hr[i] = posr[i][pdo]

for j in range(natom):
   for i in range(3):
      posr[i][j] = posr[i][j] - hr[i]

rmagr = math.sqrt((posr[0][pdo]-posr[0][pac])**2 + (posr[1][pdo]-posr[1][pac])**2 + (posr[2][pdo]-posr[2][pac])**2)
pda_distance = rmagr

vecr = [0.0,0.0,0.0]
for i in range(3):
   vecr[i] = (posr[i][pdo] - posr[i][pac])/rmagr

# angle of projection with x-axis
costhrx = vecr[0]/math.sqrt(vecr[0]**2 + vecr[1]**2)
sinthrx = vecr[1]/math.sqrt(vecr[0]**2 + vecr[1]**2)

# angle with z axis
costhrz = vecr[2]
sinthrz = math.sin(math.acos(costhrz))

rrx = numpy.zeros((3,3))
rrz = numpy.zeros((3,3))

rrx[0][0] = costhrx
rrx[1][1] = costhrx
rrx[0][1] = sinthrx
rrx[1][0] = -sinthrx
rrx[2][2] = 1.0e0

rrz[0][0] = costhrz
rrz[2][2] = costhrz
rrz[0][2] = -sinthrz
rrz[2][0] = sinthrz
rrz[1][1] = 1.0e0

posr_new_pro = numpy.dot(numpy.dot(rrz,rrx),posr)

# translate along the Z-axis so that a midpoint between donor and acceptor is at the origin
shift = posr_new_pro[2][rac]/2
for i in range(natom):
   posr_new_pro[2][i] = posr_new_pro[2][i] - shift

fr = open(proali,"w")
fr.write(str(natom)+"\n")
fr.write("Product with D and A atoms along the Z-axis and midpoint at (0,0,0): DA distance %10.6f \u212B\n" % pda_distance)
for i in range(natom):
   s = "%-2s %15.8f %15.8f %15.8f\n" % (catom[i],posr_new_pro[0][i],posr_new_pro[1][i],posr_new_pro[2][i])
   fr.write(s)
fr.close()

# Check if DA distances are the same in the reactant and product structures

if abs(rda_distance - pda_distance) > 10**(-6):
   print("\nThe donor acceptor distances in the reactant and product structures are different: " + str(rda_distance) + " and " + str(pda_distance))
   print("Donor and acceptor atoms are along the Z-axis and the midpoint is aligned)\n")
   print("Average structure is labeled by the DA distance in the reactant configuration\n")

da_distance = rda_distance

###### Rotate around Z-axis then minimize RMSD ########

# Reading in coordinates #
rea_file = open(reanew,"r")
pro_file = open(proali,"r")

rea_lines = rea_file.readlines()
pro_lines = pro_file.readlines()

natom = int(rea_lines[0])
#natom = len(rea_lines)

rea = numpy.zeros((3,natom))
pro = numpy.zeros((3,natom))
catom = []
pro_new = numpy.zeros((3,natom))
for i in range(natom):
   xyz = rea_lines[i+2].split()
   catom.append(xyz[0])
   rea[0][i] = float(xyz[1])
   rea[1][i] = float(xyz[2])
   rea[2][i] = float(xyz[3])
rea_file.close()

for i in range(natom):
   xyz = pro_lines[i+2].split()
   catom.append(xyz[0])
   pro[0][i] = float(xyz[1])
   pro[1][i] = float(xyz[2])
   pro[2][i] = float(xyz[3])
pro_file.close()


# RMSD function
def rmsdfun(x):
   e = 0
   deg_rad = x
   costhrz = math.cos(deg_rad)
   sinthrz = math.sin(deg_rad)
   rrz[0][2] = 0
   rrz[1][2] = 0
   rrz[2][0] = 0
   rrz[2][1] = 0
   rrz[0][0] = costhrz
   rrz[2][2] = 1.0e0
   rrz[1][0] = -sinthrz
   rrz[0][1] = sinthrz
   rrz[1][1] = costhrz
   pro_new = numpy.dot(rrz,pro)
   for k in range(natom):
      a = (rea[0][k]-pro_new[0][k])*(rea[0][k]-pro_new[0][k])
      b = (rea[1][k]-pro_new[1][k])*(rea[1][k]-pro_new[1][k])
      c = (rea[2][k]-pro_new[2][k])*(rea[2][k]-pro_new[2][k])
      d = a + b + c
      e = d + e
   return math.sqrt((1.0/float(natom))*e)

# minimize RMSD and rotate

#RMSD_MAX = 10**10
#irot = 0
#for i in range (3600):
#   e = 0
#   deg_rad = -i*radians/10
#   costhrz = math.cos(deg_rad)
#   sinthrz = math.sin(deg_rad)
#   rrz[0][2] = 0
#   rrz[1][2] = 0
#   rrz[2][0] = 0
#   rrz[2][1] = 0
#   rrz[0][0] = costhrz
#   rrz[2][2] = 1.0e0
#   rrz[1][0] = -sinthrz
#   rrz[0][1] = sinthrz
#   rrz[1][1] = costhrz
#   pro_new = numpy.dot(rrz,pro)
#
# RMSD
#
#   for k in range(natom):
#      a = (rea[0][k]-pro_new[0][k])*(rea[0][k]-pro_new[0][k])
#      b = (rea[1][k]-pro_new[1][k])*(rea[1][k]-pro_new[1][k])
#      c = (rea[2][k]-pro_new[2][k])*(rea[2][k]-pro_new[2][k])
#      d = a + b + c
#      e = d + e
#
#   RMSD = math.sqrt((1.0/float(natom))*e)
#
#   rmsd = "RMSD: %12.6f      Rotation angle: %12.6f degrees" % (RMSD, -float(i)/float(10))
#   print(rmsd)
#
#   if RMSD<RMSD_MAX:
#      RMSD_MAX = RMSD
#      irot = i
#
#deg_rad = -irot*radians/10

minimization = minimize_scalar(rmsdfun, bounds=(0, 2*math.pi), method='bounded')
rmsd_min = minimization.fun
deg_rad = minimization.x

print("\nMinimum RMSD: %12.6f   Rotation angle is %12.6f degrees\n" % (rmsd_min,180*deg_rad/math.pi))

costhrz = math.cos(deg_rad)
sinthrz = math.sin(deg_rad)

rrz[0][2] = 0
rrz[1][2] = 0
rrz[2][0] = 0
rrz[2][1] = 0
rrz[0][0] = costhrz
rrz[2][2] = 1.0e0
rrz[1][0] = -sinthrz
rrz[0][1] = sinthrz
rrz[1][1] = costhrz

pro_new = numpy.dot(rrz,pro)

fr = open(prorot,"w")

#print(RMSD)
#rmsd = "\nMin RMSD: %12.6f      Rotation angle: %12.6f degrees\n" % (RMSD, -float(irot)/float(10))
#print(rmsd)

fr.write(str(natom)+"\n")
fr.write("Product with D and A atoms along the Z-axis and aligned with reactant: DA distance %10.6f \u212B\n" % pda_distance)
for i in range(natom):
   s = "%-2s %15.8f %15.8f %15.8f\n" % (catom[i],pro_new[0][i],pro_new[1][i],pro_new[2][i])
   fr.write(s)
fr.close()


##### Average Structures #######
# open the two .xyz files
frea = open(reanew,"r")
fpro = open(prorot,"r")

# read the lines from the two files
rea_lines = frea.readlines()
natom = int(rea_lines[0])
pro_lines = fpro.readlines()

# close the two files
frea.close()
fpro.close()

# open an output file
output = open(options.output_name,"w")

# split the lines and calculate the average of each coordinate for each atom
numlines = len(rea_lines)

output.write(str(natom)+"\n")
output.write("Average reactant/product configuration: DA distance %10.6f \u212B\n" % da_distance)
for i in range(natom):
   atom  = (rea_lines[i+2].split()[0])
   x_rea = float(rea_lines[i+2].split()[1])
   y_rea = float(rea_lines[i+2].split()[2])
   z_rea = float(rea_lines[i+2].split()[3])
   x_pro  = float(pro_lines[i+2].split()[1])
   y_pro  = float(pro_lines[i+2].split()[2])
   z_pro  = float(pro_lines[i+2].split()[3])
   x_avg = (x_rea + x_pro)/2
   y_avg = (y_rea + y_pro)/2
   z_avg = (z_rea + z_pro)/2
   output.write("%-2s %15.8f %15.8f %15.8f\n"%(atom,x_avg,y_avg,z_avg))

output.close()
