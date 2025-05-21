from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import numpy as np
import matplotlib.pyplot as plt
import copy
import csv

"--------------------------------------------------------------------------------"

def rot_vec(mol, rot_mat):
    conf = mol.GetConformer()
    num_at = mol.GetNumAtoms()

    for i in range(num_at):
        pos = np.array(conf.GetAtomPosition(i))
        new_pos = np.matmul(rot_mat,pos)
        conf.SetAtomPosition(i, new_pos)


def rot_mat():
    random_x = np.random.uniform(0, 2*np.pi)
    random_y = np.random.uniform(0, 2*np.pi)
    random_z = np.random.uniform(0, 2*np.pi)

    rot_x = np.array([[1, 0, 0], [0, np.cos(random_x), -np.sin(random_x)], [0, np.sin(random_x), np.cos(random_x)]])
    rot_y = np.array([[np.cos(random_y), 0, np.sin(random_y)],[0,1,0],[-np.sin(random_y), 0, np.cos(random_y)]])
    rot_z = np.array([[np.cos(random_z), -np.sin(random_z), 0],[np.sin(random_z), np.cos(random_z),  0],[0, 0, 1]])

    # Multiply all three matrices to get the total rotation matrix
    return np.matmul(np.matmul(rot_x, rot_y), rot_z)


def pot_field(atom, point, mol):
    e_conv = 1.60217662e-19

    epsilon_0 = 8.854187817e-12
    atom_idx = atom.GetIdx()
    charge = float(atom.GetProp("_GasteigerCharge")) * e_conv
    pos = mol.GetConformer().GetAtomPosition(atom_idx)
    atom_pos = np.array([pos.x, pos.y, pos.z]) * 1e-10
    r_vector = point - atom_pos
    r = np.linalg.norm(r_vector)

    pot =  charge / (4*np.pi*epsilon_0*r)

    return pot

def pot_line(dx, r, atom, mol):
    points_on_line = np.arange(-r + dx/2, r, dx)
    pot_list = []

    for position in points_on_line:
        vector = np.array([position, r, 0])
        pot = pot_field(atom, vector, mol)
        pot_list.append(pot)

    pot_array = np.array(pot_list)
    return pot_array #array med potentialer i olika punkter för en och samma atom.

def pot_tot(mol, dx, r):
    n_intervals = 2*r/dx # Number of intervals

    pot_tot = np.zeros(int(n_intervals)) # Array with total potential from one amino acid
    for atom in mol.GetAtoms():
        pot_tot = pot_tot + pot_line(dx, r, atom, mol)

    return pot_tot


def transcoeff(dx, E_P, E):
    
    e_conv = 1.60217662e-19  
    #print(E_P, E)

    m = 9.10938356 * 10**(-31) #kg
    #h_bar = 6.5821196 * 10**(-16) #eVs
    h_bar = 1.05457182 * 10**(-34) #Js

    alfa = (np.sqrt(2 * m * np.abs(E_P - E))) / h_bar

    if E < E_P:
        return (1 + (E_P**2 * np.sinh(alfa * dx)**2) / (4 * E * (E_P - E)))**(-1) #E < E_P
    else:
        return (1 + (E_P**2 * np.sin(alfa * dx)**2) / (4 * E * (E - E_P)))**(-1) #E > E_P


def transcoeff_tot(pot, dx, E_0):
    T = 1

    for potential in pot:
        T = T * transcoeff(dx, potential, E_0)

    return T

"---------------------------------------------------------------------------------"

def main():
    amino_acid = input("input amino acid: ")
    
    mol_supplier = Chem.SDMolSupplier("Amino_acids_centered_SDF/" + amino_acid + "_centered.sdf", removeHs=False)
    amino = amino_acid
    N_rotations = 40000

    # ---------------------

    mol = mol_supplier[0]
    AllChem.ComputeGasteigerCharges(mol)
    radius = 6e-10
    N_intervals = 100
    dx = 2*radius/N_intervals
    x = np.arange(-radius + dx/2, radius, dx)
    coeffs = np.zeros(N_rotations)
    E_0 = 0.001*1.60217662e-19 #J

      
    orig_mol = Chem.Mol(mol)        

    for i in range(N_rotations):

         
        mol_i = Chem.Mol(orig_mol)

        rotation = rot_mat()
        rot_vec(mol_i, rotation)

        potentials = pot_tot(mol_i, dx, radius) * -1.60217662e-19
        

        T = transcoeff_tot(potentials, dx, E_0)
        coeffs[i] = T


    counts, _ = np.histogram(coeffs, bins=50, range=(0,1))

   
    filename = f"{amino}_{N_rotations}.csv"

    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for count in counts:
            writer.writerow([count])  

       
    plt.hist(coeffs, bins=50, range=(0,1), color = "#pink")
    plt.title(f'{amino} {N_rotations}')

    plt.show()


if __name__ == "__main__":
    main()