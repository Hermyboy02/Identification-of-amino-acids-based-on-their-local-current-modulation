from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

amino = input("input amino acid: ")


mol_supplier = Chem.SDMolSupplier(amino + ".sdf", removeHs=False) # Skriv in vilken aminosyra man vill ha
mol = mol_supplier[0]
AllChem.ComputeGasteigerCharges(mol)
conf = mol.GetConformer()
positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
geometric_center = positions.mean(axis=0)

print("Geometric center (x, y, z):", geometric_center)

# Shift all atoms: subtract geometric center from each atom position
for i in range(mol.GetNumAtoms()):
    pos = conf.GetAtomPosition(i)
    new_pos = pos - geometric_center
    conf.SetAtomPosition(i, new_pos)

# Save the centered molecule
Chem.MolToMolFile(mol, amino + "_centered.sdf") # Skriv in vad den nya filen ska heta

