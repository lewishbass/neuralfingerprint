import autograd.numpy as np
from rdkit import Chem
from util import one_of_k_encoding, one_of_k_encoding_unk, one_of_k_encoding_float
from mol import Atom, Bond

def atom_features(atom):
    '''a = np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                       'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                       'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
                                       'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                       'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +'''
    '''a = np.array(one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'O', 'Cl', 'P', 'S', 'F', 'N', 'Br', 'I']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()] + 
                    [])'''
    a = np.array(   one_of_k_encoding(atom.GetSymbol(), ['H', 'C', 'O', 'CL', 'P', 'S', 'F', 'N', 'BR', 'I']) + 
                    #one_of_k_encoding_float(atom.GetCharge(), list(np.arange(-1.01, 1.816, 0.7))) )
                    [atom.GetCharge(), atom.GetRadii()])
    return a

def bond_features(bond):
    bt = bond.GetBondType()
    r = np.array(bond.GetBeginAtom().GetPos()) - np.array(bond.GetEndAtom().GetPos())
    r = np.sqrt(np.sum(np.power(r, 2)))
    return np.array(one_of_k_encoding(bt, [0, 1, 2, 3]) + [r] )
    '''return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])'''

def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    #m = Chem.MolFromSmiles('CC')
    #alist = m.GetAtoms()
    #a = alist[0]
    #return len(atom_features(a))
    return len(atom_features(Atom(0, 'C')))

def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    #simple_mol = Chem.MolFromSmiles('CC')
    #Chem.SanitizeMol(simple_mol)
    #return len(bond_features(simple_mol.GetBonds()[0]))
    return len(bond_features(Bond(Atom(0, 'C'), Atom(0, 'C'), 1)))

