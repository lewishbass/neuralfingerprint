
import pandas as pd
import numpy as np
from rdkit.Chem import rdchem, MolFromSmiles

class Atom(object):
    # RDkit basis: GetIdx GetSymbol GetDegree GetTotalNimHs GetImplicitValence
    # Additional:  GetCharge Get(Ri) GetPolarizability
    def __init__(self, indx, sym, pos=[0, 0, 0], charge=0, ri=0, radii = 0):
        self.indx = indx
        self.sym = sym
        self.pos = pos
        self.charge = charge
        self.ri = ri
        self.radii = radii

    def GetIdx(self):
        return self.indx

    def GetSymbol(self):
        return self.sym
    
    #def GetDegree(self):
    #   return self.degree

    #def GetTotalNumHs(self):
    #   return self.numH

    #def GetImplicitValence(self):
    #   return self.iv
        
    def GetCharge(self):
        return self.charge

    def GetRadii(self):
        return self.radii

    def GetRi(self):
        return self.ri
    
    def GetPos(self):
        return self.pos
    
    def GetRadii(self):
        return self.radii
    
class Bond(object):
    # RDkit basis: GetBeginAtom GetEndAtom GetBondType GetIsConjugated IsInRing
    # Additional: 
    def __init__(self, atom1, atom2, bondtype):
        self.atom1 = atom1
        self.atom2 = atom2
        self.bondtype = bondtype
        #self.conjugated = conjugated
        #self.inring = inring

    def GetBeginAtom(self):
        return self.atom1

    def GetEndAtom(self):
        return self.atom2

    def GetBondType(self):
        return self.bondtype

    #def GetIsConjugated(self):
    #    return self.conjugated

    #def IsInRing(self):
     #   return self.inring

class Mol(object):
    # RDkit basis: GetAtoms, GetBonds
    # Additional: 
    def __init__(self):
        self.atoms = []
        self.bonds = []
    
    def addAtom(self, atom):
        self.atoms.append(atom)
    
    def addBond(self, bond):
        self.bonds.append(bond)
    
    def GetAtoms(self):
        return self.atoms
    
    def GetAtomByIdx(self, idx):
        for i in self.atoms:
            if (i.GetIdx() == idx):
                return i
        raise Exception('Missing Atom Index')

    def GetBonds(self):
        return self.bonds


def MolFromPDB(pqrFileName, sdfFileName):

    pq = pd.read_csv(pqrFileName, skiprows=0, header=None, delim_whitespace=True).to_numpy()
    
    f = open(sdfFileName, 'r')

    sd = []
    for x in f:
        sd.append(x)
    
    numAtoms = int(sd[3].split()[0])#num atoms
    numBonds = int(sd[3].split()[1])#num bonds
    b = sd[4+numAtoms:4+numAtoms+numBonds]

    for i in range(len(b)):
        b[i] = b[i].split()
        for j in range(len(b[i])):
            b[i][j] = int(b[i][j])

    
    mol = Mol()
    for i in pq:
        if(i[0] != 'ATOM'):
            break
        mol.addAtom(Atom(int(i[1]), i[10], pos=list(i[5:8]), charge=i[8], radii=i[9]))

    for i in b:
        mol.addBond(Bond(mol.GetAtomByIdx(i[0]), mol.GetAtomByIdx(i[1]), i[2]))

    return mol


def MolFromMobley(id):
    return MolFromPDB('pqrfiles/'+id+'.pqr', 'sdffiles/'+id+'.sdf')\

def MolFrom(x):

    #print("mol from: ", x)

    if(type(x) == Mol or type(x) == rdchem.Mol):
        mol = x
    elif('mobley_' in x):
        mol = MolFromMobley(x)
    else:
        mol = MolFromSmiles(x)

    return mol