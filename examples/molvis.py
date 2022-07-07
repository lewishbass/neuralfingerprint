
from neuralfingerprint import Atom, Bond, Mol
from neuralfingerprint.mol import MolFrom
import pygame
from pygame import gfxdraw
import numpy as np
from numpy import pi
import colorsys

def rot(t):
    r = np.matrix([ [t+pi/2.0,t+pi    ],
                    [t       ,t+pi/2.0]])
    return np.sin(r)

def roty(t):
    r = np.matrix([ [t+pi/2.0, 0    , t+pi    ],
                    [0       , pi/2    , 0       ],
                    [t       , 0    , t+pi/2.0]])
    return np.sin(r)

def rotx(t):
    return roty(t)[[1, 0, 2], :][:, [1, 0, 2]]

def rotz(t):
    return roty(t)[[0, 2, 1], :][:, [0, 2, 1]]

def lowRes(x, n):
    if(type(x) == tuple):
        c = tuple()
        for i in x:
            c = c+(i-i%n,)
        return c
    return x-x%n



class Molvis(object):
    def __init__(self, size, pos, mol):
        self.size = size
        self.pos = pos
        self.mol = mol
        self.crot = np.eye(3)*1.0
        self.mrot = np.eye(3)*1.0
        self.disp = np.matrix([0, 0, 10]).T
        self.f = max(self.size)
        #self.res = 150

        self.molParams = {u'BGB+group': u'train', u'calc_vdw': 0.378, 'key': u'mobley_2661134', u'calc_h': -19.705390842926136, u'd_vdw': 0.025, u'PubChemID': 13394, u'd_expt': 0.6, u'calc_reference': u'10.1021/acs.jced.7b00104', u'expt_h_reference': u'Not available', u'd_calc_s (cal/mol.K)': 2.3716068155221346, u'd_calc_h': 0.7064578783053079, u'h_solv': -19.97610033640596, u'expt_reference': u'10.1021/ct050097l', u'd_calc': 0.03, u'calc_s (cal/mol.K)': -40.13547155098486, u'd_h_conf': 0.05025848554606732, u'd_h_solv': 0.708194435513733, u'd_expt_s (cal/K.mol)': u'Not available', u'smiles': u'c1cc(cc(c1)O)C#N', u'groups': [u'phenol or hydroxyhetarene', u'carbonitrile', u'aromatic'], u'h_conf': 0.26622789720896745, u'nickname': u' 3-hydroxybenzonitrile', u'd_charging': 0.017, u'calc': -7.739, u'calc_charging': -8.117, u'notes': [u'Experimental uncertainty not presently available, so assigned a default value.'], u'iupac': u'3-hydroxybenzonitrile', u'BGB+': -8.53, u'expt': -9.65, u'expt_s (cal/K.mol)': u'Not available', u'expt_h': u'Not available', u'd_expt_h': u'Not available'}


        self.hideH = False
    
    def resize(self, size, pos):
        self.pos = pos
        self.size = size
        self.f = max(self.size)

    def reMol(self, id, params):
        self.mol = MolFrom(id)
        self.molParams = params

    def rotMol(self, x, y):
        x, y = 5.0*float(x)/self.f, 5.0*float(y)/self.f
        #self.mrot = np.matmul(self.mrot, rotx(y))
        #self.mrot = np.matmul(self.mrot, roty(x))
        self.mrot = np.matmul(rotx(y), self.mrot)
        self.mrot = np.matmul(roty(x), self.mrot)

    def transform(self, a, center):
        a = a-center
        a = np.matmul(self.mrot, a)
        a = a+self.disp
        a = np.matmul(self.crot, a)
        return a
    
    def getMol(self):
        return self.mol

    def draw(self, screen, font, ft):
        #print(pygame.time.get_ticks())
        #self.disp = np.matrix([0, 2*np.sin(pygame.time.get_ticks()/1000.0), 0.005]).T
        #self.mrot = np.matmul(roty(pygame.time.get_ticks()/4000.0), rotx(pygame.time.get_ticks()/2800.0))
        self.mrot = np.matmul(self.mrot, roty(1.2*0.2*ft))
        self.mrot = np.matmul(self.mrot, rotx(1.2*0.07*ft))
        self.mrot = np.matmul(self.mrot, rotz(1.2*0.11*ft))

        #self.mrot = np.matmul(roty(0.75*0.2 *ft), self.mrot)
        #self.mrot = np.matmul(rotx(0.75*0.07*ft), self.mrot)
        #self.mrot = np.matmul(rotz(0.75*0.11*ft), self.mrot)

        atoms = []
        for i in self.mol.GetAtoms():
            atoms.append(i.GetPos())
        atoms = np.matrix(atoms).T
        center = np.mean(atoms, 1)

        zbuff = list(np.argsort(self.transform(atoms, center)[2,:].tolist()[0]))
        #zbuff = list(np.arange(len(self.transform(atoms, center)[2,:].tolist()[0])))
        zbuff.reverse()
        #print(zbuff)
        bc = screen.get_at((int(self.pos[0]+self.size[0]/2), int(self.pos[1]+self.size[1]/2)))
        
        for i in self.mol.GetBonds():
            if(self.hideH and (i.GetBeginAtom().GetSymbol() == 'H' or i.GetEndAtom().GetSymbol() == 'H')):
                continue
            a = np.hstack((np.matrix(i.GetBeginAtom().GetPos()).T, np.matrix(i.GetEndAtom().GetPos()).T))
            a = self.transform(a, center)
            if(a[2, 0]>0 or a[2, 1]>0):
                b = (self.f*a[0, 0]/a[2, 0]+self.size[0]/2.0+self.pos[0], self.f*a[1, 0]/a[2, 0]+self.size[1]/2.0+self.pos[1])
                c = (self.f*a[0, 1]/a[2, 1]+self.size[0]/2.0+self.pos[0], self.f*a[1, 1]/a[2, 1]+self.size[1]/2.0+self.pos[1])
                #b, c = lowRes(b, self.f/self.res), lowRes(c, self.f/self.res)
                if(i.GetBondType() == 2):
                    pygame.draw.line(screen, (200, 220, 200), b, c, int(0.4*0.6*self.f/np.mean(a[2,:])))
                    pygame.draw.line(screen, bc, b, c, int(0.1*0.6*self.f/np.mean(a[2,:])))
                else:
                    pygame.draw.line(screen, (200, 220, 200), b, c, int(0.2*0.6*self.f/np.mean(a[2,:])))

        for z in zbuff:
            i = self.mol.GetAtoms()[z]
            if(i.GetSymbol() == 'H' and self.hideH):
                continue
            a = np.matrix(i.GetPos()).T
            a = self.transform(a, center)
            if(a[2, 0]>0.1):
                b = (self.f*a[0, 0]/a[2, 0]+self.size[0]/2.0+self.pos[0], self.f*a[1, 0]/a[2, 0]+self.size[1]/2.0+self.pos[1])
                #b = lowRes(b, self.f/self.res)
                #col = colorsys.hsv_to_rgb(0.5, 1, 1)
                #col = (col[0]*255.0, col[1]*255.0, col[2]*255.0)
                col = ( 255.0-200.0*min(1.0, max(0.0, i.GetCharge())), 
                        250.0-200.0*min(1.0, np.abs(i.GetCharge())), 
                        255.0+200.0*max(-1.0, min(0.0, i.GetCharge())))
                pygame.draw.circle(screen, col, b, 0.25*self.f/a[2, 0]*i.GetRadii())
                sym = font.render(i.GetSymbol(), False, (0, 0, 0))
                rect = sym.get_rect()
                sym = pygame.transform.scale(sym, (0.015*rect.size[0]*self.f/a[2,0]*i.GetRadii(), 0.015*rect.size[1]*self.f/a[2,0]*i.GetRadii()))
                rect = sym.get_rect()
                rect.center=b
                screen.blit(sym, rect)

        text = []
        text.append("id:  " + str(self.molParams['key']))
        text.append("name:  " + str(self.molParams['iupac']))
        text.append("smiles:  " + str(self.molParams['smiles']))
        text.append("solv:  " + str(self.molParams['expt']))
        if('BGB+' in self.molParams.keys()):
            text.append("BGB+:  " + str(self.molParams['BGB+'])[:5])
            #text.append("Group:  " + str(self.molParams['BGB+group']))
        if('ML' in self.molParams.keys()):
            text.append("ML:  " + str(self.molParams['ML'])[:5])
            text.append("Group:  " + str(self.molParams['MLgroup'])[:5])

        ts = np.sqrt(np.prod(self.size))/1000.0
        for i in range(len(text)):
            surf = font.render(text[i], True, (230, 230, 230))
            rect = surf.get_rect()
            surf = pygame.transform.smoothscale(surf, (ts*rect.size[0], ts*rect.size[1]))
            rect = sym.get_rect()
            rect.topleft = (ts*(15+self.pos[0]), ts*(15+20*i+self.pos[1]))
            screen.blit(surf, rect)

        #for i in self.mol.GetBonds():

