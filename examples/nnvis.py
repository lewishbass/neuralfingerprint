

import pygame
import numpy as np
from molvis import Molvis
from neuralfingerprint import MolFrom
import time
import pickle

def lerp(x1, x2, k):
    return x1+(x2-x1)*k

def ilerp(x1, x2, k):
    return (k-x1)/(x2-x1)

def mapRange(x1, x2, y1, y2, k):
    x1, x2, y1, y2, k = float(x1), float(x2), float(y1), float(y2), float(k)
    return lerp(y1, y2, ilerp(x1, x2, k))

class NNvis(object):
    
    def __init__(self, size=[500, 500], freeSolve = pickle.load(open('consol.pickle2', 'rb'))):
        pygame.init()
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.size = self.screen.get_size()
        self.running = True
        self.colors = [(252, 186, 3), (41, 181, 36), (22, 135, 135), (55, 67, 222), (112, 18, 140), (176, 21, 37)]
        self.uicolors = [(35, 35, 40), (20, 25, 36), (40, 46, 51), (53, 65, 80)]
        self.b = 5.0

        self.mvis = Molvis((self.size[0]-2*self.b, self.size[1]/2.0-1.5*self.b), (self.b, self.b), MolFrom('mobley_2661134'))
        self.t = time.time()
        self.ml = np.array(pygame.mouse.get_pos())
        self.free = freeSolve
        self.indx = 0
        self.data = np.array([])#np.array([[0, 1, 2], [1000, 2, 3], [2000, 3, 4], [3000, 3, 4], [4000, 2, 3], [5000, 1, 2]])

    def draw(self):
        ft = time.time()-self.t
        self.t = time.time()

        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                self.running = False
                print("closed by user")
                quit()
            if(event.type == pygame.VIDEORESIZE):
                self.size = self.screen.get_size()
                self.mvis.resize((self.size[0]-2*self.b, self.size[1]/2.0-1.5*self.b), (self.b, self.b))
            if(event.type == pygame.MOUSEWHEEL):
                #self.b = self.b + event.y
                #self.mvis.resize((self.size[0]-2*self.b, self.size[1]/2.0-1.5*self.b), (self.b, self.b))
                self.indx = self.indx + event.y
                i = self.free.keys()[self.indx%len(self.free.keys())]
                self.mvis.reMol(i, self.free[i])

        mn = np.array(pygame.mouse.get_pos())
        if(pygame.mouse.get_pressed()[0]):
            self.mvis.rotMol((mn-self.ml)[0], (mn-self.ml)[1])
        self.ml = mn

        self.screen.fill(self.uicolors[0])

        
        n = 0#100
        for i in range(n):
            k = i/float(n)
            #k = (k-(k%0.1))*(-k%0.1+0.4)*2
            pygame.draw.circle(self.screen, tuple(np.array((1, 1, 1))*255*k), tuple(np.array(self.size)/2), (n-i)*(min(self.size)/(n*2.0)))

        f = [self.b, self.b, self.size[0]-2*self.b, self.size[1]/2-1.5*self.b]
        pygame.draw.rect(self.screen, self.uicolors[2], pygame.Rect(f[0], f[1], f[2], f[3]))
        
        self.mvis.draw(self.screen, self.font, ft)

        f = [self.b, self.size[1]/2+self.b/2, self.size[0]-2*self.b, self.size[1]/2-1.5*self.b]
        pygame.draw.rect(self.screen, self.uicolors[1], pygame.Rect(f[0], f[1], f[2], f[3]))
        if(self.data.size<=0):
            return
        gf = [min(self.data[:,0]), max(self.data[:,0]), min(0, 1.05*np.min(self.data[:, 1:])), max(0, 1.05*np.max(self.data[:, 1:]))]
        
        for i in range(self.data.shape[1]-1):
            for j in range(self.data.shape[0]-1):
                pygame.draw.line(self.screen, self.colors[i%len(self.colors)], 
                    (mapRange(gf[0], gf[1], f[0], f[0]+f[2], self.data[j  , 0]), mapRange(gf[2], gf[3], f[1]+f[3], f[1], self.data[j  , i+1])),
                    (mapRange(gf[0], gf[1], f[0], f[0]+f[2], self.data[j+1, 0]), mapRange(gf[2], gf[3], f[1]+f[3], f[1], self.data[j+1, i+1])))
                #pygame.draw.circle(self.screen, self.colors[i], (mapRange(gf[0], gf[1], f[0], f[0]+f[2], self.data[j+1, 0]), mapRange(gf[2], gf[3], f[1]+f[3], f[1], self.data[j+1, i+1])), 3)

        text = []
        text.append("iteration: " + str(self.data[0, 0])[:5])
        text.append("test   RMSD: " + str(self.data[0, 2])[:5])
        text.append("train RMSD: " + str(self.data[0, 1])[:5])
        for i in range(len(text)):
            surf = self.font.render(text[i], True, (230, 230, 230))
            rect = surf.get_rect()
            rect.topleft = (f[0]+15, f[1]+15+20*i)
            self.screen.blit(surf, rect)

        pygame.display.flip()

    def graph(self, data):
        if(self.data.size<=0):
            self.data = np.array([np.array(data)])
            return
        self.data = np.vstack((np.array(data), self.data))

    def kill(self):
        pygame.quit()